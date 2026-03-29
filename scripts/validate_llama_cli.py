#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import shutil
from pathlib import Path


TOKENS_RE = re.compile(r"^\s*prompt eval time =\s+([0-9.]+) ms .*? ([0-9.]+) tokens per second", re.MULTILINE)
GEN_RE = re.compile(r"^\s*eval time =\s+([0-9.]+) ms .*? ([0-9.]+) tokens per second", re.MULTILINE)
TOTAL_RE = re.compile(r"^\s*total time =\s+([0-9.]+) ms", re.MULTILINE)
SIDECAR_RE = re.compile(r"TurboQuant compact K sidecar = ([0-9.]+) MiB vs f16-K ([0-9.]+) MiB \(ratio ([0-9.]+)x, savings ([0-9.\-]+)%\)")
CTX_RE = re.compile(r"^\s*llama_context:\s+n_ctx\s*=\s*([0-9]+)", re.MULTILINE)
KV_SAVINGS_RE = re.compile(r"effective ratio = ([0-9.]+)x \(K ([0-9.]+)x, V ([0-9.]+)x\), savings = ([0-9.\-]+)%")
CPU_KV_RE = re.compile(r"CPU KV buffer size =\s+([0-9.]+) MiB")
CUDA_KV_RE = re.compile(r"CUDA0 KV buffer size =\s+([0-9.]+) MiB")
LAYER_IDS_RE = re.compile(r"TurboQuant exact layer ids = ([0-9,\-<>]+)")
OFFLOAD_RE = re.compile(r"offloaded\s+([0-9]+)/([0-9]+)\s+layers to GPU")


def parse_metrics(text: str) -> dict:
    out = {}
    m = TOKENS_RE.search(text)
    if m:
        out["prompt_ms"] = float(m.group(1))
        out["prompt_tps"] = float(m.group(2))
    m = GEN_RE.search(text)
    if m:
        out["gen_ms"] = float(m.group(1))
        out["gen_tps"] = float(m.group(2))
    m = TOTAL_RE.search(text)
    if m:
        out["total_ms"] = float(m.group(1))
    m = SIDECAR_RE.search(text)
    if m:
        out["sidecar_mib"] = float(m.group(1))
        out["f16_k_mib"] = float(m.group(2))
        out["sidecar_ratio"] = float(m.group(3))
        out["sidecar_savings_pct"] = float(m.group(4))
    m = CTX_RE.search(text)
    if m:
        out["fitted_ctx"] = int(m.group(1))
    m = KV_SAVINGS_RE.search(text)
    if m:
        out["effective_ratio"] = float(m.group(1))
        out["k_ratio"] = float(m.group(2))
        out["v_ratio"] = float(m.group(3))
        out["savings_pct"] = float(m.group(4))
    m = CPU_KV_RE.search(text)
    if m:
        out["cpu_kv_mib"] = float(m.group(1))
    m = CUDA_KV_RE.search(text)
    if m:
        out["cuda0_kv_mib"] = float(m.group(1))
    m = LAYER_IDS_RE.search(text)
    if m:
        out["selected_layer_ids"] = [int(x) for x in m.group(1).split(",") if x.strip().isdigit()]
    m = OFFLOAD_RE.search(text)
    if m:
        out["gpu_layers_offloaded"] = int(m.group(1))
        out["gpu_layers_total"] = int(m.group(2))
    return out


def attach_gains(metrics: dict, baseline: dict) -> dict:
    out = dict(metrics)
    if "prompt_tps" in baseline and "prompt_tps" in out:
        out["prompt_gain_pct"] = 100.0 * (out["prompt_tps"] / baseline["prompt_tps"] - 1.0)
    if "gen_tps" in baseline and "gen_tps" in out:
        out["gen_gain_pct"] = 100.0 * (out["gen_tps"] / baseline["gen_tps"] - 1.0)
    if "total_ms" in baseline and "total_ms" in out:
        out["total_latency_gain_pct"] = 100.0 * (1.0 - out["total_ms"] / baseline["total_ms"])
    if "fitted_ctx" in baseline and "fitted_ctx" in out:
        out["ctx_gain_pct"] = 100.0 * (out["fitted_ctx"] / baseline["fitted_ctx"] - 1.0)
    return out


def objective_score(row: dict, objective: str) -> tuple:
    if not row.get("ok"):
        return (float("-inf"),)
    if objective == "ctx_first":
        return (
            row.get("fitted_ctx", 0),
            row.get("savings_pct", float("-inf")),
            row.get("total_latency_gain_pct", float("-inf")),
            row.get("gen_gain_pct", float("-inf")),
            row.get("prompt_gain_pct", float("-inf")),
        )
    if objective == "latency_first":
        return (
            row.get("total_latency_gain_pct", float("-inf")),
            row.get("gen_gain_pct", float("-inf")),
            row.get("prompt_gain_pct", float("-inf")),
            row.get("savings_pct", float("-inf")),
            row.get("fitted_ctx", 0),
        )
    return (
        row.get("total_latency_gain_pct", float("-inf")),
        row.get("savings_pct", float("-inf")),
        row.get("gen_gain_pct", float("-inf")),
        row.get("prompt_gain_pct", float("-inf")),
        row.get("fitted_ctx", 0),
    )


def profile_definition(name: str, args: argparse.Namespace) -> tuple[str, dict]:
    normalized = name.strip().lower()
    env = build_turboquant_env(args)
    env.pop("LLAMA_ARG_TURBOQUANT_LAYER_LIST", None)
    env.pop("LLAMA_ARG_TURBOQUANT_LAYER_MIN", None)
    env.pop("LLAMA_ARG_TURBOQUANT_LAYER_MAX", None)

    if normalized in {"baseline", "off"}:
        return "baseline", {"LLAMA_ARG_KV_EXPERIMENT": "off"}
    if normalized in {"balanced", "late4"}:
        env["LLAMA_ARG_TURBOQUANT_ROLLOUT"] = "balanced"
        env["LLAMA_ARG_TURBOQUANT_GPU_MAX_LAYERS"] = "4"
        return "balanced", env
    if normalized in {"headroom", "late8"}:
        env["LLAMA_ARG_TURBOQUANT_ROLLOUT"] = "headroom"
        env["LLAMA_ARG_TURBOQUANT_GPU_MAX_LAYERS"] = "8"
        return "headroom", env
    if normalized in {"fast150k", "speed150k", "latency150k"}:
        env["LLAMA_ARG_TURBOQUANT_RECIPE"] = "lab_context_fast"
        env["LLAMA_ARG_TURBOQUANT_V_TYPE"] = "q8_0"
        env["LLAMA_ARG_TURBOQUANT_LAYER_LIST"] = "51,55"
        env.pop("LLAMA_ARG_TURBOQUANT_ROLLOUT", None)
        env.pop("LLAMA_ARG_TURBOQUANT_GPU_MAX_LAYERS", None)
        return "fast150k", env
    if normalized in {"late_sparse", "sparse"}:
        env["LLAMA_ARG_TURBOQUANT_ROLLOUT"] = "late_sparse"
        return "late_sparse", env
    if normalized in {"late_dense", "dense"}:
        env["LLAMA_ARG_TURBOQUANT_ROLLOUT"] = "late_dense"
        return "late_dense", env
    if normalized in {"full", "full_gpu", "full_layers"}:
        env["LLAMA_ARG_TURBOQUANT_ROLLOUT"] = "full_gpu"
        return "full_gpu", env
    if normalized in {"memory_hybrid", "memhybrid", "hybrid_memory", "memory-balanced"}:
        env["LLAMA_ARG_TURBOQUANT_RECIPE"] = "lab_context_fast"
        env["LLAMA_ARG_TURBOQUANT_V_TYPE"] = "q8_0"
        env["LLAMA_ARG_TURBOQUANT_ROLLOUT"] = "memory_hybrid"
        env["LLAMA_ARG_TURBOQUANT_GPU_MAX_LAYERS"] = "8"
        env["LLAMA_ARG_TURBOQUANT_CPU_MAX_LAYERS"] = "4"
        return "memory_hybrid", env
    if normalized in {"memory_gpu_only", "memory-gpu-only", "memory_max_gpu", "memory-max-gpu", "memory_gpu", "memory-gpu"}:
        env["LLAMA_ARG_TURBOQUANT_RECIPE"] = "memory_max"
        env["LLAMA_ARG_TURBOQUANT_V_TYPE"] = "q4_0"
        env["LLAMA_ARG_TURBOQUANT_ROLLOUT"] = "memory_gpu_only"
        env["LLAMA_ARG_TURBOQUANT_GPU_MAX_LAYERS"] = "8"
        env.pop("LLAMA_ARG_TURBOQUANT_CPU_MAX_LAYERS", None)
        return "memory_gpu_only", env
    if normalized in {"memory_max", "memmax", "max_memory", "memory-first"}:
        env["LLAMA_ARG_TURBOQUANT_RECIPE"] = "memory_max"
        env["LLAMA_ARG_TURBOQUANT_V_TYPE"] = "q8_0"
        env["LLAMA_ARG_TURBOQUANT_ROLLOUT"] = "memory_max"
        env.pop("LLAMA_ARG_TURBOQUANT_GPU_MAX_LAYERS", None)
        return "memory_max", env
    raise ValueError(f"unknown profile: {name}")


def run_case(
    bin_path: str,
    model: str,
    prompt: str,
    output: Path,
    env: dict,
    n_predict: int,
    n_gpu_layers: str,
    flash_attn: bool,
    ctx_size: int | None,
    fit_enabled: bool,
    batch_size: int | None,
    ubatch_size: int | None,
) -> dict:
    full_env = os.environ.copy()
    full_env.update(env)
    cmd = [
        bin_path,
        "-m", model,
        "-ngl", n_gpu_layers,
        "-fa", "1" if flash_attn else "0",
        "-n", str(n_predict),
        "-p", prompt,
        "-st",
        "--simple-io",
        "--log-verbosity", "3",
        "--no-warmup",
    ]
    if batch_size is not None:
        cmd.extend(["-b", str(batch_size)])
    if ubatch_size is not None:
        cmd.extend(["-ub", str(ubatch_size)])
    if ctx_size is not None:
        cmd.extend(["-c", str(ctx_size)])
    cmd.extend(["--fit", "on" if fit_enabled else "off"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as fh:
        proc = subprocess.run(cmd, env=full_env, stdout=fh, stderr=subprocess.STDOUT, check=False)
    text = output.read_text(errors="replace")
    metrics = parse_metrics(text)
    metrics["log"] = str(output)
    metrics["env"] = env
    metrics["exit_code"] = proc.returncode
    metrics["ok"] = proc.returncode == 0
    if proc.returncode != 0:
        tail = "\n".join(text.splitlines()[-20:])
        metrics["error_tail"] = tail
    return metrics


def run_json_tool(cmd: list[str], output: Path, env: dict | None = None) -> dict:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as fh:
        proc = subprocess.run(cmd, env=full_env, stdout=fh, stderr=subprocess.STDOUT, check=False)
    text = output.read_text(errors="replace")
    result = {"log": str(output), "exit_code": proc.returncode, "ok": proc.returncode == 0}
    if proc.returncode != 0:
        result["error_tail"] = "\n".join(text.splitlines()[-20:])
        return result
    json_start = text.find("{")
    json_end = text.rfind("}")
    if json_start == -1 or json_end == -1 or json_end < json_start:
        result["ok"] = False
        result["error_tail"] = "tool did not emit JSON payload"
        return result
    try:
        payload = json.loads(text[json_start:json_end + 1])
        result.update(payload)
    except json.JSONDecodeError as exc:
        result["ok"] = False
        result["error_tail"] = f"failed to parse JSON payload: {exc}"
    return result


def build_turboquant_env(args: argparse.Namespace, *, layer_list: str | None = None, layer_min: int | None = None, layer_max: int | None = None, recipe: str | None = None) -> dict:
    env = {
        "LLAMA_ARG_KV_EXPERIMENT": "turboquant_exact",
        "LLAMA_ARG_TURBOQUANT_CACHE": "1",
    }
    effective_recipe = recipe or args.recipe
    if effective_recipe:
        env["LLAMA_ARG_TURBOQUANT_RECIPE"] = effective_recipe
    if args.bits is not None:
        env["LLAMA_ARG_TURBOQUANT_BITS"] = str(args.bits)
    if args.qjl_dim is not None:
        env["LLAMA_ARG_TURBOQUANT_QJL_DIM"] = str(args.qjl_dim)
    if args.outliers is not None:
        env["LLAMA_ARG_TURBOQUANT_OUTLIERS"] = str(args.outliers)
    if args.metadata_path:
        env["LLAMA_ARG_TURBOQUANT_METADATA_PATH"] = args.metadata_path
    if args.v_type:
        env["LLAMA_ARG_TURBOQUANT_V_TYPE"] = args.v_type
    if getattr(args, "rollout", None):
        env["LLAMA_ARG_TURBOQUANT_ROLLOUT"] = args.rollout
    if getattr(args, "gpu_max_layers", None) is not None:
        env["LLAMA_ARG_TURBOQUANT_GPU_MAX_LAYERS"] = str(args.gpu_max_layers)
    if layer_list:
        env["LLAMA_ARG_TURBOQUANT_LAYER_LIST"] = layer_list
    else:
        effective_layer_min = layer_min if layer_min is not None else getattr(args, "layer_min", None)
        effective_layer_max = layer_max if layer_max is not None else getattr(args, "layer_max", None)
        if effective_layer_min is not None:
            env["LLAMA_ARG_TURBOQUANT_LAYER_MIN"] = str(effective_layer_min)
        if effective_layer_max is not None:
            env["LLAMA_ARG_TURBOQUANT_LAYER_MAX"] = str(effective_layer_max)
    return env


def compare(args: argparse.Namespace) -> None:
    base_log = Path(args.output_dir) / "baseline.log"
    tq_log = Path(args.output_dir) / "turboquant.log"

    baseline = run_case(
        args.bin,
        args.model,
        args.prompt,
        base_log,
        {"LLAMA_ARG_KV_EXPERIMENT": "off"},
        args.n_predict,
        args.n_gpu_layers,
        args.flash_attn,
        args.ctx_size,
        args.fit,
        args.batch_size,
        args.ubatch_size,
    )

    tq_env = build_turboquant_env(
        args,
        layer_list=args.layer_list,
        layer_min=args.layer_min,
        layer_max=args.layer_max,
    )
    turboquant = run_case(
        args.bin,
        args.model,
        args.prompt,
        tq_log,
        tq_env,
        args.n_predict,
        args.n_gpu_layers,
        args.flash_attn,
        args.ctx_size,
        args.fit,
        args.batch_size,
        args.ubatch_size,
    )

    turboquant = attach_gains(turboquant, baseline)
    report = {"baseline": baseline, "turboquant": turboquant}
    for key in ("prompt_gain_pct", "gen_gain_pct", "total_latency_gain_pct", "ctx_gain_pct"):
        if key in turboquant:
            report[key] = turboquant[key]

    print(json.dumps(report, indent=2))


def compare_profiles(args: argparse.Namespace) -> None:
    base_log = Path(args.output_dir) / "baseline.log"
    baseline = run_case(
        args.bin,
        args.model,
        args.prompt,
        base_log,
        {"LLAMA_ARG_KV_EXPERIMENT": "off"},
        args.n_predict,
        args.n_gpu_layers,
        args.flash_attn,
        args.ctx_size,
        args.fit,
        args.batch_size,
        args.ubatch_size,
    )

    profile_names = [item.strip() for item in args.profiles.split(",") if item.strip()]
    profiles = []
    for profile_name in profile_names:
        resolved_name, env = profile_definition(profile_name, args)
        log = Path(args.output_dir) / f"{resolved_name}.log"
        metrics = run_case(
            args.bin,
            args.model,
            args.prompt,
            log,
            env,
            args.n_predict,
            args.n_gpu_layers,
            args.flash_attn,
            args.ctx_size,
            args.fit,
            args.batch_size,
            args.ubatch_size,
        )
        metrics["profile"] = resolved_name
        profiles.append(attach_gains(metrics, baseline))

    profiles.sort(key=lambda row: objective_score(row, args.objective), reverse=True)
    recommended = profiles[0] if profiles else None
    recommendation_env = recommended["env"] if recommended and recommended.get("ok") else None

    print(json.dumps({
        "objective": args.objective,
        "baseline": baseline,
        "profiles": profiles,
        "recommended": recommended,
        "recommended_env": recommendation_env,
    }, indent=2))


def promotion_check(args: argparse.Namespace) -> None:
    bin_dir = Path(args.bin).resolve().parent
    bench_bin = str((bin_dir / "llama-turboquant-bench").resolve())
    probe_bin = str((bin_dir / "llama-turboquant-probe").resolve())
    report_dir = Path(args.output_dir)

    bench_cmd = [
        bench_bin,
        "--dim", str(args.synthetic_dim),
        "--bits", str(args.synthetic_bits),
        "--qjl-dim", str(args.synthetic_qjl_dim),
        "--samples", str(args.synthetic_samples),
        "--queries", str(args.synthetic_queries),
    ]
    bench = run_json_tool(bench_cmd, report_dir / "synthetic_bench.log")

    probe_cmd = [
        probe_bin,
        "-m", args.model,
        "-p", args.prompt,
        "--layer", str(args.probe_layer),
        "--bits", str(args.synthetic_bits),
        "--qjl-dim", str(args.synthetic_qjl_dim),
        "--max-vectors", str(args.probe_max_vectors),
        "--max-prompt-tokens", str(args.probe_max_prompt_tokens),
        "--probe-ctx", str(args.probe_ctx),
    ]
    if args.probe_cpu_only:
        probe_cmd.append("--cpu-only")
    probe = run_json_tool(probe_cmd, report_dir / "real_model_probe.log")

    baseline = run_case(
        args.bin,
        args.model,
        args.prompt,
        report_dir / "baseline.log",
        {"LLAMA_ARG_KV_EXPERIMENT": "off"},
        args.n_predict,
        args.n_gpu_layers,
        args.flash_attn,
        args.ctx_size,
        args.fit,
        args.batch_size,
        args.ubatch_size,
    )

    profiles = []
    for profile_name in [item.strip() for item in args.profiles.split(",") if item.strip()]:
        resolved_name, env = profile_definition(profile_name, args)
        metrics = run_case(
            args.bin,
            args.model,
            args.prompt,
            report_dir / f"{resolved_name}.log",
            env,
            args.n_predict,
            args.n_gpu_layers,
            args.flash_attn,
            args.ctx_size,
            args.fit,
            args.batch_size,
            args.ubatch_size,
        )
        metrics["profile"] = resolved_name
        profiles.append(attach_gains(metrics, baseline))

    profiles.sort(key=lambda row: objective_score(row, args.objective), reverse=True)
    recommended = profiles[0] if profiles else None
    recommendation_env = recommended["env"] if recommended and recommended.get("ok") else None

    print(json.dumps({
        "objective": args.objective,
        "baseline": baseline,
        "profiles": profiles,
        "recommended": recommended,
        "recommended_env": recommendation_env,
        "synthetic_bench": bench,
        "real_model_probe": probe,
    }, indent=2))


def sweep_layers(args: argparse.Namespace) -> None:
    results = []
    for layer in range(args.layer_start, args.layer_end + 1):
        log = Path(args.output_dir) / f"layer_{layer}.log"
        env = build_turboquant_env(args, layer_min=layer, layer_max=layer)
        metrics = run_case(args.bin, args.model, args.prompt, log, env, args.n_predict, args.n_gpu_layers, args.flash_attn, args.ctx_size, args.fit, args.batch_size, args.ubatch_size)
        metrics["layer"] = layer
        results.append(metrics)

    results.sort(key=lambda row: (-row.get("gen_tps", 0.0), -row.get("prompt_tps", 0.0)))
    print(json.dumps({"results": results, "recommended": results[0] if results else None}, indent=2))


def autotune(args: argparse.Namespace) -> None:
    recipes = [item.strip() for item in args.recipes.split(",") if item.strip()]
    layer_candidates = []
    if args.layer_lists:
        layer_candidates.extend(item.strip() for item in args.layer_lists.split(";") if item.strip())
    else:
        layer_candidates.extend(str(layer) for layer in range(args.layer_start, args.layer_end + 1))

    baseline_log = Path(args.output_dir) / "baseline.log"
    baseline = run_case(
        args.bin,
        args.model,
        args.prompt,
        baseline_log,
        {"LLAMA_ARG_KV_EXPERIMENT": "off"},
        args.n_predict,
        args.n_gpu_layers,
        args.flash_attn,
        args.ctx_size,
        args.fit,
        args.batch_size,
        args.ubatch_size,
    )

    results = []
    for recipe in recipes:
        for layer_list in layer_candidates:
            safe_recipe = recipe.replace("/", "_")
            safe_layers = layer_list.replace(",", "_").replace("-", "to")
            log = Path(args.output_dir) / f"{safe_recipe}_{safe_layers}.log"
            env = build_turboquant_env(args, layer_list=layer_list, recipe=recipe)
            metrics = run_case(args.bin, args.model, args.prompt, log, env, args.n_predict, args.n_gpu_layers, args.flash_attn, args.ctx_size, args.fit, args.batch_size, args.ubatch_size)
            metrics["recipe"] = recipe
            metrics["layer_list"] = layer_list
            results.append(attach_gains(metrics, baseline))

    results.sort(key=lambda row: objective_score(row, args.objective), reverse=True)
    recommended = results[0] if results else None
    recommendation_env = None
    if recommended:
        recommendation_env = {
            "LLAMA_ARG_KV_EXPERIMENT": "turboquant_exact",
            "LLAMA_ARG_TURBOQUANT_RECIPE": recommended["recipe"],
            "LLAMA_ARG_TURBOQUANT_LAYER_LIST": recommended["layer_list"],
            "LLAMA_ARG_TURBOQUANT_CACHE": "1",
        }
        if args.v_type:
            recommendation_env["LLAMA_ARG_TURBOQUANT_V_TYPE"] = args.v_type
        if args.metadata_path:
            recommendation_env["LLAMA_ARG_TURBOQUANT_METADATA_PATH"] = args.metadata_path

    print(json.dumps({
        "baseline": baseline,
        "results": results,
        "recommended": recommended,
        "recommended_env": recommendation_env,
        "objective": args.objective,
    }, indent=2))


def autotune_context(args: argparse.Namespace) -> None:
    recipes = [item.strip() for item in args.recipes.split(",") if item.strip()]
    v_types = [item.strip() for item in args.v_types.split(",") if item.strip()]
    layer_candidates = []
    if args.layer_lists:
        layer_candidates.extend(item.strip() for item in args.layer_lists.split(";") if item.strip())
    else:
        for start in range(args.layer_start, args.layer_end + 1, max(1, args.layer_stride)):
            stop = min(args.layer_end, start + args.layer_span - 1)
            layer_candidates.append(",".join(str(v) for v in range(start, stop + 1, max(1, args.layer_inner_stride))))

    baseline_log = Path(args.output_dir) / "baseline.log"
    baseline = run_case(
        args.bin,
        args.model,
        args.prompt,
        baseline_log,
        {"LLAMA_ARG_KV_EXPERIMENT": "off"},
        args.n_predict,
        args.n_gpu_layers,
        args.flash_attn,
        args.ctx_size,
        args.fit,
        args.batch_size,
        args.ubatch_size,
    )

    results = []
    for recipe in recipes:
        for v_type in v_types:
            for layer_list in layer_candidates:
                safe_recipe = recipe.replace("/", "_")
                safe_layers = layer_list.replace(",", "_").replace("-", "to")
                log = Path(args.output_dir) / f"{safe_recipe}_{v_type}_{safe_layers}.log"
                env = build_turboquant_env(args, layer_list=layer_list, recipe=recipe)
                env["LLAMA_ARG_TURBOQUANT_V_TYPE"] = v_type
                metrics = run_case(args.bin, args.model, args.prompt, log, env, args.n_predict, args.n_gpu_layers, args.flash_attn, args.ctx_size, args.fit, args.batch_size, args.ubatch_size)
                metrics["recipe"] = recipe
                metrics["layer_list"] = layer_list
                metrics["v_type"] = v_type
                metrics["ctx_target"] = args.ctx_size
                if args.ctx_size and "fitted_ctx" in metrics:
                    metrics["ctx_target_reach_pct"] = 100.0 * (metrics["fitted_ctx"] / args.ctx_size)
                results.append(attach_gains(metrics, baseline))

    results.sort(key=lambda row: objective_score(row, args.objective), reverse=True)
    recommended = results[0] if results else None
    recommendation_env = None
    if recommended:
        recommendation_env = {
            "LLAMA_ARG_KV_EXPERIMENT": "turboquant_exact",
            "LLAMA_ARG_TURBOQUANT_RECIPE": recommended["recipe"],
            "LLAMA_ARG_TURBOQUANT_LAYER_LIST": recommended["layer_list"],
            "LLAMA_ARG_TURBOQUANT_V_TYPE": recommended["v_type"],
            "LLAMA_ARG_TURBOQUANT_CACHE": "1",
        }
    print(json.dumps({
        "baseline": baseline,
        "results": results,
        "recommended": recommended,
        "recommended_env": recommendation_env,
        "objective": args.objective,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="TurboQuant llama.cpp lab validation harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--bin", required=True)
    common.add_argument("--model", required=True)
    common.add_argument("--prompt", default="TurboQuant validation prompt on long-context inference.")
    common.add_argument("--output-dir", required=True)
    common.add_argument("--n-predict", type=int, default=8)
    common.add_argument("--n-gpu-layers", default="auto")
    common.add_argument("--batch-size", type=int, default=None)
    common.add_argument("--ubatch-size", type=int, default=None)
    common.add_argument("--flash-attn", action=argparse.BooleanOptionalAction, default=True)
    common.add_argument("--ctx-size", type=int, default=None)
    common.add_argument("--fit", action=argparse.BooleanOptionalAction, default=True)
    common.add_argument("--recipe", default="lab_context_fast")
    common.add_argument("--bits", type=int, default=None)
    common.add_argument("--qjl-dim", type=int, default=None)
    common.add_argument("--outliers", type=int, default=None)
    common.add_argument("--metadata-path")
    common.add_argument("--v-type", choices=["f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1"], default="q8_0")
    common.add_argument("--rollout", choices=["auto", "balanced", "headroom", "late_sparse", "late_dense", "full_gpu", "memory_hybrid", "memory_gpu_only", "memory_max"], default=None)
    common.add_argument("--gpu-max-layers", type=int, default=None)
    common.add_argument("--objective", choices=["ctx_first", "latency_first", "balanced"], default="balanced")

    compare_p = sub.add_parser("compare", parents=[common])
    compare_p.add_argument("--layer-list")
    compare_p.add_argument("--layer-min", type=int)
    compare_p.add_argument("--layer-max", type=int)
    compare_p.set_defaults(func=compare)

    profiles_p = sub.add_parser("compare-profiles", parents=[common])
    profiles_p.add_argument("--profiles", default="balanced,headroom,fast150k,full_gpu,memory_hybrid,memory_gpu_only,memory_max")
    profiles_p.set_defaults(func=compare_profiles)

    promote_p = sub.add_parser("promotion-check", parents=[common])
    promote_p.add_argument("--profiles", default="balanced,headroom,fast150k,full_gpu,memory_hybrid,memory_gpu_only,memory_max")
    promote_p.add_argument("--synthetic-dim", type=int, default=128)
    promote_p.add_argument("--synthetic-bits", type=int, default=4)
    promote_p.add_argument("--synthetic-qjl-dim", type=int, default=512)
    promote_p.add_argument("--synthetic-samples", type=int, default=256)
    promote_p.add_argument("--synthetic-queries", type=int, default=64)
    promote_p.add_argument("--probe-layer", type=int, default=3)
    promote_p.add_argument("--probe-max-vectors", type=int, default=32)
    promote_p.add_argument("--probe-max-prompt-tokens", type=int, default=32)
    promote_p.add_argument("--probe-ctx", type=int, default=256)
    promote_p.add_argument("--probe-cpu-only", action=argparse.BooleanOptionalAction, default=False)
    promote_p.set_defaults(func=promotion_check)

    sweep_p = sub.add_parser("sweep-layers", parents=[common])
    sweep_p.add_argument("--layer-start", type=int, default=0)
    sweep_p.add_argument("--layer-end", type=int, default=8)
    sweep_p.set_defaults(func=sweep_layers)

    autotune_p = sub.add_parser("autotune", parents=[common])
    autotune_p.add_argument("--layer-start", type=int, default=0)
    autotune_p.add_argument("--layer-end", type=int, default=8)
    autotune_p.add_argument("--layer-lists")
    autotune_p.add_argument("--recipes", default="lab_best,turboquant35,turboquant25")
    autotune_p.add_argument("--layer-min", type=int)
    autotune_p.add_argument("--layer-max", type=int)
    autotune_p.set_defaults(func=autotune)

    autotune_ctx_p = sub.add_parser("autotune-context", parents=[common])
    autotune_ctx_p.add_argument("--layer-start", type=int, default=3)
    autotune_ctx_p.add_argument("--layer-end", type=int, default=15)
    autotune_ctx_p.add_argument("--layer-span", type=int, default=1)
    autotune_ctx_p.add_argument("--layer-stride", type=int, default=4)
    autotune_ctx_p.add_argument("--layer-inner-stride", type=int, default=1)
    autotune_ctx_p.add_argument("--layer-lists")
    autotune_ctx_p.add_argument("--recipes", default="lab_context,lab_best,turboquant35")
    autotune_ctx_p.add_argument("--v-types", default="q8_0,q4_0,f16")
    autotune_ctx_p.add_argument("--layer-min", type=int)
    autotune_ctx_p.add_argument("--layer-max", type=int)
    autotune_ctx_p.set_defaults(func=autotune_context)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

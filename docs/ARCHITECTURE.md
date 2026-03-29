# Architecture

## Layers

This repository is split into three layers.

### 1. Portable TurboQuant Core

Location:

- `cpp/include/turboquant/turboquant.h`
- `cpp/src/turboquant.cpp`

This layer contains:

- random orthogonal rotation
- Lloyd-Max scalar quantization
- QJL residual correction
- exact inner-product scoring utilities
- synthetic evaluation helpers

This is the most reusable part of the project.

### 2. Standalone Tools

Location:

- `cpp/tools/turboquant-bench.cpp`
- `scripts/validate_llama_cli.py`

This layer contains:

- standalone synthetic benchmarking
- llama.cpp log parsing and profile validation

### 3. llama.cpp Integration Surface

Location:

- `patches/llama.cpp/`

This layer is where the engine-specific work belongs:

- packed KV layout
- fused CUDA attention integration
- rollout policy
- runtime memory/latency tuning

That code still needs to be extracted cleanly from the lab engine before publication.

## Current Truth

The portable algorithm core is publishable now.

The full llama.cpp integration exists in the private lab tree and is being separated into:

- reusable source patches
- benchmark recipes
- reproducible notes

## Validation Scope

Current practical runtime validation has focused on:

- `Qwen3.5-9B-Q8_0.gguf`
- `Qwen3.5-27B-Q3_K_M.gguf`

So this project should currently be read as:

- model-agnostic at the algorithm layer
- Qwen-oriented at the extracted runtime-validation layer

Broader model-family support should be treated as an explicit validation task, not assumed by default.

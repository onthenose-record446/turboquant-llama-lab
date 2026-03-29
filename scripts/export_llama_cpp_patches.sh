#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAB_DIR="${LAB_DIR:-$ROOT_DIR/../llama.cpp_engine_lab}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/patches/llama.cpp/generated}"

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/*.patch

if [[ ! -d "$LAB_DIR" ]]; then
    echo "error: lab tree not found at $LAB_DIR" >&2
    exit 1
fi

append_tracked_patch() {
    local out_file="$1"
    shift
    if git -C "$LAB_DIR" diff -- "$@" > "$out_file"; then
        :
    fi
}

append_untracked_patch() {
    local out_file="$1"
    shift
    local file
    for file in "$@"; do
        git -C "$LAB_DIR" diff --no-index -- /dev/null "$file" >> "$out_file" || true
    done
}

PATCH1="$OUT_DIR/0001-turboquant-core-hooks.patch"
PATCH2="$OUT_DIR/0002-turboquant-exact-core-and-tools.patch"
PATCH3="$OUT_DIR/0003-turboquant-llama-runtime-integration.patch"

append_tracked_patch "$PATCH1" \
    common/CMakeLists.txt \
    common/arg.cpp \
    common/common.cpp \
    common/common.h \
    common/speculative.cpp \
    tools/CMakeLists.txt \
    tools/llama-bench/llama-bench.cpp \
    src/CMakeLists.txt

append_untracked_patch "$PATCH2" \
    common/turboquant.h \
    common/turboquant.cpp \
    ggml/src/ggml-cuda/turboquant-exact.cuh \
    ggml/src/ggml-cuda/turboquant-probe.cu \
    tools/turboquant-bench/turboquant-bench.cpp \
    tools/turboquant-probe/turboquant-probe.cpp \
    tools/turboquant-probe/turboquant-probe-cuda.h

append_tracked_patch "$PATCH3" \
    ggml/src/ggml-cpu/ops.cpp \
    ggml/src/ggml-cuda/cpy.cu \
    ggml/src/ggml-cuda/fattn-vec.cuh \
    ggml/src/ggml-cuda/fattn.cu \
    ggml/src/ggml-cuda/ggml-cuda.cu \
    ggml/src/ggml-cuda/set-rows.cu \
    ggml/src/ggml.c \
    src/llama-context.cpp \
    src/llama-graph.cpp \
    src/llama-kv-cache.cpp \
    src/llama-kv-cache.h

for patch in "$PATCH1" "$PATCH2" "$PATCH3"; do
    if [[ ! -s "$patch" ]]; then
        rm -f "$patch"
    fi
done

echo "generated patches:"
find "$OUT_DIR" -maxdepth 1 -type f -name '*.patch' | sort

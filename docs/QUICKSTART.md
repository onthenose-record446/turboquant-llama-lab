# Quickstart

## 1. Build the portable benchmark

```bash
cmake -S . -B build
cmake --build build -j
```

## 2. Run the synthetic benchmark

```bash
./build/turboquant-bench --dim 128 --bits 4 --qjl-dim 128 --samples 32 --queries 8
```

This checks that the portable TurboQuant core is working independently of `llama.cpp`.

## 3. Run llama.cpp validation

```bash
python3 scripts/validate_llama_cli.py compare \
  --bin /path/to/llama-cli \
  --model /path/to/model.gguf \
  --output-dir /tmp/turboquant-compare
```

## 4. Export llama.cpp patch sets

```bash
scripts/export_llama_cpp_patches.sh
```

This generates reviewable patch files in:

- `patches/llama.cpp/generated/`

## 5. Read the profile guidance

Use:

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [USAGE_PROFILES.md](USAGE_PROFILES.md)
- [VALIDATED_MODELS.md](VALIDATED_MODELS.md)

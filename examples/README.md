# Examples

## Build the standalone benchmark

```bash
cmake -S . -B build
cmake --build build -j
```

## Run the synthetic benchmark

```bash
./build/turboquant-bench --dim 128 --bits 4 --qjl-dim 512 --samples 256 --queries 64
```

## Run llama.cpp log validation

```bash
python3 scripts/validate_llama_cli.py compare \
  --bin /path/to/llama-cli \
  --model /path/to/model.gguf \
  --output-dir /tmp/turboquant-compare
```

# Validated Models

## Current Validation Focus

The runtime-oriented work has been validated mainly on:

- `Qwen3.5-9B-Q8_0.gguf`
- `Qwen3.5-27B-Q3_K_M.gguf`

## Interpretation

### Portable Core

Status:

- generic
- not tied to one model family

### llama.cpp Runtime Integration

Status:

- validated mainly on Qwen 3.5 GGUF variants
- expected to need tuning on other families

## What To Revalidate For New Models

When trying a different model, users should re-check:

- layer rollout choice
- bit budget
- QJL dimension
- outlier-channel count
- prompt-time latency
- decode-time latency
- real long-context quality

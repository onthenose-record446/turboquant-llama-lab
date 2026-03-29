# Scripts

Current script:

- `validate_llama_cli.py`

Purpose:

- run baseline vs TurboQuant comparisons against a llama.cpp-style binary
- parse prompt/gen/total timing
- parse KV savings and selected layers
- compare rollout profiles

Current origin:

- extracted from the V6 Alpha TurboQuant validation workflow

Expected future cleanup:

- remove remaining V6-specific naming
- add standalone examples and canned benchmark presets

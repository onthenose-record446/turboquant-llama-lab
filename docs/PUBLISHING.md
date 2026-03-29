# Publishing Plan

## Name

Repository name:

- `turboquant-llama-lab`

Why this name:

- `turboquant`: clear research target
- `llama`: clear runtime family
- `lab`: honest about experimental status

## Initial Release Scope

The first public release should include:

1. portable TurboQuant C++ core
2. standalone benchmark tool
3. validation script for llama.cpp runs
4. architecture notes
5. explicit limitations

## What Should Not Be Overclaimed

Do not claim:

- official Google release
- paper-complete production parity
- universal speedup
- benchmark parity with H100/JAX numbers

Do claim:

- practical TurboQuant research implementation
- portable exact core
- llama.cpp-oriented lab integration path
- measured local benchmark results where available

## Proposed Next Extraction Steps

1. publish the core and bench first
2. add extracted llama.cpp patch sets under `patches/llama.cpp/`
3. add reproducible benchmark reports
4. add a short technical report comparing:
   - baseline
   - balanced
   - headroom
   - memory-max

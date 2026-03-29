# v0.1.0 - Initial public TurboQuant Llama Lab release

## Highlights

- portable TurboQuant C++ core
- standalone benchmark tool
- `llama.cpp`-style validation script
- extracted runtime patch sets
- architecture and usage documentation

## What This Release Is

This is the first public release of the lab.

It is intended to give users:

- a clean starting point for TurboQuant experiments
- reusable algorithm pieces
- reviewable runtime patch groups
- honest guidance on where the implementation is strongest today

## What This Release Is Not

- an official Google release
- a claim of universal benchmark parity
- a guarantee of one best profile for every model and hardware target

## Suggested Next Steps For Users

1. build the standalone benchmark
2. read `docs/QUICKSTART.md`
3. run the validator against a local `llama.cpp` binary
4. inspect the generated patch sets

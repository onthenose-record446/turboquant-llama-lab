# Contributing

Thanks for helping improve `turboquant-llama-lab`.

## Good Contributions

The most useful contributions right now are:

- correctness fixes in the portable TurboQuant core
- benchmark and validation improvements
- cleaner `llama.cpp` patch extraction
- reproducible reports on new model families
- documentation improvements that make the repo easier to use honestly

## Ground Rules

- keep claims truthful
- separate portable-core changes from runtime-integration changes
- prefer small reviewable patch sets
- attach benchmark evidence when changing runtime behavior
- document model and hardware assumptions

## Suggested Workflow

1. build the standalone benchmark
2. run a synthetic check
3. if changing runtime behavior, run the validation script
4. describe:
   - what changed
   - what hardware was used
   - what model was used
   - whether the result favored speed, memory, or both

## Things To Avoid

- claiming paper-level parity without evidence
- merging unrelated runtime and algorithm changes into one patch
- presenting unstable profiles as default recommendations

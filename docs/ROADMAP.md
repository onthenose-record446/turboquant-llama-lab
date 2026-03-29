# Roadmap

This repository is organized around a few practical milestones.

## Milestone 1: Portable Core

- keep the standalone TurboQuant core easy to build
- keep the benchmark tool small and reproducible
- document algorithm choices clearly

## Milestone 2: Reviewable Runtime Patches

- keep `llama.cpp` integration changes split into clean patch groups
- explain what each patch does and why it exists
- make it easy for users to apply only the pieces they need

## Milestone 3: Validation

- publish repeatable baseline vs TurboQuant comparisons
- cover both memory-first and speed-first profiles
- expand validated model coverage beyond the current Qwen focus

## Milestone 4: Better Long-Context Performance

- reduce prompt-time packing overhead
- improve fused CUDA execution paths
- widen stable rollout coverage without sacrificing correctness

## Milestone 5: Broader Backend Support

- keep the core backend-agnostic where possible
- make it easier to adapt the implementation to other runtimes
- document portability boundaries honestly

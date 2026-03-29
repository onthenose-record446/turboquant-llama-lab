# llama.cpp Patch Extraction

This directory is reserved for the engine-integration layer.

Planned contents:

- packed KV-cache storage patches
- fused CUDA attention-path patches
- rollout-policy hooks
- benchmark and validation notes for each patch set

## Source of Truth Today

The current working integration still lives in a separate local lab tree outside this
standalone repository.

The goal of this repository is to extract those changes into:

- minimal reviewable patch sets
- portable documentation
- reproducible benchmark instructions

## Patch Strategy

1. isolate algorithm-independent engine hooks
2. isolate packed K/V storage changes
3. isolate fused CUDA TurboQuant path
4. attach benchmark evidence to every patch group

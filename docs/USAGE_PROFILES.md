# Usage Profiles

## Goal

TurboQuant does not currently collapse into one universal best profile.

In practice, users should choose a profile based on what they want most:

- speed
- balanced long-context behavior
- maximum memory reduction

## Fast Profile

Best for:

- interactive long-context use
- low latency
- live control loops

Typical direction:

- lighter rollout
- later sparse layers
- less aggressive `V` compression

## Balanced Profile

Best for:

- useful memory savings
- lower risk than memory-max
- practical long-context benchmarking

Typical direction:

- moderate number of late GPU-resident layers
- exact TurboQuant only where it buys the most

## Memory-Max Profile

Best for:

- extreme context pressure
- headroom experiments
- memory-first evaluation

Typical direction:

- broader rollout
- more aggressive `V` compression
- acceptance that speed may degrade

## Current Truth

The strongest practical split so far has been:

- one speed-oriented profile
- one memory-oriented profile

That is why this repository documents profiles explicitly instead of promising one setting that is best for every workload.

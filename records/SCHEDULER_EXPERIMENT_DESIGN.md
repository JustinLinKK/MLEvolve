# GPU-Normalized Scheduler Experiment Design

## Claim

- Compare time-aware scheduler vs FIFO serial and MP2

- Lower mean flow, non-inferior makespan, safer memory

## Policies

- P0 FIFO serial cap1, P1 FIFO MP2 cap2

- P2 adaptive

- P3 oracle exact timings, screens before real GPU

## Suites

- A SRT 12 jobs, B 9-cell matrix, C mixed

- D homogeneous control, E warm profiles versus cold trials

## Gates

- Need p90/p10 >= 4, solo CV <= 10%

- Need 20% beneficial pairs, 20% harmful pairs

## Metrics

- Primary mean flow time, finish minus release

- Also makespan, p95 wait, OOM count, energy

## Acceptance

- Mean flow improves 10%, CI upper bound below 1.0

- Makespan non-inferior to MP2, ratio bound below 1.05

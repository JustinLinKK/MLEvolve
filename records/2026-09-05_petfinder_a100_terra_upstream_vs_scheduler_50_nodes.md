# PetFinder A100: 50 execution-node comparison

## Scope

- Task: PetFinder Pawpularity.
- Agent: Codex GPT-5.6 Terra.
- Hardware: one NVIDIA A100 GPU for each recorded run.
- Comparison unit: the first 50 completed non-root execution nodes in creation-time order. This includes nodes that failed at runtime; it is **not** a budget-counted-only comparison.

## Sources

- Upstream MLEvolve (Multi-Process Service corrected): `.cache/petfinder_terra_a100_20260905/upstream_60_mpsuuid/journal.json`.
- Branch-profile Scheduler clean 50-node run: `.cache/petfinder_terra_a100_20260905/scheduler_50_clean/journal.json`.
- Figure: `2026-09-05_petfinder_a100_terra_upstream_vs_scheduler_50_nodes.png`.

## Observed 50-node prefix

| Run | Executions shown | Scored nodes | Runtime-error nodes | Wall-clock span | Peak execution concurrency | Best validation root mean squared error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Upstream MLEvolve (MPS-corrected) | 50 | 26 | 24 | 1.63 h | 3 | 17.9972 |
| Branch-profile Scheduler (clean run) | 50 | 46 | 4 | 6.64 h | 2 | 17.8790 |

The Gantt panels encode successful executions in blue and runtime-error (buggy) nodes in red. The lower panels only plot validation metrics for scored nodes. The timing panels are directly comparable in node count, but the runs differ in scheduler configuration and therefore do not by themselves establish a causal throughput advantage.

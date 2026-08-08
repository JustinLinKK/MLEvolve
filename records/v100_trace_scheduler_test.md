# V100 Trace Scheduler Test

## Goal

- Replay an MLEvolve-derived workload trace through the scheduler and compare policies on 1x V100

## Trace

- Path: `traces/mlebench_v100_100jobs.jsonl`

- 100 jobs sampled from 20 real LLM-agent search steps recorded in the cassava sweep (`sweep_cassava_2026-05-06/workload_trace.jsonl`)

- Arrivals: Poisson, $\lambda = 4$ jobs/min, span 0-1678.1 s

- Solo runtimes taken from measured `exec_duration_s` of the source steps, with 5 % Gaussian jitter

- Total solo compute: 15231 s (253.8 min); mean 152.3 s/job; mean 6100 MiB/job

> Architecture mix

| Architecture | Count |
|---|---|
| efficientnet_b0 | 34 |
| convnext_small | 32 |
| resnet101 | 26 |
| swin_tiny_patch4_window7_224 | 6 |
| resnet50 | 2 |

## Scheduler Configuration

- `memory_budget_mb = 31000` (per CLAUDE.md)

- `parallel_cap = 5`

- `default_slowdown = 4.0` (measured V100 CNN pair slowdown, see [v100_packing_results.md](v100_packing_results.md))

- `colocation_trial_epochs = 2`, `colocation_min_gain = 1.0`

- `starvation_timeout_seconds = 1800`

- `backend_allowlist = ["cuda_process"]`

## Results (100 jobs)

| Policy | Makespan (s) | Mean flow (s) | Avg slowdown | Slowdown rejections | Trial epochs | Starved |
|---|---|---|---|---|---|---|
| serial (priority-FIFO) | 15230.9 | 6721.6 | 1.00 | 0 | 0.0 | 86 |
| time_aware (SRT-first) | 15230.9 | 6321.5 | 1.00 | 0 | 0.0 | 80 |
| recursive_time_aware (packing) | 30406.0 | 13973.1 | 3.98 | 0 | 97.1 | 89 |

> Makespan relative to serial

| Policy | Ratio |
|---|---|
| time_aware | 1.000x |
| recursive_time_aware | 0.501x |

- Gantt chart: `records/scheduling_gantt_v100.png`

## Colocation Trial Sensitivity (12-job subset)

- Serial makespan on the subset: 1755.2 s

| `colocation_trial_epochs` | Makespan (s) | vs serial | Slowdown rejections | Avg slowdown |
|---|---|---|---|---|
| 1 | 3339.8 | 1.90x worse | 2 | 3.74 |
| 2 | 3401.1 | 1.94x worse | 0 | 3.77 |
| 3 | 3401.1 | 1.94x worse | 0 | 3.75 |

## Observations

- `recursive_time_aware` admitted a second job on essentially every dispatch; the Gantt chart shows black-outlined (packed) bars across the whole timeline

- Measured average slowdown under packing was 3.98, matching the configured pair slowdown of 4.0

- 97.1 epochs were consumed by colocation trials and zero packing decisions were rejected

- With `colocation_trial_epochs = 2` or `3`, `slowdown_rejections` stayed at 0; lowering it to 1 produced 2 rejections

- The trace's jobs have `planned_epochs` in {1, 2, 3}, at or below the trial length of 2 epochs

- In `trace_simulator.simulate_recursive_time_aware`, `run_trial` returns `"completed"` when the candidate finishes during the trial window and returns `"accepted"` when all pre-existing members finish; the gain comparison against `colocation_min_gain` is only reached when the trial runs to completion with the original membership intact

- `time_aware` and `serial` produced identical makespans; `time_aware` reduced mean flow by 6.0 % and starvation count from 86 to 80

## Reproduction

```bash
python3 -m scheduler_benchmark_test.run_trace_experiment
```

```bash
python3 -m scheduler_benchmark_test.test_trace_policies
```

# V100 Trace Scheduler Test

## Goal

- Replay MLEvolve-derived workload traces through the scheduler and compare policies on 1x V100

## Traces

> `traces/mlebench_v100_100jobs.jsonl` (CNN workload)

- 100 jobs sampled from 20 real LLM-agent search steps recorded in the cassava sweep (`sweep_cassava_2026-05-06/workload_trace.jsonl`)

- Solo runtimes from measured `exec_duration_s` of the source steps, with 5% Gaussian jitter

- Total solo compute 15231 s; mean 152.3 s/job; mean 6100 MiB/job

- Architectures: efficientnet_b0 34, convnext_small 32, resnet101 26, swin_tiny 6, resnet50 2

> `traces/mlebench_tabular_v100_100jobs.jsonl` (tabular workload)

- 100 jobs over the two tabular-playground MLEBench-Lite tasks (dec-2021, may-2022)

- Built from the two configurations measured to pack on V100 in [v100_tabular_packing.md](v100_tabular_packing.md)

- Solo runtimes derived from measured solo step rates and published training-set sizes

- Total solo compute 9094 s; mean 90.9 s/job

- Split: `tabular_mlp_w256_d2_b1024` 57, `tabular_mlp_w128_d1_b512` 43

- Both traces use Poisson arrivals at $\lambda = 4$ jobs/min

## Scheduler Configuration

- `memory_budget_mb = 31000` (per CLAUDE.md)

- `parallel_cap = 5`

- `colocation_trial_epochs = 2`, `colocation_min_gain = 1.0`

- `starvation_timeout_seconds = 1800`

- `backend_allowlist = ["cuda_process"]`

- `default_slowdown` is the measured pair slowdown. The simulator composes a member's slowdown as $1 + \sum (\text{pair} - 1)$ over co-runners, so the pair value is recovered from the N=5 measurement:

$$
\begin{equation}
\begin{aligned}
\textbf{pair} &= 1 + \frac{\textbf{sd}_{N=5} - 1}{4} \\
\textbf{CNN:} \quad \textbf{pair} &= 4.0 \\
\textbf{tabular:} \quad \textbf{pair} &= 1 + \frac{1.067 - 1}{4} = 1.017
\end{aligned}
\end{equation}
$$

## Results: Tabular Workload (packable)

| Policy | Makespan (s) | Mean flow (s) | Avg slowdown | Slowdown rejections | Trial epochs | Starved |
|---|---|---|---|---|---|---|
| serial (priority-FIFO) | 9094.4 | 3821.2 | 1.00 | 0 | 0.0 | 74 |
| time_aware (SRT-first) | 9094.4 | 2352.0 | 1.00 | 0 | 0.0 | 48 |
| recursive_time_aware (packing) | 2748.3 | 374.6 | 1.05 | 0 | 636.2 | 3 |

- `recursive_time_aware` finishes at **3.309x** the serial makespan

- Mean flow drops from 3821.2 s to 374.6 s, a 10.2x reduction

- Starvation count drops from 74 to 3

- Realized average slowdown is 1.05, inside the 1.15x gate

- Gantt chart: `records/scheduling_gantt_v100_tabular.png`

## Results: CNN Workload (not packable)

| Policy | Makespan (s) | Mean flow (s) | Avg slowdown | Slowdown rejections | Trial epochs | Starved |
|---|---|---|---|---|---|---|
| serial (priority-FIFO) | 15230.9 | 6721.6 | 1.00 | 0 | 0.0 | 86 |
| time_aware (SRT-first) | 15230.9 | 6321.5 | 1.00 | 0 | 0.0 | 80 |
| recursive_time_aware (packing) | 30406.0 | 13973.1 | 3.98 | 0 | 97.1 | 89 |

- `recursive_time_aware` finishes at **0.501x** the serial makespan, i.e. twice as slow

- Realized average slowdown is 3.98, matching the configured pair slowdown of 4.0

- 97.1 epochs were spent on colocation trials and **zero** packing decisions were rejected

- Gantt chart: `records/scheduling_gantt_v100_cnn.png`

## Colocation Trial Sensitivity (CNN workload, 12-job subset)

- Serial makespan on the subset: 1755.2 s

| `colocation_trial_epochs` | Makespan (s) | vs serial | Slowdown rejections | Avg slowdown |
|---|---|---|---|---|
| 1 | 3339.8 | 1.90x worse | 2 | 3.74 |
| 2 | 3401.1 | 1.94x worse | 0 | 3.77 |
| 3 | 3401.1 | 1.94x worse | 0 | 3.75 |

## Observations

- On the packable tabular workload the packing policy behaves as intended and delivers the largest win of any policy tested

- On the non-packable CNN workload the packing policy never rejected a colocation, despite a realized slowdown of 3.98

- The CNN trace's jobs have `planned_epochs` in {1, 2, 3}, at or below the trial length of 2 epochs

- In `trace_simulator.simulate_recursive_time_aware`, `run_trial` returns `"completed"` when the candidate finishes during the trial window and `"accepted"` when all pre-existing members finish; the `colocation_min_gain` comparison is only reached when the trial runs to completion with the original membership intact

- Lowering `colocation_trial_epochs` to 1 produced 2 rejections but still finished 1.90x worse than serial, because the anchor absorbs the slowdown for the whole trial before any evidence exists

- `time_aware` never changed makespan relative to `serial` on either workload, but reduced mean flow by 6.0% (CNN) and 38.4% (tabular), and cut starvation counts

## Reproduction

```bash
python3 -m scheduler_benchmark_test.gen_tabular_trace
```

```bash
python3 -m scheduler_benchmark_test.run_trace_experiment tabular
```

```bash
python3 -m scheduler_benchmark_test.run_trace_experiment cnn
```

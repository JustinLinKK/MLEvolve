# Scheduler Comparison on the Cassava Trace

## Setup

- Trace: `traces/mlevolve_cassava_v100_deepseek.jsonl`, 42 recorded nodes, 39 after dropping sub-second agent no-ops

- Predictor: profile-based only. `ResourceEstimator` builds the ML predictor solely when `settings.prediction.mode == PREDICTION_MODE_ML_PREDICTOR`, so leaving the mode alone means every estimate is sourced `branch_profile`

- Budget 31000 MB per CLAUDE.md, parallel cap 5

## Two Corrections Needed Before Any Number Was Valid

- `device_peak_vram_mib` is whole-device and reaches 32751 MB because it includes every co-runner. Using it exceeds the 31 GB budget and makes every pack infeasible, so the per-job field `delta_peak_vram_mib` is used instead: median 7894 MB, max 26364 MB

- No cassava row ran solo, so `exec_duration_s` is colocation-contaminated. Device power sampled during the run gives the achieved aggregate throughput at each concurrency, and each recorded duration is divided by the slowdown implied at its own concurrency:

| N | mean power W | above idle | aggregate | slowdown |
|---:|---:|---:|---:|---:|
| 1 | 127.1 | 70.4 | 1.00 | 1.00 |
| 2 | 174.0 | 117.3 | 1.67 | 1.20 |
| 3 | 194.6 | 137.9 | 1.96 | 1.53 |
| 4 | 196.8 | 140.1 | 1.99 | 2.01 |
| 5 | 251.4 | 194.7 | 2.77 | 1.81 |

- Idle draw is 56.7 W. The pair slowdown implied by N=2 is 1.198, used as `default_slowdown`

## Repo Scheduler, Recorded Arrivals

| policy | makespan | mean flow | p95 flow | max wait | avg sd | starved | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| serial, priority-FIFO | 464.9 min | 85.4 | 179.6 | 171.4 | 1.00 | 33 | 1.000x |
| time_aware, SRT-first | 464.9 min | 82.6 | 179.6 | 171.4 | 1.00 | 31 | 1.000x |
| recursive_time_aware | 314.2 min | 22.8 | 62.1 | 33.7 | 1.16 | 1 | 1.480x |

- `recursive_time_aware` is this repo's scheduler and it wins clearly: 1.48x makespan, 3.7x better mean flow, starvation from 33 jobs down to 1

- SRT ordering on its own contributes nothing to makespan. `_serial_choice` and `_time_aware_choice` both call `feasible_packs(problem, (anchor,))`, which passes only the anchor, so neither can ever emit a pack larger than one job. All the gain comes from packing

## My Scheduler

- Two changes over the repo design:

- `simulate_policy` advances the clock by a pack's `drain_seconds`, so every slot stays blocked until the slowest member finishes. A one-minute job packed behind a sixty-minute job wastes a slot for fifty-nine minutes. My simulator lets each job finish independently

- Free slots are refilled the moment a job completes or arrives, with no two-epoch colocation trial, because under the measured curve aggregate throughput rises monotonically with concurrency and there is nothing a trial can discover

| policy | makespan | mean flow | starved | mean conc | speedup |
|---|---:|---:|---:|---:|---:|
| recursive_time_aware, repo | 314.2 min | 22.8 | 1 | - | 1.480x |
| occupancy-lpt | 304.7 min | 22.7 | 4 | 1.94 | 1.526x |
| occupancy-density | 304.7 min | 22.4 | 4 | 1.91 | 1.526x |
| occupancy-small | 304.7 min | 21.4 | 2 | 1.91 | 1.526x |

- 1.031x over the repo scheduler, 9.5 min saved. All three admission preferences tie exactly, which is the signal that something other than the policy is binding

## The Fixed-Arrival Replay Is the Wrong Experiment

- At most five jobs are ever ready simultaneously and the median gap between arrivals is 275 s, so every policy starves and they all finish within minutes of each other

- Arrivals are not independent of the schedule. A node cannot be generated until its parent produces a result, which is the model Trace_Generation.md specifies:

$$
\begin{equation}
\begin{aligned}
t^{\textbf{arrive}}_{n} =
t^{\textbf{exec-end}}_{\textbf{parent}(n)} + d^{\textbf{gen}}_{n}
\end{aligned}
\end{equation}
$$

- Under this model finishing a parent early pulls its child forward and the saving compounds along each branch. The repo simulator cannot express it, because `TraceJob.release_seconds` is fixed at construction

## Dependency-Aware Replay

| policy | makespan | mean flow | p95 flow | avg sd | mean conc | speedup |
|---|---:|---:|---:|---:|---:|---:|
| serial, cap 1 | 472.4 min | 44.4 | 67.9 | 1.00 | 1.00 | 1.000x |
| occupancy-lpt | 314.9 min | 24.2 | 55.5 | 1.27 | 1.77 | 1.500x |
| occupancy-density | 316.4 min | 23.1 | 59.6 | 1.28 | 1.75 | 1.493x |
| occupancy-small | 316.4 min | 23.1 | 59.6 | 1.28 | 1.75 | 1.493x |
| occupancy-critical-path | 299.9 min | 23.3 | 54.8 | 1.27 | 1.88 | 1.575x |
| critical-path plus contention guard | 299.7 min | 25.7 | 54.2 | 1.22 | 1.86 | 1.576x |

- Critical-path admission, ranking by the total training plus generation cost of a node's whole subtree, beats longest-processing-time by 15 min. Under dependency-aware arrivals the makespan is set by the longest chain of train-then-generate steps, so the job whose subtree carries the most remaining work must never wait behind a leaf

- Protecting critical jobs from contention adds almost nothing, 299.9 to 299.7

## What Actually Limits This Workload

| configuration | makespan | mean conc |
|---|---:|---:|
| cap 5, 31 GB, as measured | 299.9 min | 1.88 |
| unlimited memory, cap 5 | 239.9 min | 2.80 |
| unlimited cap, 31 GB | 299.9 min | 1.88 |
| unlimited cap and memory | 235.6 min | 2.92 |
| free colocation, pure critical path | 200.9 min | 2.34 |

- The 31 GB budget costs 60 min. The parallel cap of 5 costs nothing at all, since raising it changes neither makespan nor concurrency

- Trading criticality for a smaller footprint was tested by scoring nodes as criticality minus alpha times budget fraction. Every alpha at or above 0.5 made the result worse, 319.2 min, so pure critical-path ordering is the best available under this budget

- The scheduler is therefore at its achievable optimum here. Further gain needs more memory or smaller per-job footprints, not a cleverer policy

## Reproduction

```bash
python3 -m scheduler_benchmark_test.run_cassava_scheduler_test
python3 -m scheduler_benchmark_test.compare_schedulers
python3 -m scheduler_benchmark_test.compare_dag_schedulers
```

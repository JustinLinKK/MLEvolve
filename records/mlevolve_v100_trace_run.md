# MLEvolve V100 Trace Run

## Goal

- Run the original MLEvolve agent on Nautilus against an MLEBench-Lite task, record a trace, and replay that trace through the scheduler

- Single V100, branch-profile scheduling only, no PerfSeer predictor

## Running Original MLEvolve Without an API Key

- The original MLEvolve lives on Nautilus at `/root/downeyflyfan/perfseer_test/exp_run` and drives its agents through `llm/__init__.py`, which dispatched only to `llm/openai.py` and `llm/gemini.py`

- Both backends require an API key, and `config/config.yaml` still carries the `enter-your-api-key` placeholder, which is what previously blocked this step

- The `claude` CLI is installed and authenticated on Nautilus, and the earlier cassava trace run drove MLEvolve the same way (claude CLI as a subprocess), so a CLI backend was added rather than a new key introduced

- `llm/claude_cli.py` implements the existing backend contract:

| Entry point | Returns |
|---|---|
| `query(system_message, user_message, func_spec, cfg, **kwargs)` | `(output, req_time, in_tokens, out_tokens, info)` |
| `generate(prompt, cfg, ..., json_schema, max_retries)` | generated text |

- When `func_spec` is supplied, the CLI is asked for a single JSON object matching the function's `json_schema`, and the parsed dict is returned in place of text, matching how the OpenAI backend surfaces a tool call

- `_provider()` in `llm/__init__.py` now routes `provider: claude_cli` to the new module; `openai` and `gemini` routing is unchanged

> Backend verification on Nautilus

| Path | Result |
|---|---|
| plain text query | returned `PONG` in 4.7 s |
| function-calling query | returned a `dict` with keys `['plan', 'epochs']`, `epochs = 3`, in 8.8 s |

- Python dependencies installed to make `run.py` import: `backoff`, `jsonschema`, `dataclasses-json`, `omegaconf`, `rich`, `humanize`, `funcy`, `google-genai`, `openai`, `coolname`, `shutup`, `black`, `genson`

## Trace Recording

- `run_traced.py` wraps `engine.executor.Interpreter.run`, so one row is appended per code execution the agent performs

- The interpreter is pinned to a single execution slot, because per-job VRAM attribution by device sampling is only valid when one job holds the GPU

- Recorded per job:

| Field | Meaning |
|---|---|
| `job_id`, `step_idx`, `node_id` | identity and position in the agent's search |
| `release_seconds` | submission time relative to run start, the scheduler's arrival time |
| `exec_submit_at`, `exec_complete_at`, `exec_duration_s` | measured execution window |
| `peak_vram_mib`, `avg_vram_mib` | NVML device memory across the window |
| `avg_sm_util_percent`, `peak_sm_util_percent` | NVML SM utilization across the window |
| `architecture`, `family`, `batch_size`, `epochs` | parsed back out of the generated script |
| `is_buggy`, `exc_type` | whether the execution failed |
| `code` | the full generated script, so the job can be re-run |

- Fields parsed from source are best-effort; `None` means the value was not stated literally and must not be read as measured

- Per-epoch validation metrics are not captured, since agent scripts report progress in arbitrary formats, so early stopping is left disabled when replaying

## Configuration

- Task: `leaf-classification` (MLEBench-Lite), 891 training rows, 192 features, 99 classes

- `scheduler.enabled=false` and `coldstart.use_coldstart=false`, so the recorded trace reflects unscheduled agent behavior and no predictor is involved

- `agent.search.parallel_search_num=1`, `CUDA_VISIBLE_DEVICES=0`

- Model: `claude-sonnet` via `provider: claude_cli` for both the code and feedback stages

## Measured Jobs

> First four non-buggy executions

| Job | Family | Batch | Duration (s) | Peak VRAM (MiB) | Avg SM (%) | Peak SM (%) |
|---|---|---|---|---|---|---|
| mlevolve_0000 | mlp | 64 | 56.4 | 669.5 | 1.9 | 11.0 |
| mlevolve_0001 | cnn | 64 | 113.8 | 1359.5 | 4.4 | 14.0 |
| mlevolve_0002 | mlp | 32 | 23.7 | 665.5 | 1.9 | 7.0 |
| mlevolve_0003 | mlp | — | — | 669.5 | 4.7 | 11.0 |

## Finding: leaf-classification Does Not Fill V100 Compute

- Real agent jobs on this task sustain 1.9-4.7 % average SM and peak at 7-14 %

- Packing 4-5 of them would reach roughly 8-20 % SM, so this task does not satisfy the requirement that 4-5 concurrent jobs fill the device

- Average SM is taken across the whole execution window, which includes data loading and preprocessing, so it understates the training phase; the 7-14 % peaks bound the training phase and are still far below saturation

- The cause is dataset size, not model family. 891 training rows give too little work per epoch to occupy 80 SMs, regardless of what the agent builds

- This is consistent with the standalone sweep in [v100_tabular_packing.md](v100_tabular_packing.md): reaching the packable-and-filling regime needed roughly 23 % solo SM, which came from tabular-playground data of 900k-4M rows

- Within MLEBench-Lite, the tasks that meet the requirement are therefore the tabular-playground ones, not the small tasks

| Task | Train rows | Solo SM | 4-5 jobs fill device |
|---|---|---|---|
| tabular-playground-series-dec-2021 | 4,000,000 | 23 % (w256_d2_b1024) | yes, 99 % at N=4 |
| tabular-playground-series-may-2022 | 900,000 | same configuration | yes |
| leaf-classification | 891 | 1.9-4.7 % measured | no |

## Measured Packing of the Agent's Own Script

- The longest non-buggy execution (`mlevolve_0004`, 685 s) was re-run at N = 1 to 5 concurrent copies on the V100, each in its own working directory with `input` linked to the task data

| N | Mean wall (s) | Slowdown | Avg SM (%) | Peak SM (%) | Failures | Gate |
|---|---|---|---|---|---|---|
| 1 | 673.3 | 1.000 | 1.2 | 10 | 0 | PASS |
| 2 | 833.1 | 1.237 | 6.3 | 78 | 0 | FAIL |
| 3 | 968.9 | 1.439 | 9.4 | 96 | 0 | FAIL |
| 4 | 1099.1 | 1.632 | 11.5 | 99 | 0 | FAIL |
| 5 | 1422.8 | 2.113 | 10.6 | 100 | 0 | FAIL |

- Average SM never exceeds 11.5 %, so even five concurrent copies leave the device's compute almost entirely unused

- Despite that, the 1.15x gate is already breached at N = 2, so these jobs do not pack either

- The contention is not on SM. Peak SM climbs to 78-100 % while the average stays near 10 %, which is the signature of short bursty kernels separated by long CPU-side phases; the copies collide on data loading and preprocessing, not on the GPU

- This is the opposite failure mode from the CNN sweep in [v100_packing_results.md](v100_packing_results.md), where jobs saturated SM at N = 1. Low SM utilization does not imply packing is free

## Correction to the Replay Assumption

- The scheduler replay above used a pair slowdown of 1.017, carried over from the synthetic tabular sweep

- The measured pair slowdown for this workload is 1.237, so the replay understates colocation cost for this trace; the 1.502x makespan figure should be read as the result for a workload that packs at 1.017, not as a measured outcome for leaf-classification

- Replayed at the measured 1.237, at 4 jobs/min

| Policy | Makespan (s) | Mean flow (s) | Avg slowdown | vs serial |
|---|---|---|---|---|
| serial (priority-FIFO) | 1625.1 | 914.8 | 1.00 | 1.000x |
| time_aware (SRT-first) | 1625.1 | 413.2 | 1.00 | 1.000x |
| recursive_time_aware | 1203.5 | 306.2 | 1.22 | 1.350x |

- The packing policy still wins at 1.350x makespan and cuts mean flow from 914.8 s to 306.2 s, but its realized slowdown of 1.22 sits above the 1.15x gate, and it recorded zero rejections

- That is the same defect already documented in [v100_trace_scheduler_test.md](v100_trace_scheduler_test.md): these jobs have no per-epoch metrics recorded, so `planned_epochs` defaults to 1, below `colocation_trial_epochs = 2`, and `run_trial` returns before the `colocation_min_gain` check runs

## Re-record at Native Parallelism

> Why the first trace was invalid

- The first recorder pinned `max_parallel_run` to 1 so that device-level VRAM sampling could be attributed to a single job

- That pin did more than serialize execution: `release_seconds` was stamped on entry to `Interpreter.run`, and with a one-worker pool that call blocks until the previous job finishes

- The recorded arrivals were therefore gated by the pin, not produced by the agent, which is why the offered load came out at 0.28 with 100-350 s gaps

- That trace does not describe original MLEvolve, so the run was repeated with `agent.search.parallel_search_num=2`, MLEvolve's configured default, confirmed in the log as `ThreadPool max_workers set to: 2`

> Recorder changes

- Arrival and dispatch are now separate fields: `ready_at` when the agent has code, `dispatch_at` when a slot frees, and `queue_delay_s` between them

- Each row records `concurrent_jobs`, `concurrency_degree`, and `ran_solo`, so colocation can be measured from the trace rather than assumed

- Per-process GPU attribution was attempted through NVML's compute-process list and abandoned: `nvmlDeviceGetComputeRunningProcesses` returns zero processes in this container while `nvmlDeviceGetMemoryInfo` reports 273.9 MiB in use, because the PID namespace hides them. Device-level sampling with a baseline subtraction is used instead, and is only attributable to one job on rows where `ran_solo` is true

> Trace: `traces/mlevolve_leaf_v100_mp2.jsonl`

| Quantity | Value |
|---|---|
| executions | 14 (10 non-buggy) |
| concurrency degree 1 / 2 | 7 / 7 |
| mean duration, solo vs packed | 139.3 s vs 111.3 s |
| mean device SM, solo vs packed | 33.6 % vs 17.5 % |
| mean device peak VRAM, solo vs packed | 519 vs 1058 MiB |

- Half the executions genuinely overlapped, so this trace reflects MLEvolve's own concurrency

- Device SM reaches 33.6 % on solo rows here, well above the 1.9-4.7 % seen in the pinned run, so those earlier low figures reflected which scripts that run happened to generate rather than a property of the task

- `delta_peak_vram_mib` reads 0.0 on several rows because a previous job's memory was still resident when the window opened; `device_peak_vram_mib` is the trustworthy column

> Replay, pair slowdown 1.237

| Arrivals | Policy | Makespan (s) | Mean flow (s) | Avg slowdown | vs serial |
|---|---|---|---|---|---|
| as recorded (load 0.60) | serial | 2841.0 | 155.2 | 1.00 | 1.000x |
| as recorded | time_aware | 2841.0 | 155.2 | 1.00 | 1.000x |
| as recorded | recursive_time_aware | 2749.1 | 129.6 | 1.11 | 1.033x |
| lambda = 4/min | serial | 1225.2 | 473.9 | 1.00 | 1.000x |
| lambda = 4/min | time_aware | 1225.2 | 349.5 | 1.00 | 1.000x |
| lambda = 4/min | recursive_time_aware | 786.3 | 245.4 | 1.21 | 1.558x |

- With real concurrency the recorded arrivals now carry an offered load of 0.60 rather than 0.28, so packing wins slightly (1.033x) even without re-timing, which the pinned trace could not show

- At the 4 jobs/min rate the packing policy reaches 1.558x

- Gantt chart: `records/scheduling_gantt_mlevolve_leaf_v100_mp2.png`

## Status

- Steps completed: MLEvolve runs on Nautilus without an API key, a real trace is recorded, and the trace replays through the scheduler with no predictor involved

- The task-selection requirement is answered negatively for leaf-classification and positively for the tabular-playground tasks, per the table above

## Reproduction

```bash
ssh Nautilus
```

```bash
cd /root/downeyflyfan/perfseer_test/exp_run && MLEVOLVE_TRACE_PATH=/root/downeyflyfan/mlevolve_runs/leaf_trace_v1.jsonl CUDA_VISIBLE_DEVICES=0 python3 -u run_traced.py data_dir=/root/downeyflyfan/mle-bench-data/leaf-classification/prepared/public desc_file=/root/downeyflyfan/mle-bench-data/leaf-classification/prepared/public/description.md exp_name=leaf_trace_v1 log_dir=/root/downeyflyfan/mlevolve_runs/leaf_trace_v1/logs workspace_dir=/root/downeyflyfan/mlevolve_runs/leaf_trace_v1/ws copy_data=True scheduler.enabled=false coldstart.use_coldstart=false agent.steps=14 agent.initial_drafts=3 agent.code.model=claude-sonnet agent.code.provider=claude_cli agent.feedback.model=claude-sonnet agent.feedback.provider=claude_cli agent.search.parallel_search_num=1
```

```bash
python3 -m scheduler_benchmark_test.load_mlevolve_trace traces/mlevolve_leaf_v100.jsonl
```

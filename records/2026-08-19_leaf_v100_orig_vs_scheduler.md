# leaf-classification on 1x V100: original MLEvolve vs scheduler + hardware database

- Date: 2026-08-19

- Host: Nautilus, 1x Tesla V100 32 GB, container cgroup quota 16 CPUs

- Agent backend: claude-sonnet-5 via `claude_cli` for both code and feedback

## Traces

| trace | file | scheduler |
|---|---|---|
| 1 | `traces/mlevolve_leaf_v100_orig_sonnet5.jsonl` | disabled |
| 2 | `traces/mlevolve_leaf_v100_sched_db_sonnet5.jsonl` | enabled + hardware knowledge database |

- Chart: `records/leaf_v100_orig_vs_scheduler.png`, Gantt and metric-vs-node in one image

- Metric sidecars: `records/metrics_leaf_orig.json`, `records/metrics_leaf_sched.json`

## Settings

- Identical for both runs except `scheduler.enabled`

```
data_dir=<mle-bench-data>/leaf-classification/prepared/public
copy_data=False preprocess_data=True
coldstart.use_coldstart=false
exec.timeout=1800
agent.steps=60 agent.time_limit=7200 agent.initial_drafts=3
agent.search.parallel_search_num=2
agent.code.model=claude-sonnet-5 agent.code.provider=claude_cli
agent.feedback.model=claude-sonnet-5 agent.feedback.provider=claude_cli
MLEVOLVE_GPUS=0
```

- Trace 2 adds the scheduler flags

```
scheduler.enabled=true
scheduler.settings_path=scheduler_settings_leaf.yaml
scheduler.runtime_root=scheduler_runtime_leaf
```

- `scheduler_settings_leaf.yaml` sets `prediction.mode: branch_profile`, `gpu_scheduler.mode: parallel_time_aware`, `parallel_job_cap: 2`, `memory.gpu_vram_gib: 32`, `predicted_budget_fraction: 0.945`, `live_admission_stop_fraction: 0.95`, `submission_defaults.packing_eligible: true`

- `wait_timeout_seconds` left unset, so the executor default applies: `exec.timeout * max_parallel_run + 60 = 1800 * 2 + 60 = 3660 s`

- Debug prompts are stock MLEvolve defaults and active in both runs: `debug_prob: 1`, `max_debug_depth: 20`, `back_debug_depth: 3`

## Results

| | trace 1 original | trace 2 scheduler + DB |
|---|---|---|
| nodes | 60 | 60 |
| successful | 43 (72%) | 50 (83%) |
| wall per node, median | 471 s | 1066 s |
| pure execution, median | 470 s | 681 s |
| overhead per node, median | 0.6 s | 314.9 s |
| generation per node, median | 240 s | 232 s |
| total span | 6.5 h | 13.2 h |
| peak concurrency, sweep line | 2 | 2 |
| job peak VRAM, median | 487 MiB | 412 MiB |
| best metric, log loss | 0.0001 after 40 scored | 0.0059 after 50 scored |

- Pure execution is `reported_exec_time_s`, overhead is wall minus reported

- Ratios trace 2 over trace 1: pure execution 1.45x, wall per node 2.26x, span 2.03x

- Exceptions on buggy nodes

| exception | trace 1 | trace 2 |
|---|---|---|
| RuntimeError | 11 | 3 |
| TimeoutError | 4 | 4 |
| AssertionError | 1 | 0 |
| AttributeError | 1 | 0 |
| ValueError | 0 | 3 |

- VRAM label attribution

| attribution | trace 1 | trace 2 |
|---|---|---|
| exact | 30 | 51 |
| earliest_of_2 | 11 | 6 |
| earliest_of_3 | 3 | 0 |
| no_client_observed | 16 | 3 |

## Hardware knowledge database after trace 2

- Contents of `scheduler_runtime_leaf/db/scheduler.sqlite3`

| table | rows |
|---|---|
| solo_profiles | 60 |
| pair_profiles | 0 |
| runtime_profiles | 0 |
| batch_size_observations | 0 |
| colocation_timing_profiles | 0 |

- Total `sample_count` across solo profiles: 7841

- All 60 scheduler dispatches recorded `reason="runtime estimate unavailable; exclusive fallback"` with `placement_backend="exclusive"`

## Contention curve measured separately

- Same generated script, baseline node `mlevolve_0000` from `traces/mlevolve_leaf_v100_mp2.jsonl`, run N times concurrently on one V100

| N | per-job | slowdown | throughput | speedup |
|---|---|---|---|---|
| 1 | 39 s | 1.00x | 0.026 job/s | 1.00x |
| 2 | 43 s | 1.10x | 0.047 | 1.82x |
| 4 | 47 s | 1.21x | 0.085 | 3.32x |
| 8 | 57 s | 1.46x | 0.140 | 5.48x |

- Same script solo without MPS: 38 s

- Recorded value for that node on 2026-08-08: 63 s

## Recorder changes made during this experiment

- `ProcessVramSampler` samples every card in `MLEVOLVE_GPUS` and reports the one the job ran on; it previously sampled device 0 only, so a job on any other card recorded `delta_peak_vram_mib = 0`

- `job_peak_vram_mib`, `vram_attribution` and `vram_clients_mib` added, because `delta_peak_vram_mib` is a device-level delta that counts co-runners: a 388 MiB job recorded 776 MiB with one co-runner and 1164 MiB with two

- NVML reports host-namespace pids that cannot be matched to container pids, verified both with and without MPS, so the new field attributes by NVML client arrival order within the node's window and keeps the arrival count in `vram_attribution`

- `exc_message` and `ran_on_gpu` added, so a node killed waiting in the scheduler queue with `exc_info["message"] == "scheduler wait timeout"` is separable from one that trained for its full timeout; both surface as `exc_type == "TimeoutError"`

- `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` set on every training subprocess, because protobuf 4.24.4 rejects the protoc 3.x descriptors shipped with onnx and torchvision imports onnx, so `import torchvision` and `import timm` both failed at import without it

- `assigned_gpu` and `submission_slot` recorded

## Configuration finding

- `submission_defaults.packing_eligible` defaults to `false`

- `build_mlevolve_job` computes `packing.signature` only when that flag is set, at `localml_scheduler/adapters/mlevolve.py:63-65`

- `_record_solo_profiles` skips any job missing either the flag or the signature, at `localml_scheduler/scheduler/service.py:653-656`

- With the default, no profile table can gain a row; all earlier runs in this project on cassava with 4x V100 ran with every profile table at 0

- Two gaps remain that are code rather than configuration

- `max_epochs` is `None` on submitted jobs, and `estimate_batch_options` returns `[]` when `job.max_epochs or job.config.max_epochs` is `None` at `scheduler/resource_estimator.py:83-85`, which is what produces the exclusive fallback on every dispatch; `build_mlevolve_job` accepts `max_epochs` but `SchedulerSubmissionDefaults` has no such field and the executor passes none

- `runtime_profiles` are written only by `upsert_runtime_profile` at `localml_scheduler/execution/runner_protocol.py:95`, which `examples/toy_pytorch_runner.py` and `examples/benchmark_timm_runner.py` call but `adapters/mlevolve_runner.py` does not

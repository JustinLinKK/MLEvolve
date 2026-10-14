# PetFinder Scheduler Current-VRAM Admission Fix

## Scope

- Task: PetFinder Pawpularity replay of the frozen 40-source September 2
  original MLEvolve control.
- Scheduler condition: Predictor v3.2 plus Scheduler only.  Hardware
  Knowledge Database, graph database, and preflight checker are disabled.
- Reproducibility: each source is copied byte-for-byte before execution;
  replay jobs and candidate sources use seed `42`.
- Admission: CUDA-process backend with dynamic VRAM admission only.  No fixed
  `Max Parallel Jobs` or `parallel_job_cap` is configured.

## Invalidated Partial Run

The partial run at
`/root/downeyflyfan/petfinder_seed42_pairs/scheduler_only_0902_original_baseline40_seed42_cuda_capture_20260917_032341`
was deliberately stopped after diagnosis.  Its artifacts are preserved, but it
is not a performance result.

### Root cause

`SchedulerService._active_vram_occupancy()` selected the maximum value over all
samples retained by the still-active runs.  These samples are device-wide, so
after a co-runner exited, its old peak (about 21 GB) remained in the other
run's history.  The memory gate could correctly reopen at the current lower
usage, while planning still rejected every READY job using the departed
worker's obsolete peak.

### Fix

The admission calculation now uses the newest device-wide telemetry sample;
for samples with exactly the same timestamp it retains the largest value.
Thus a completed co-runner releases capacity without weakening the current
live-VRAM safety check.

### Tests

The new regression test
`test_active_vram_occupancy_uses_current_sample_after_a_copack_exits` first
failed under the old maximum-of-history behavior (`9200 != 7000`) and passed
after the change.  The relevant scheduler suite passed: 6 passed.

## Replacement Run

- Durable output:
  `/root/downeyflyfan/petfinder_seed42_pairs/scheduler_only_0902_original_baseline40_seed42_current_vram_20260917_034500`
- Volatile runtime:
  `/dev/shm/mlevolve_petfinder_fixed_scheduler_replay_20260917_034735`
- Controller PID at launch: `781534`.
- Start configuration: 40 frozen sources; 172800-second ceiling; A10;
  Predictor v3.2 CUDA source capture; dynamic stop/resume fractions 0.90/0.85.

At the time of this record, three jobs had been admitted only after required
live epoch evidence, A10 was fully utilized, and the 90% high-memory gate was
closed.  Final speed and metric conclusions are deferred until all terminal
nodes are collected and matched against the same frozen sources.

## Live Verification of the Fix

At `2026-09-17T04:03:11Z`, frozen candidate 1 completed its expected terminal
failure (`KeyError: Pawpularity`) after nine epochs.  At `04:03:25Z`, the
10-second memory gate reopened at 0.7803.  The Scheduler dispatched candidate
4 at `04:03:25.290Z` and started it about two seconds later.  The placement
record used current live occupancy `17968 MB` plus the candidate estimate
`807.65 MB`, rather than the departed candidate's earlier roughly `21 GB`
peak.  This is the exact case that the regression test and fix target.

## Second Invalidated Partial Run: CPU Candidate Blocking GPU Work

The replacement run was stopped at `2026-09-17T04:24Z` after five terminal
nodes and a sixth active candidate.  Its durable artifact directory is the
same directory listed above.  It remains preserved for diagnosis only and is
not used in a speed or quality comparison.

### Evidence and root cause

Frozen candidate 6 (`fafc1b2982964e689fc89e2e73b25230`) is a CPU-only
LightGBM/OpenCV program.  The submitted job correctly had
`resource_requirements.requires_gpu=false` and
`predictor_not_applicable=cpu_candidate`.  While it was running, the A10
reported `3 MiB, 0%`: CUDA was idle.

The Scheduler nevertheless kept the CPU worker in the active GPU planning
set.  Its incompatible packing contract made every READY GPU job ineligible.
At the supervisor boundary the CPU worker was also represented as an
`exclusive` group, so it would reject a CUDA-process dispatch even when the
device had no GPU worker.  This caused an avoidable GPU-idle gap and
invalidated the run.

### Fix and verification

GPU admission now passes only GPU-requiring active jobs to the GPU planner.
The worker supervisor separately ignores CPU-only active groups when deciding
whether a CUDA process may occupy the GPU; unknown job metadata remains
conservatively GPU-reserving.  CPU workers retain normal lifecycle logging.

Both regressions were written before the production change and failed under
the faulty behavior:

- `test_cpu_only_active_job_does_not_block_ready_gpu_dispatch`
- `test_cpu_only_active_group_does_not_block_cuda_process_admission`

They now pass together with the live-VRAM admission tests (`8 passed`).  A
broader scheduler run currently reports four unrelated trial/probe expectation
failures outside these changed code paths; it is therefore not used as
evidence for this fix.

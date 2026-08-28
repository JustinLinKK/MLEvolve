# Petfinder task 1: Sonnet 5, scheduler, and HWKD on RTX 5090

## Scope and provenance

- Run: `20260828_152623_petfinder_sonnet5090_liveprofile` on `Justin-Linux`.
- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB total memory.
- Agent: local Claude Code subscription, canonical model `claude-sonnet-5`.
- Task: Petfinder Pawpularity task 1, image plus metadata validation RMSE.
- Scheduler: branch-profile method with `gpu_vram_gib=31` and
  `parallel_job_cap=null`; no fixed maximum parallel-job setting was used.
- Hardware knowledge: local HWKD evidence was attached to the agent context.
  The optional Neo4j graph mirrors at ports 7687/7688 were unavailable, so
  SQLite remained the scheduler source of truth.

## Fixes validated

1. `MAX_EPOCHS` is now recognized by script introspection, so the generated
   `NUM_EPOCHS=3` became scheduler `max_epochs=3` rather than `None`.
2. The MLEvolve runner now streams unbuffered stdout. At the first
   `MLEVOLVE_EPOCH_METRIC`, it persists a branch runtime profile and heartbeat
   while the script is still executing.
3. At completion, the profile is calibrated to actual wall-clock duration for
   future matching jobs. Task metric direction is now copied into scheduler
   metadata, preventing a `Final Validation Score` label from overriding the
   task's RMSE-minimization direction.
4. WSL NVIDIA telemetry discovery now supports
   `/usr/lib/wsl/lib/nvidia-smi` and `MLEVOLVE_NVIDIA_SMI`.

## Result

| Item | Value |
|---|---:|
| Candidate | `7c4876a907fb4da5a7d224ac9fda1422` |
| Model | EfficientNet-B0 tabular fusion |
| Epochs | 3 / 3 |
| Validation RMSE | **20.70016267132074** |
| Scheduler execution time | 103.259 s |
| Epoch-1 runtime estimate | 48.9999 s |
| Observed subprocess wall-clock duration | 103.259 s |
| Peak model memory emitted by candidate | 2,808 MiB |
| Submission | `submission_7c4876a907fb4da5a7d224ac9fda1422.csv` |

The first cold job necessarily used `exclusive` placement because no matching
profile existed at submission. Crucially, the profile was created before the
job completed; it is therefore available for the next same-signature branch.
The first-epoch estimate was low because it did not yet include all later
validation and submission work. The completion-calibration code was added
immediately after this run and is regression-tested; this pre-fix run itself
still stores its original 49.0-second profile.

## Artifacts

- Timeline and node metric graph:
  `records/2026-08-28_petfinder_sonnet5090_liveprofile.png`.
- Scheduler result payload:
  `records/2026-08-28_petfinder_sonnet5090_liveprofile_result.json`.
- Scheduler events:
  `records/2026-08-28_petfinder_sonnet5090_liveprofile_events.jsonl`.
- Submission snapshot:
  `records/2026-08-28_petfinder_sonnet5090_liveprofile_submission.csv`.
- Full MLEvolve trace:
  `records/2026-08-28_petfinder_sonnet5090_liveprofile.log`.

## Verification

- Targeted regression suites: 109 passed before the WSL telemetry patch.
- Hardware monitor and scheduler telemetry regressions: 7 passed after it.
- On the 5090 after deployment, the telemetry sampler returned
  `memory_total_mb=32607` and a valid live sample.
- Branch head: `d75ccb3` on `hardware-awared`.

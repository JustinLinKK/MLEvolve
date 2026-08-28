# RTX 5090 scheduler smoke validation

## Scope

- Host: `Justin-Linux` (`Justin-WS`).
- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB; CUDA compute capability 12.0.
- Upstream source snapshot: `origin/hardware-awared` at `0ac3148`.
- Scheduler: `parallel_time_aware`, branch-profile prediction, CUDA-process backend.
- Admission: incremental profile and live-VRAM telemetry only; no
  `Max Parallel Jobs` / `parallel_job_cap` is configured.

## Regression fixed before the run

`scheduler_benchmark_test/stress_bench/run_bench.py` had legacy fixed-cap
configuration on the upstream snapshot.  The local current driver removes it.
This was guarded with `test_scheduler_stress_settings_leave_parallel_cap_unset`.
The targeted regression suite passed: 2 tests.

## Runs

### Cold start: 16 distinct signatures

- Workload: the bundled stress fixture, rewritten only for the remote fixture
  location and reduced from 50 to 2 epochs per node.
- Result: 16/16 completed, no Out-Of-Memory failure.
- Makespan: 46.86 s.
- Mean training time: 0.836 s.
- All placements were exclusive because each signature had no previous runtime
  estimate. Every dispatch event explicitly recorded
  `runtime estimate unavailable; exclusive fallback`.
- Peak GPU memory: 11,152 MiB; average Streaming Multiprocessor utilization:
  15.1%; peak: 99%.

### Profiled shared-signature run: 4 nodes

- Workload: one anchor, then three same-signature nodes after the anchor could
  write its runtime profile; 8 epochs per node.
- Result: 4/4 completed, no Out-Of-Memory failure.
- Makespan: 19.86 s.
- Placement: 1 exclusive anchor and 3 CUDA-process incremental admissions.
- Mean training time: 3.666 s; peak GPU memory: 13,401 MiB.
- Mean ready-to-dispatch wait: 1.158 s. The largest 4.448 s wait was a
  bounded two-epoch live colocation trial, not CPU scheduler computation.

## Conclusion

The profile-based scheduler works on the RTX 5090 after a profile exists.
The cold-start result is intentionally conservative: distinct first-seen
signatures cannot safely co-locate without runtime estimates. The combined
Gantt and metric-node artifact is
`2026-08-28_rtx5090_scheduler_smoke.png`.

Claude Code is installed on the remote host, but its subscription OAuth login
was still awaiting the account authorization code at the time of these runs;
therefore these scheduler runs do not claim to have used Sonnet 5 as the agent.

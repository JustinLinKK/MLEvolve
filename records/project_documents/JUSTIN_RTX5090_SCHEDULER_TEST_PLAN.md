# Justin RTX 5090 Scheduler Test Plan

## Goal

Run the current `hardware-awared` scheduler on the RTX 5090 host with Claude
Sonnet 5 authenticated through the user's Claude subscription. Detect and fix
excessive one-node training time or scheduler decision latency before accepting
the run.

## Fixed execution contract

- Remote host: `Justin-Linux` (`Justin-WS`), NVIDIA GeForce RTX 5090, 32GB.
- Source revision: `origin/hardware-awared` commit `0ac3148` in a separate
  remote clone at `/home/justin/MLEvolve`; do not overwrite the dirty local tree.
- Python environment: `/home/justin/MLEvolve/.venv`, created by `uv`.
- Agent: Claude Code authenticated to a Claude subscription, Sonnet 5 model.
- Scheduler mode: production `parallel_time_aware`; do not set a fixed
  parallel-job cap.

## Execution sequence

1. Verify the RTX 5090, Python environment, repository revision, and Claude
   subscription authentication.
2. Run the scheduler benchmark preflight and capture the calibration profile.
3. Run a bounded production replay with per-node training duration and
   scheduling-decision duration records.
4. Stop on an abnormal duration, preserve logs and traces, determine the root
   cause, add a failing regression test, and make one targeted local fix.
5. Synchronize the verified fix to Justin-Linux, rerun the same workload, and
   write a detailed record and combined Gantt/metric-node image in `records/`.

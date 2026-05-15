# Nautilus Deployment Notes

These manifests keep the scheduler compliant with Nautilus-style shared-cluster expectations:

- Benchmark jobs default to `BENCH_GPU_SAMPLER=query`, which uses `nvidia-smi --query-gpu=...` in a tracked loop instead of `nohup` or global `dmon`.
- No manifest requests privileged access, mutates compute mode, or uses `sudo`.
- `benchmark-job.yaml` is the default no-MPS profile.
- `benchmark-job-mps.yaml` is opt-in and fails fast unless `nvidia-cuda-mps-control` exists in the image.
- `mlevolve-job.yaml` runs the grading server as a child process owned by the Job entrypoint, then cleans it up on exit.

Database guidance:

- Use [scheduler.nautilus.yaml](/workspaces/MLEvolve/localml_scheduler/configs/scheduler.nautilus.yaml) for SQLite-only control-plane state.
- Use [scheduler.nautilus.fullstack.yaml](/workspaces/MLEvolve/localml_scheduler/configs/scheduler.nautilus.fullstack.yaml) when you want SQLite primary plus best-effort Neo4j mirroring and Postgres analytics.
- `neo4j.yaml` is namespace-local and intentionally off the scheduler critical path.
- `postgres-cluster.yaml` targets the Zalando Postgres operator. Create a `scheduler-db-env` Secret with `LOCALML_SCHEDULER_LOG_DSN` and optionally `LOCALML_SCHEDULER_NEO4J_PASSWORD` before enabling the full-stack preset.

Expected PVCs:

- `mlevolve-datasets`
- `mlevolve-results`

Before applying:

- Replace `GIT_REPO_URL`, `GIT_REF`, and `EXP_ID` placeholders.
- Swap the base image if you already have a prebuilt CUDA runtime with this repo’s Python dependencies.
- If your Nautilus namespace uses an explicit Linstor storage class, add it to the Neo4j PVC and the Zalando Postgres volume section before deployment.

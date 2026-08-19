# Nautilus Deployment Notes

These manifests keep the scheduler compliant with Nautilus-style shared-cluster expectations:

- Benchmark jobs default to `BENCH_GPU_SAMPLER=query`, which uses `nvidia-smi --query-gpu=...` in a tracked loop instead of `nohup` or global `dmon`.
- No manifest requests privileged access, mutates compute mode, or uses `sudo`.
- `benchmark-job.yaml` is the default no-MPS profile.
- `benchmark-job-mps.yaml` is opt-in and fails fast unless `nvidia-cuda-mps-control` exists in the image.
- `mlevolve-job.yaml` runs the grading server as a child process owned by the Job entrypoint, then cleans it up on exit.

Database guidance:

- Use the root `config.yaml` or `config.example.yaml` through `MLEVOLVE_CONFIG`; scheduler runtime settings live under `scheduler.settings`.
- The Nautilus job entrypoint applies cluster-specific overrides through CLI values such as `scheduler.settings.runtime_root`, graph DB mode, backend allowlists, and Qdrant URL.
- Do not run Docker inside the MLEvolve container. Run databases as normal Kubernetes workloads (`qdrant.yaml` and `neo4j.yaml`) and connect to them through ClusterIP Services.
- `qdrant.yaml` is the namespace-local vector store for `schema/hardware_feature_records`, `schema/code_doc_chunks`, and `schema/api_symbol_chunks`.
- `knowledge-ingest-job.yaml` vectorizes the schema folders into Qdrant. It disables graph writes during ingestion so a CPU-only ingest pod does not pollute Neo4j with fake hardware.
- `neo4j.yaml` is the namespace-local graph evidence store. It can be empty at the start of a run; meaningful job/profile evidence is written by the scheduler after generated training scripts are probed or executed.
- `postgres-cluster.yaml` targets the Zalando Postgres operator. Create a `scheduler-db-env` Secret with `LOCALML_SCHEDULER_LOG_DSN` and optionally `LOCALML_SCHEDULER_NEO4J_PASSWORD` before enabling the full-stack preset.

Expected PVCs:

- `mlevolve-datasets`
- `mlevolve-results`

Before applying:

- Replace `GIT_REPO_URL`, `GIT_REF`, and `EXP_ID` placeholders.
- Swap the base image if you already have a prebuilt CUDA runtime with this repo’s Python dependencies.
- If your Nautilus namespace uses an explicit Linstor storage class, add it to the Neo4j PVC and the Zalando Postgres volume section before deployment.

Full hardware-aware database flow:

```bash
kubectl create secret generic scheduler-db-env \
  --from-literal=LOCALML_SCHEDULER_NEO4J_PASSWORD=change-me \
  --from-literal=OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f k8s/nautilus/qdrant.yaml
kubectl apply -f k8s/nautilus/neo4j.yaml
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=qdrant --timeout=300s
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=neo4j --timeout=300s

kubectl apply -f k8s/nautilus/knowledge-ingest-job.yaml
kubectl wait --for=condition=complete job/mlevolve-knowledge-ingest --timeout=1800s

kubectl apply -f k8s/nautilus/mlevolve-job.yaml
```

Keep `neo4j-auth` and `scheduler-db-env` in sync: the password after `neo4j/` in `neo4j-auth.NEO4J_AUTH` must equal `scheduler-db-env.LOCALML_SCHEDULER_NEO4J_PASSWORD`.

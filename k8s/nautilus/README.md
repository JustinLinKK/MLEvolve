# Nautilus Deployment Notes

These manifests keep the scheduler compliant with Nautilus-style shared-cluster expectations:

- Benchmark jobs default to `BENCH_GPU_SAMPLER=query`, which uses `nvidia-smi --query-gpu=...` in a tracked loop instead of `nohup` or global `dmon`.
- No manifest requests privileged access, mutates compute mode, or uses `sudo`.
- `benchmark-job.yaml` is the default no-MPS profile.
- `benchmark-job-mps.yaml` is opt-in and fails fast unless `nvidia-cuda-mps-control` exists in the image.
- `mlevolve-job.yaml` runs the grading server as a child process owned by the Job entrypoint, then cleans it up on exit.

Database guidance:

- Use the root `config.yaml` or `config.example.yaml` through `MLEVOLVE_CONFIG`; scheduler runtime settings live under `scheduler.settings`.
- The Nautilus job entrypoint applies cluster-specific overrides through CLI values such as `scheduler.settings.runtime_root`, backend allowlists, and `hardware_knowledge.settings.graph.uri`.
- Do not run Docker inside the MLEvolve container. Run the hardware knowledge database as a normal Kubernetes workload (`neo4j.yaml`) and connect to it through a ClusterIP Service.
- `knowledge-ingest-job.yaml` loads `schema/hardware_knowledge_graph.json` into the hardware Neo4j database.
- `neo4j.yaml` is the namespace-local hardware knowledge graph store. Scheduler empirical profiles remain in runtime SQLite under `db/branch_profile.sqlite3`.
- `postgres-cluster.yaml` targets the Zalando Postgres operator. Create a `scheduler-db-env` Secret with `LOCALML_SCHEDULER_LOG_DSN` before enabling the full-stack preset.

Expected PVCs:

- `mlevolve-datasets`
- `mlevolve-results`

Before applying:

- Replace `GIT_REPO_URL`, `GIT_REF`, and `EXP_ID` placeholders.
- Swap the base image if you already have a prebuilt CUDA runtime with this repo’s Python dependencies.
- If your Nautilus namespace uses an explicit Linstor storage class, add it to the Neo4j PVC and the Zalando Postgres volume section before deployment.

Full hardware-aware database flow:

```bash
kubectl create secret generic hardware-knowledge-env \
  --from-literal=HARDWARE_KNOWLEDGE_NEO4J_PASSWORD=change-me \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f k8s/nautilus/neo4j.yaml
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=neo4j --timeout=300s

kubectl apply -f k8s/nautilus/knowledge-ingest-job.yaml
kubectl wait --for=condition=complete job/mlevolve-knowledge-ingest --timeout=1800s

kubectl apply -f k8s/nautilus/mlevolve-job.yaml
```

Keep `neo4j-auth` and `hardware-knowledge-env` in sync: the password after `neo4j/` in
`neo4j-auth.NEO4J_AUTH` must equal
`hardware-knowledge-env.HARDWARE_KNOWLEDGE_NEO4J_PASSWORD`.

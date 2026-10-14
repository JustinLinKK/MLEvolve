# NVIDIA CUDA documentation enrichment

MLEvolve can enrich selected agent prompts with source-labelled results from NVIDIA's hosted `search_cuda_docs` MCP tool. The integration is disabled by default, local-first, and owned by `AgentSearch`; it is never imported by scheduler ranking, admission, placement, or execution workers.

## Policy boundary

- `debug` is the only role that may block on a hosted lookup in the default live mode, and only for an allowlisted CUDA/cuBLAS/cuDNN/MPS/architecture incident after run-memory, RAM, Redis, and Qdrant miss.
- `draft`, `improve`, and `code_review` consume cached evidence and may queue background refreshes. Their generation path never waits for NVIDIA.
- `evolution`, `fusion`, and `aggregation` continue to use the existing local hardware context only.
- Queries contain stable installed-stack applicability and a normalized incident signature. Source code, paths, datasets, job/node IDs, secrets, hostnames, raw traces, measured utilization, and measured memory values stay local.
- Retrieved evidence cannot control scheduler-owned MPS services, admission, placement, or cross-job CUDA state.

Raw verified results are stored as `code_doc_chunk_v1`. Recipe publication is a separate background operation that accepts structured JSON fields only, rejects scheduler-control content, and revalidates source provenance plus exact CUDA/driver/PyTorch/GPU/backend applicability. Markdown bullets are never treated as recipes.

## Rollout

Configure `agent.cuda_docs` in the main MLEvolve YAML. Credentials are never placed in YAML. Set the bearer credential in the environment variable named by `auth_token_env` (default `NVIDIA_CUDA_MCP_TOKEN`) before the run. If it is absent, remote access is marked unavailable without opening a connection or initiating an interactive login.

1. `off`: no service is constructed and baseline behavior is restored without cleanup.
2. `shadow`: deterministic routes are reported, with no NVIDIA connection or lookup.
3. `prefetch_only`: background prewarm is allowed; all agent requests remain local-only.
4. `debug_cached`: debug consumes local evidence and queues refreshes, but never blocks remotely.
5. `debug_live`: one eligible debug lookup may block within the configured 10-second total deadline.
6. `improve_live`: optional experimental mode; keep disabled until debug evaluation passes.

Only the configured `cuda_process` or `mps_process` backend is included in applicability and prewarm. Retired stream backends are neither queried nor prefetched.

## Cache and failure behavior

The hierarchy is run memo → bounded process TTL/LRU → existing local Redis → exact-key Qdrant chunks → hosted MCP. Redis is reused through `LOCALML_SCHEDULER_REDIS_URL` and has a separate `localml:cuda_docs:v2` namespace/capacity. Positive, stale, negative, transient-failure, and authentication TTLs are independently configurable. Local and Redis single-flight locks prevent duplicate calls; lock release uses compare-and-delete.

Redis, Qdrant, authentication, normalization, persistence, background work, and hosted-call failures all fail open. Stale evidence is served when available. The circuit opens after three retryable failures in the default 60-second window and permits one half-open probe after cooldown. Shutdown cancels queued background work without waiting for network prefetch.

## Observability and validation

`CudaDocsMetrics.snapshot()` exposes the plan's request, cache-hit, tier-latency, remote-call, stale, single-flight, circuit, prompt-size, and redaction series. Events allowlist only role, normalized topic, cache-key hash, tier/timing/status, rollout mode, and source domains. Full errors, remote responses, credentials, and code are excluded.

The pre-integration trace baseline is [cuda_docs_integration_baseline.json](../records/cuda_docs_integration_baseline.json), reproducible with:

```bash
python scripts/benchmark_cuda_docs_baseline.py \
  traces/mlevolve_leaf_v100_sched_db_sonnet5.jsonl
```

CI uses a fake hosted transport and requires neither NVIDIA authentication nor internet access. Before moving a deployment to `debug_live`, run a GPU-heavy A/B workload and require the acceptance thresholds in `docs/MLEvolve_NVIDIA_CUDA_MCP_Coding_Plan.md`: 95% local service after warmup, one hosted call per fresh canonical key, cache-hit p95 below 100 ms, non-debug critical-path hosted latency of zero, bounded timeout behavior, unchanged scheduler decisions, complete provenance/applicability, and no query leakage.

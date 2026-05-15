# Hardware-Aware Optimization for MLEvolve

This report describes how MLEvolve can become a hardware-aware training optimizer by combining two complementary knowledge sources:

1. the scheduler graph MCP database, which records empirical runtime, batch, VRAM, backend, and job-profile evidence from actual runs
2. a Qdrant-backed hardware feature vector database, which records curated accelerator capabilities, architecture details, and coding guidance
3. an optional Redis read-through cache, which keeps hot derived MCP responses in RAM so repeated agent calls avoid unnecessary graph, vector, and embedding work

The goal is not to replace validation-metric search. The goal is to make every generated training program more feasible, faster to evaluate, and easier for later agents to improve.

## Why Hardware Awareness Matters

Many MLEvolve code changes are also hardware decisions. Model family, image size, sequence length, batch size, precision, gradient accumulation, checkpointing, dataloader settings, and number of epochs all affect both score and feasibility.

The current search tree learns mainly from validation metric and bug state. That is necessary, but incomplete for GPU-heavy workloads because two candidates with similar metrics can have very different runtime, VRAM pressure, OOM risk, and scheduler packing behavior.

The scheduler graph already stores or exposes the empirical evidence needed for better choices:

- `Hardware`
- `RunProfile`
- `RuntimeProfile`
- `BatchProbeProfile`
- `BatchSizeObservation`
- `SoloProfile`
- `PacketProfile`
- backend capabilities
- safe scheduler limits

The vector database adds semantic hardware knowledge that may not yet exist in local run history. For example, an RTX 5090 record can explain that it is a Blackwell accelerator, what precision paths are reasonable to try, what should remain inference-only unless the framework supports it, and which PyTorch coding patterns usually expose the hardware well.

## Two Complementary Knowledge Layers

Scheduler graph MCP database:

- answers what actually happened on this machine
- captures resolved batch sizes, runtime estimates, peak VRAM, backend placement, OOMs, packet compatibility, and run outcomes
- should be treated as the strongest evidence when it has matching current-hardware profiles

Hardware feature vector database:

- answers what the accelerator and software stack are expected to support
- stores architecture features, framework usage guidance, precision policy, memory-transfer patterns, checkpointing practices, and source metadata
- is most useful during cold start, before the graph has enough run-specific evidence

Redis read-through cache:

- answers repeated MCP context requests quickly from RAM
- should cache derived context payloads, not replace the graph database or Qdrant
- is most useful once hardware-aware context is requested before every draft, improve, debug, evolution, fusion, and aggregation prompt

Together they produce a stronger `HardwareOptimizationBrief`:

```text
HardwareOptimizationBrief =
  scheduler graph evidence from actual runs
  + vector-retrieved hardware capability notes
  + current hardware and toolkit limits
  + Redis-cached hot context when available
  + fallback-safe coding recommendations
```

## Redis Read-Through Cache

Redis should sit in front of the graph and vector databases as a hot derived-context cache:

```text
Agent or MCP caller
  -> Redis read-through cache
  -> scheduler graph MCP database and Qdrant on cache miss
  -> compact HardwareOptimizationBrief
```

The source of truth remains:

- graph database or SQLite fallback for scheduler run evidence
- Qdrant for hardware feature vectors
- scheduler config for current limits and backend policy

Redis should cache the result of expensive or frequently repeated reads:

- `get_hardware_context(hardware_key)`
- `get_job_design_context(candidate_hash)`
- `search_hardware_features(query_hash + filters)`
- `get_hardware_feature_context(hardware_key + workload_type + model_family + framework)`
- `get_hardware_optimization_context(candidate_hash)`

The highest-value cache entry is the final `get_hardware_optimization_context(...)` payload because it avoids repeated graph queries, Qdrant vector search, and embedding calls for similar stage-agent prompts.

Redis should not aggressively cache:

- live job status
- active queue state
- recent debug or failure state
- write paths such as `record_tuning_outcome(...)`
- full Qdrant collections or large graph neighborhoods

Recommended TTLs:

| Payload | Suggested TTL | Reason |
| --- | ---: | --- |
| `hardware_context` | 10-60 minutes | Current GPU and scheduler limits rarely change during a run. |
| `hardware_feature_context` | 6-24 hours | Curated hardware capability records change slowly. |
| `search_hardware_features` | 1-24 hours | Vector feature retrieval is mostly static between corpus ingests. |
| `job_design_context` | 1-10 minutes | Scheduler profiles can update after each run. |
| `hardware_optimization_context` | 1-10 minutes | Best overall speedup, but should follow graph updates closely. |
| failed or empty database calls | 15-60 seconds | Avoid hammering unavailable services without hiding recovery for long. |

Cache keys should include:

- tool name
- normalized hardware key
- normalized candidate hash or query hash
- workload type, model family, framework, architecture, and vendor filters
- embedding model name and vector collection name
- graph schema or profile-data version
- hardware feature corpus version

Invalidate or version-bump cache keys when:

- new hardware feature records are ingested into Qdrant
- scheduler profiles are updated
- `record_tuning_outcome(...)` records a new outcome
- the embedding model, Qdrant collection, graph schema, or scheduler config changes
- hardware detection changes the current `hardware_key`

The intended behavior is read-through:

1. MCP tool receives a context request.
2. It builds a stable cache key from normalized inputs and version fields.
3. It returns the Redis payload immediately on hit.
4. On miss, it queries graph/Qdrant, builds the same public response shape, stores it with TTL, and returns it.
5. If Redis is unavailable, the system falls back to direct graph/Qdrant reads.

Redis is therefore an acceleration layer, not a correctness layer. A cache miss should be slower but equivalent; a stale cache entry should be bounded by TTL and versioned keys.

## MCP Integration

The scheduler graph MCP surface already exposes:

- `search_hardware(...)`
- `get_hardware_context(...)`
- `get_job_design_context(...)`
- `recommend_batch_size(...)`
- `recommend_epochs(...)`
- `get_packet_compatibility(...)`

The hardware feature vector layer adds:

- `search_hardware_features(...)`
- `get_hardware_feature_context(...)`
- `get_hardware_optimization_context(...)`

`get_hardware_optimization_context(candidate, limit=8)` is the preferred agent-facing entrypoint. It combines graph evidence and vector retrieval into one response containing:

- `hardware_context`
- `job_design_context`
- `feature_context`
- `combined_recommendations`
- `risk_notes`
- `evidence_refs`
- `confidence`

The MCP tools are read-only. Ingestion happens through the CLI:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
python -m localml_scheduler.cli hardware-features ingest --settings localml_scheduler/configs/scheduler.example.yaml
```

Use `--dry-run` to validate records without writing to Qdrant.

## Hardware Feature Record Shape

The curated vector corpus uses `hardware_feature_record_v1` records:

```yaml
schema_version: hardware_feature_record_v1
record_id: vendor.arch.feature.unique_id
title: Human-readable capability or coding guidance
summary_text: Short retrieval summary
detail_text: Detailed report-ready guidance
vendor: nvidia
architectures: [blackwell]
accelerator_names: [rtx_5090]
compute_capabilities: ["12.0"]
toolkits: [cuda]
frameworks: [pytorch]
workload_types: [vision_training, transformer_training]
features: [tensor_cores, amp, bf16, fp16]
recommended_patterns:
  - Use torch.amp autocast.
  - Keep batch size configurable.
avoid_patterns:
  - Do not hard-code unsupported precision modes.
source_refs:
  - title: NVIDIA GeForce RTX 5090 product page
    url: https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/
    source_type: vendor_product_page
    retrieved_or_verified_date: "2026-05-15"
confidence: 0.86
tags: [memory, precision, throughput]
```

Records should remain factual and source-metadata-heavy. They should not overclaim that a feature is safe for all training workloads.

## Recommended Agent Flow

Before each stage agent prompt, create a proposed job-design candidate:

```python
candidate = {
    "stage": stage_name,
    "task_type": inferred_task_type,
    "model_key": proposed_or_parent_model_key,
    "model_family": proposed_or_parent_model_family,
    "packing_family": proposed_or_parent_packing_family,
    "proposed_batch_size": proposed_or_parent_batch_size,
    "proposed_epochs": proposed_or_parent_epochs,
    "requires_gpu": True,
    "script_signature": proposed_or_parent_workload_signature,
    "backend_preference": proposed_backend,
    "uses_amp": parent_or_planned_amp_flag,
    "notes": short_training_plan_summary,
}
```

Then call:

```python
context = scheduler_client.get_hardware_optimization_context(candidate=candidate, limit=8)
```

Inject a compact brief into the draft, improve, debug, evolution, fusion, and aggregation prompts. The brief should include:

- current GPU name, VRAM, compute capability, toolkit, and torch version
- scheduler safe VRAM budget and backend allowlist
- recommended batch size and its evidence source
- recommended epoch count or historical median
- estimated runtime for the closest matching workload
- feature recommendations from Qdrant
- risk flags and avoid-pattern notes
- confidence score and evidence refs

The prompt should treat this as a constraint and tuning guide, not as an absolute rule. If competition score requires intentionally exceeding a recommendation, the generated plan should say why and add a fallback such as gradient accumulation, AMP, smaller image size, or shorter probing epochs.

## Stage-Specific Behavior

Draft agent:

- choose an initial model size, precision policy, batch size, and epoch budget that fit the current hardware
- prefer graph-supported defaults when confidence is high
- keep first drafts diverse by varying model family or augmentation strategy, but not by blindly exceeding memory limits

Improve agent:

- if the parent metric is promising and runtime or VRAM headroom exists, try higher resolution, larger batch, longer training, stronger augmentation, or a larger backbone
- if the parent is slow, memory-heavy, or close to scheduler limits, improve efficiency first through AMP, gradient accumulation, dataloader tuning, layer freezing, checkpointing, or smaller architecture changes
- use `recommend_batch_size(...)`, `get_runtime_estimate(...)`, and vector feature hits as tie-breakers when several improvements look equally plausible

Debug agent:

- interpret OOM, timeout, CUDA launch failures, worker crashes, and checkpoint-resume failures as hardware-aware debug cases
- lower batch size or enable accumulation when graph evidence flags an oversized proposal
- avoid applying fixes learned from other hardware unless retrieved evidence matches current hardware

Evolution and fusion agents:

- compare resource profiles across successful branches, not just validation metrics
- borrow training tactics only when graph and vector evidence are compatible with the current accelerator, toolkit, and backend
- favor fusing a high-score idea with a low-runtime or low-VRAM implementation when the final score is similar

Aggregation agent:

- synthesize root-level candidates that are both algorithmically diverse and scheduler-compatible
- use `PacketProfile` and `get_packet_compatibility(...)` evidence if the run is using packed or concurrent scheduler modes

## Search Node Metadata

Future prompt integration should store hardware evidence directly on each search node:

- `hardware_context`
- `job_design_context`
- `feature_context`
- `scheduler_risk_flags`
- `scheduler_confidence`
- `hardware_evidence_refs`
- `resolved_batch_size`
- `estimated_runtime_seconds`
- `peak_vram_mb`
- `backend_name`

This makes hardware-aware decisions visible in the journal and gives later stage agents branch-local memory about which tuning choices actually worked.

## Code Review and Evaluation

The code review agent can reject or patch code when:

- hard-coded batch size is above the graph recommendation without a fallback
- training uses GPU but does not use AMP when the brief recommends it
- dataloaders omit GPU-throughput settings such as `pin_memory`, `persistent_workers`, or `non_blocking=True`
- checkpointing is missing for long-running jobs
- the code ignores scheduler-provided batch or epoch settings
- the script uses a backend preference outside the scheduler allowlist

Evaluation should remain metric-first, but hardware-aware tie-breakers can:

- penalize OOM, timeout, or repeated hardware-risk failures
- prefer lower runtime or lower peak VRAM when validation metrics are statistically similar
- boost branches that improve metric while staying inside safe VRAM and runtime estimates
- deprioritize branches whose proposals repeatedly exceed graph recommendations without producing better metrics
- reserve exploration budget for unknown model families when graph confidence is low

## Feedback Loop

After each scheduler-backed run, MLEvolve should feed tuning outcomes back into the graph:

- resolved batch size
- chosen epochs
- backend
- validation metric
- runtime
- peak VRAM
- OOM or timeout reason
- whether AMP, gradient accumulation, checkpointing, or packing was used

The vector layer supplies architecture guidance. The graph layer keeps that guidance honest with empirical results.

The target loop is:

1. Redis returns a hot hardware optimization context when available
2. graph MCP context and vector feature retrieval fill the context on cache miss
3. the stage agent writes hardware-aware code
4. code review catches unsafe resource choices
5. the scheduler executes and profiles the run
6. result parsing and evaluation update metric and resource state
7. scheduler graph evidence and cache invalidation improve future recommendations

That turns MLEvolve from a metric-searching agent into a hardware-aware training optimizer that learns which code patterns work best on the actual GPU, toolkit, scheduler backend, and workload family it is using.

# Graph Schema Proposal For `localml_scheduler`

This folder defines a property-graph replacement for the current SQLite state in `localml_scheduler`, designed to work well with graph stores such as Neo4j and retrieval layers such as GraphRAG.

## Goal

The existing scheduler stores operational state and profiling evidence across:

- `jobs`
- `commands`
- `events`
- `checkpoints`
- `cache_entries`
- `solo_profiles`
- `pair_profiles`
- `batch_probe_profiles`
- `batch_size_observations`
- `combination_profiles`
- `runtime_profiles`

The target graph keeps all of that information, but reorganizes it into:

- stable entity nodes such as `Model`, `WorkloadSignature`, `Hardware`, and `Toolkit`
- factual observation nodes such as `RunProfile`, `BatchProbeProfile`, `RuntimeProfile`, `SoloProfile`, and `PacketProfile`
- control-plane nodes such as `Job`, `Command`, `Event`, `Checkpoint`, and `CacheEntry`

## Important Assumption

Your request mentions a "packet profile" node. The current repo does not contain a packet concept, but it does contain GPU packing profiles:

- `pair_profiles`
- `combination_profiles`

In this schema I model that as `PacketProfile`, meaning a packed or co-run profile. If you want, this can be renamed to `PackProfile` later without changing the overall design.

## How This Schema Was Constructed

I built the schema from the repo's current source of truth:

- the SQLite DDL in [localml_scheduler/storage/models.py](/workspaces/MLEvolve/localml_scheduler/storage/models.py:1)
- the persistence API in [localml_scheduler/storage/sqlite_store.py](/workspaces/MLEvolve/localml_scheduler/storage/sqlite_store.py:1)
- the domain objects in [localml_scheduler/domain/jobs.py](/workspaces/MLEvolve/localml_scheduler/domain/jobs.py:1) and [localml_scheduler/domain/profiles.py](/workspaces/MLEvolve/localml_scheduler/domain/profiles.py:1)
- hardware identity in [localml_scheduler/hardware.py](/workspaces/MLEvolve/localml_scheduler/hardware.py:1)
- telemetry and profile recording in [localml_scheduler/scheduler/telemetry.py](/workspaces/MLEvolve/localml_scheduler/scheduler/telemetry.py:1), [localml_scheduler/scheduler/service.py](/workspaces/MLEvolve/localml_scheduler/scheduler/service.py:406), [localml_scheduler/profiling/batch_probe.py](/workspaces/MLEvolve/localml_scheduler/profiling/batch_probe.py:1), and [localml_scheduler/profiling/runtime_probe.py](/workspaces/MLEvolve/localml_scheduler/profiling/runtime_probe.py:1)

The design follows five rules:

1. Preserve scheduler-native keys such as `job_id`, `hardware_key`, `signature`, `probe_key`, `profile_key`, and `combination_key`.
2. Separate dimension nodes from measurement nodes so GraphRAG can retrieve facts cleanly.
3. Keep both exact hardware scope and hardware-class scope.
4. Materialize a `RunProfile` node for every meaningful execution or probe attempt.
5. Add `summary_text` to factual nodes so they are retrieval-ready, not just scheduler-ready.

## Telemetry Reality Check

The current repo already records:

- `avg_gpu_utilization`
- `avg_memory_utilization`
- `peak_vram_mb`
- runtime timing such as startup time, epoch-1 time, and average step time

The schema also reserves fields for:

- `avg_sm_utilization`
- `avg_ram_utilization`

Those are included because they match your target graph, but today they should be populated conservatively:

- map `avg_sm_utilization` from current `avg_gpu_utilization` unless you later add lower-level SM counters
- keep `avg_ram_utilization` null unless you add explicit host RAM telemetry; `avg_memory_utilization` currently reflects GPU memory utilization

## Files

- [property_graph_schema.yaml](/workspaces/MLEvolve/schema/property_graph_schema.yaml) is the canonical graph structure.
- [neo4j_constraints.cypher](/workspaces/MLEvolve/schema/neo4j_constraints.cypher) contains Neo4j-friendly uniqueness constraints and indexes.
- [migration_mapping.md](/workspaces/MLEvolve/schema/migration_mapping.md) explains how each current SQLite table maps into the graph.
- [example_queries.cypher](/workspaces/MLEvolve/schema/example_queries.cypher) shows the kinds of queries this schema supports.

## MCP Query Contract

The graph is also exposed through `localml_scheduler.mcp_server`. In addition
to the existing job/profile/tuning helpers, the read-only aggregate MCP tools
now include:

- `search_hardware(...)`
- `get_hardware_context(...)`
- `get_job_design_context(...)`
- `search_hardware_features(...)`
- `get_hardware_feature_context(...)`
- `get_hardware_optimization_context(...)`

These tools are intentionally additive and future-facing. They provide a stable
database/MCP surface for later agent integration without requiring the agent
layer to know Neo4j queries, graph layout details, or scheduler config wiring.
The hardware feature tools combine this graph context with a Qdrant-backed
vector corpus of curated accelerator capability and coding-pattern records.

## Core Modeling Choices

- `Model` is the human-meaningful model identity such as `resnet101` or `convnext_small`.
- `WorkloadSignature` preserves the scheduler's exact packing/runtime identity from `packing.signature`.
- `Hardware` is the exact execution environment keyed by `hardware_key`.
- `Accelerator` is the reusable hardware class such as `RTX 5090` or `MI300X`.
- `Toolkit` captures runtime stack identity such as `cuda:12.4` or `rocm:6.1`.
- `RunProfile` is the atomic fact node for one run or probe attempt.
- `BatchProbeProfile`, `RuntimeProfile`, `SoloProfile`, and `PacketProfile` are reusable aggregate profile nodes for planning and retrieval.

## Recommended Migration Shape

Use a layered load order:

1. Load dimensions: `Model`, `WorkloadSignature`, `Backend`, `Toolkit`, `Accelerator`, `Hardware`, `Workflow`.
2. Load control-plane state: `Job`, `Command`, `Event`, `Checkpoint`, `CacheEntry`.
3. Materialize `RunProfile` facts from job status, probe results, runtime profile sources, telemetry summaries, and completion payloads.
4. Load aggregate profile nodes and connect them back to their originating `RunProfile` nodes where possible.
5. Populate `summary_text` on factual nodes for GraphRAG ingestion.

## Why This Fits GraphRAG Well

GraphRAG is most useful when the graph contains semantically meaningful entities plus grounded evidence nodes. This schema keeps:

- entity-centric traversal such as `Model -> RuntimeProfile -> Hardware`
- evidence-centric traversal such as `Job -> RunProfile -> Event`
- packing analysis such as `Model -> PacketProfile -> Model`
- text-rich factual nodes that can be embedded and retrieved directly

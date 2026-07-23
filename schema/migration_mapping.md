# Migration Mapping From SQLite To Evidence Graph

This mapping follows the canonical schema in `schema/graph_schema.yaml`.

SQLite/current scheduler state remains the control-plane source of truth.
Neo4j stores measured evidence only.

## Synthetic Evidence IDs

- scheduler job evidence: `scheduler_job::<job_id>`
- batch probe evidence: `batch_probe::<probe_key>`
- batch observation evidence: `batch_observation::<observation_key>`
- runtime evidence: `runtime_profile::<profile_key>`
- solo baseline evidence: `solo_profile::<hardware_key>::<signature>`
- pair packing evidence: `pair_profile::<pair_key>::<hardware_key>`
- group packing evidence: `combination_profile::<combination_key>`
- training config: `config:<sha1(canonical_config)>`

## Table Mapping

| SQLite source | Evidence graph target | Notes |
| --- | --- | --- |
| terminal `jobs` | `(:Job:SingleJob)` | Store completed/failed/cancelled training outcomes only; do not mirror pending queue state. |
| `jobs.payload_json.batch_probe.model_key` or `baseline_model_id` | `Model` | Prefer explicit batch-probe model key. |
| job runner kwargs, batch size, epochs, steps, signature | `TrainingConfig` | Immutable hash of workload/hyperparameter identity. |
| hardware detection | `Hardware` | Exact observed hardware/runtime environment keyed by `hardware_key`. |
| technology keys from metadata/backend/precision | `Technology` | Shared ontology key used by graph and vector retrieval. |
| `batch_probe_profiles` | `(:Job:SingleJob)` with `purpose=batch_size_probe` | `resolved_batch_size` and `max_safe_batch_size` are populated from the probe. |
| `batch_size_observations` | `(:Job:SingleJob)` with `purpose=batch_size_probe` | Concrete batch-size timing and memory evidence. |
| `runtime_profiles` | `(:Job:SingleJob)` with `purpose=runtime_probe` | Runtime estimates stay estimates unless a full epoch/training run was observed. |
| `solo_profiles` | `(:Job:SingleJob)` with `purpose=baseline_benchmark` | Baseline resource evidence for one model running alone. |
| `pair_profiles` | `(:Job:PackedJob)` plus two `PackedJobMember` nodes | Pair compatibility and slowdown evidence. |
| `combination_profiles` | `(:Job:PackedJob)` plus one `PackedJobMember` per signature | Group packing evidence. |

## Not Migrated To Neo4j

These remain in runtime/log storage and should not be mirrored as graph nodes:

- `commands`
- `events`
- `checkpoints`
- `cache_entries`
- raw `jobs.payload_json`
- queue status for non-terminal jobs

## Load Order

1. Optionally wipe Neo4j only when the operator passes the explicit wipe flag.
2. Apply `neo4j_constraints.cypher`.
3. Upsert `Hardware`, `Model`, `TrainingConfig`, and `Technology` dimensions as needed.
4. Upsert `SingleJob` evidence from terminal jobs, probes, observations, runtime records, and solo profiles.
5. Upsert `PackedJob` and `PackedJobMember` evidence from pair/group profiles.

Use:

```bash
python -m localml_scheduler.cli scheduler rebuild-evidence-graph --dry-run
python -m localml_scheduler.cli scheduler rebuild-evidence-graph --execute
python -m localml_scheduler.cli scheduler rebuild-evidence-graph --execute --wipe
```

The dry-run mode reports planned writes without mutating Neo4j.

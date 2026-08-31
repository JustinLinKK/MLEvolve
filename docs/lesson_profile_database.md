# Lesson Profile Database

MLEvolve has two independent knowledge domains:

| Domain | Authority | Acceleration | Purpose |
| --- | --- | --- | --- |
| Hardware knowledge | Existing curated Neo4j/schema stores | Existing hardware Qdrant collections and optional Redis namespace | Hardware capabilities, constraints, APIs, and optimization recipes |
| Lesson profiles | `lesson_profile_database/runtime/db/lesson_profiles.sqlite3` | Qdrant collection `lesson_profile_records_v1` and Redis database 1 | Evidence-backed experience learned from final-validated search nodes |

The lesson database does not replace or modify scheduler evidence, Neo4j, or task-scoped global memory. It never launches a GPU probe. The validation path only freezes a bounded evidence packet and enqueues a durable job; one daemon worker builds and publishes revisions later.

## Consistency and failure behavior

SQLite uses WAL and is authoritative for observations, leases, pending/active immutable revisions, lessons, conflicts, and the Qdrant outbox. Publication writes a pending SQLite revision, idempotently upserts deterministic Qdrant points, activates the revision transactionally, and invalidates the profile's Redis namespace. Expired leases can be reclaimed. Qdrant or Redis read failures fall back to SQLite and never affect search-node validity.

Exact retrieval requires the complete family, architecture, scheduler hardware, accelerator/slice, runtime, actual backend, and workload identity. Compatible results share the declared framework/CUDA major and family/hardware/workload boundaries but are advisory. Similar search is inspiration-only and removes numeric defaults and reusable code.

## Operations

```bash
# Start/health-check Neo4j, Qdrant, and persistent Redis.
bash docker_host_databases.sh up

# Initialize SQLite plus the Qdrant collection and payload indexes.
python -m lesson_profile_database.cli --config config.yaml init

# Inspect or operate the durable queue.
python -m lesson_profile_database.cli --config config.yaml status
python -m lesson_profile_database.cli --config config.yaml worker
python -m lesson_profile_database.cli --config config.yaml retry

# Read profiles and audit history.
python -m lesson_profile_database.cli --config config.yaml query PROFILE_KEY --role improve
python -m lesson_profile_database.cli --config config.yaml search "channel mismatch" --role debug
python -m lesson_profile_database.cli --config config.yaml revisions PROFILE_KEY
python -m lesson_profile_database.cli --config config.yaml conflicts --profile-key PROFILE_KEY
python -m lesson_profile_database.cli --config config.yaml rollback PROFILE_KEY REVISION
python -m lesson_profile_database.cli --config config.yaml benchmark PROFILE_KEY --role improve

# Idempotently replay final-validation events from completed run directories.
python -m lesson_profile_database.cli --config config.yaml replay runs/RUN_NAME
```

Replay should run on the original hardware host because the strict identity includes the detected accelerator and runtime. Records with uncertain family or incomplete actual-backend identity are skipped instead of being generalized.

## Read-only MCP

`lesson_profile_database.mcp_server.build_mcp_server()` exposes only:

- `get_family_hardware_profile`
- `search_lesson_profiles`

Writes, retries, conflict handling, and rollback remain explicit local CLI/runtime operations.

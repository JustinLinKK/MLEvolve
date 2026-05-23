# Mini Graph DB Architecture Demo — Design Spec

> Date: 2026-05-21

> Owner: yuw313@ucsd.edu

> Status: Approved (brainstorming phase)

> Target codebase: `MLEvolve_Schedule/`

## 1. Purpose

- Build a small, standalone demonstration of the graph database architecture defined by `schema/graph_schema.yaml`, `schema/neo4j_constraints.cypher`, and `schema/example_queries.cypher`.

- The demo must be readable in one sitting, exercise every node type and relationship type, and ship with verification scripts that prove the seeded graph is correct.

- The mini-project is intentionally decoupled from `localml_scheduler/` so a reader can study the schema without wading through scheduler code.

## 2. Scope and Non-Goals

### 2.1 In Scope

- Live Neo4j 5.x backend (instance already running at `bolt://127.0.0.1:7687`).

- Hand-written, minimal fixture data covering all node and relationship types defined in `schema/graph_schema.yaml`.

- Three verification scripts: constraint enforcement, type coverage, and example-query result correctness.

- JSON export of the seeded graph for `neo4j_viz/visualize.html`.

### 2.2 Out of Scope

- Real PyTorch training runs (deferred — pure demonstration only).

- Loading from `schema/hardware_feature_records/*.yaml` (deferred — hand-written fixtures keep the demo readable).

- Schema B/C/D vector-database integration.

- Schema migration or import from the legacy SQLite store.

## 3. Architecture Overview

### 3.1 Directory Layout

```
mini_graph_demo/
├── README.md              # Usage, prerequisites, expected output
├── fixtures.py            # Hand-written Hardware/Model/Tech/Job dataclasses
├── seed.py                # Connect, apply constraints, MERGE nodes, CREATE rels
├── queries.py             # Run schema/example_queries.cypher + pretty-print
├── export.py              # Dump full demo subgraph to graph.json
├── verify_constraints.py  # Verify uniqueness constraints are active + enforced
├── verify_coverage.py     # Verify every node/rel type has >= 1 instance
└── verify_queries.py      # Verify example_queries return expected row counts
```

### 3.2 Data Flow

```
fixtures.py ──▶ seed.py ──▶ Neo4j ──▶ queries.py
                            │
                            ├──▶ export.py ──▶ graph.json ──▶ neo4j_viz
                            │
                            └──▶ verify_*.py (3 scripts)
```

### 3.3 Isolation Strategy

- The local Neo4j instance is the user's primary working database, so the demo MUST NOT touch unrelated data.

- Because Neo4j Community Edition does not support multiple databases, isolation is achieved by two layers:

- Key namespacing: every primary key (`hardware_key`, `model_key`, `config_key`, `technology_key`, `job_id`, `member_id`) is prefixed with `mgd:` (mini-graph-demo). Example: `mgd:nvidia.rtx_5090.cu128`. This guarantees MERGE never collides with user-authored data that uses the same logical keys without the prefix.

- Property tag: every node and relationship sets `demo_run = true` as a secondary safety net for cleanup and verifier scoping.

- `seed.py` first runs `MATCH (n {demo_run: true}) WHERE n.<key> STARTS WITH 'mgd:' DETACH DELETE n` (key chosen per label) to clear only previous demo state. Both `demo_run` AND the `mgd:` prefix must match — defense in depth.

- Verification scripts scope every query with `WHERE n.demo_run = true` so pre-existing nodes outside the demo are ignored.

- If the user later migrates to Enterprise, swapping to a dedicated database named `mini_graph_demo` is a one-line change in `seed.py`.

## 4. Components

### 4.1 fixtures.py

- Pure Python data definitions. No I/O, no network.

- Defines dataclasses mirroring the schema's six node types, plus a builder function that returns the full demo graph as a `MiniGraphFixture` object.

#### 4.1.1 Node Counts

| Node type         | Count | Instances (key, all `mgd:`-prefixed)                                           |
| ----------------- | ----- | ------------------------------------------------------------------------------ |
| `Hardware`        | 2     | `mgd:nvidia.rtx_5090.cu128`, `mgd:nvidia.h100_sxm.cu128`                       |
| `Model`           | 3     | `mgd:resnet50.timm`, `mgd:vit_base_patch16_224.timm`, `mgd:llama2-7b.hf`       |
| `Technology`      | 4     | `mgd:pytorch_amp`, `mgd:bf16_autocast`, `mgd:tf32_matmul`, `mgd:cuda_graphs`   |
| `TrainingConfig`  | 3     | `mgd:cfg-rn50-bs256-bf16`, `mgd:cfg-vit-bs128-bf16`, `mgd:cfg-llama-bs4-bf16`  |
| `SingleJob`       | 2     | `mgd:job-001` (rn50 baseline), `mgd:job-002` (vit batch_size_probe)            |
| `PackedJob`       | 1     | `mgd:job-003` (rn50 + vit co-resident on RTX 5090)                             |
| `PackedJobMember` | 2     | `mgd:member-003-a` (rn50), `mgd:member-003-b` (vit)                            |

#### 4.1.2 Relationship Coverage

| Relationship                  | Demo instances                                          |
| ----------------------------- | ------------------------------------------------------- |
| `SINGLE_TRAINS_MODEL`         | `job-001 → resnet50`, `job-002 → vit_base`              |
| `SINGLE_USES_CONFIG`          | `job-001 → cfg-rn50-...`, `job-002 → cfg-vit-...`       |
| `JOB_USED_HARDWARE`           | `job-001,002,003 → rtx_5090` (with `resource_role`, `allocation_scope`, `memory_cap_mb=31744`) |
| `JOB_USES_TECHNOLOGY`         | `job-001 → {amp, bf16, tf32, cuda_graphs}` with `role`  |
| `HAS_PACKED_MEMBER`           | `job-003 → member-003-a`, `→ member-003-b`              |
| `MEMBER_TRAINS_MODEL`         | `member-003-a → resnet50`, `-b → vit_base`              |
| `MEMBER_USES_CONFIG`          | `-a → cfg-rn50-...`, `-b → cfg-vit-...`                 |
| `MEMBER_USES_TECHNOLOGY`      | `-a → pytorch_amp`, `-b → pytorch_amp`                  |
| `MEMBER_BASELINE_SINGLE_JOB`  | `-a → job-001`, `-b → job-002`                          |

- Total: 13 nodes, 21 relationships.

#### 4.1.3 Hardware Capability Tags (concrete)

- `nvidia.rtx_5090.cu128.technology_keys` = `[tensor_cores_5gen, fp4, fp8, fp8_e4m3, fp8_e5m2, sm_120, pcie5_x16]`

- `nvidia.h100_sxm.cu128.technology_keys` = `[tensor_cores_4gen, fp8, fp8_e4m3, fp8_e5m2, sm_90, nvlink]`

### 4.2 seed.py

- Responsibilities in order:

- Read `LOCALML_SCHEDULER_NEO4J_PASSWORD` from env (same convention as `localml_scheduler/storage/neo4j_store.py`).

- Open a Bolt session against `bolt://127.0.0.1:7687`, database `neo4j`.

- Run `MATCH (n {demo_run: true}) DETACH DELETE n` to wipe prior demo state.

- Apply `schema/neo4j_constraints.cypher` (idempotent — uses `IF NOT EXISTS`).

- `MERGE` every node from `fixtures.MiniGraphFixture` (so the script is re-runnable). Every node sets `demo_run: true`.

- `CREATE` every relationship with relationship properties as specified by the fixture. Every relationship sets `demo_run: true`.

- Print a summary: `seeded <n_nodes> nodes, <n_rels> relationships`.

### 4.3 queries.py

- Loads `schema/example_queries.cypher`, splits on `;`, and for each parameterized query injects fixture-known parameter values:

- `$model_key = "mgd:resnet50.timm"`

- `$hardware_key = "mgd:nvidia.rtx_5090.cu128"`

- `$model_family = "resnet"` (model_family is not key-prefixed; it is a coarse tag)

- `$job_id = "mgd:job-001"`

- Pretty-prints the result table for each query. Read-only — does not modify the graph.

### 4.4 export.py

- Reads all `demo_run=true` nodes and relationships, builds a JSON document compatible with `neo4j_viz/visualize.html`'s expected format (`nodes`, `edges`), and writes it to `neo4j_viz/graph.json`.

- If the viz format is ambiguous, `export.py` falls back to a generic `{"nodes": [...], "edges": [...]}` shape that the visualizer can be adjusted to read.

### 4.5 verify_constraints.py

- `SHOW CONSTRAINTS` → assert the six uniqueness constraints from `neo4j_constraints.cypher` are present (`job_id`, `hardware_key`, `model_key`, `config_key`, `technology_key`, `member_id`).

- Negative test: attempt `CREATE (:Job:SingleJob {job_id: 'mgd:job-001', demo_run: true})` (which already exists). Assert Neo4j raises a `ConstraintValidationFailed` (Python driver: `neo4j.exceptions.ConstraintError`).

- Exit 0 on success; non-zero with a one-line diagnostic per failure.

### 4.6 verify_coverage.py

- Parse `schema/graph_schema.yaml` to extract the canonical list of node labels and relationship types.

- For each node label, run `MATCH (n:<Label> {demo_run: true}) RETURN count(n) AS c` and assert `c >= 1`.

- For each relationship type, run `MATCH ()-[r:<TYPE> {demo_run: true}]->() RETURN count(r) AS c` and assert `c >= 1`.

- Print a table:

```
kind   | type                | expected_min | observed | pass
node   | Hardware            | 1            | 2        | yes
...
rel    | MEMBER_BASELINE_... | 1            | 2        | yes
```

- Exit 0 if every row passes; non-zero otherwise.

### 4.7 verify_queries.py

- Runs the four example queries with the fixture parameter values, asserting expected row counts and key field values:

| # | Query                                                   | Expected rows | Field assertions                              |
| - | ------------------------------------------------------- | ------------- | --------------------------------------------- |
| 1 | succeeded SingleJob for `mgd:resnet50.timm` on rtx5090  | 1             | `resolved_batch_size == 256`                  |
| 2 | batch_size_probe for `mgd:vit_base_*` on rtx5090        | 1             | `max_safe_batch_size > 0`                     |
| 3 | packed combos for `model_family='resnet'` on 5090       | 1             | `len(members) == 2`                           |
| 4 | retrieval keys for `mgd:job-001`                        | 1             | `hardware_key` non-empty, ≥1 technology key   |

- Each query prints `PASS` or `FAIL: <reason>`. Exit code reflects total.

## 5. Prerequisites and Usage

### 5.1 Environment

- Neo4j 5.x running at `bolt://127.0.0.1:7687` (already running on the dev machine).

- Python virtualenv: `uv venv --system-site-packages .venv && uv pip install neo4j pyyaml`.

- Password: `export LOCALML_SCHEDULER_NEO4J_PASSWORD=<password>` — same env var as the main scheduler so a working scheduler implies a working demo.

### 5.2 Run Order

```
# Wipe + reseed the demo subgraph
python -m mini_graph_demo.seed

# Exercise the example queries
python -m mini_graph_demo.queries

# Run verifications (each script exits non-zero on failure)
python -m mini_graph_demo.verify_constraints
python -m mini_graph_demo.verify_coverage
python -m mini_graph_demo.verify_queries

# (Optional) export for the visualizer
python -m mini_graph_demo.export
```

## 6. Error Handling

- Missing `LOCALML_SCHEDULER_NEO4J_PASSWORD` → `seed.py` fails fast with a one-line message naming the env var. Same behavior in all `verify_*.py` scripts.

- Neo4j unreachable → driver raises `ServiceUnavailable`; we let it propagate with a readable trace rather than catching and re-raising a custom error.

- Constraint violation during seeding (would indicate a fixtures bug) → fatal; the script exits non-zero and prints the offending node key.

## 7. Testing Strategy

- The verify scripts ARE the tests. There are no separate `pytest` tests because:

- The demo is itself a fixture, not production code.

- A second test layer that re-asserts what `verify_*.py` already asserts would duplicate intent.

- CI integration is out of scope for this iteration; the user can invoke verifiers manually.

## 8. Open Questions

- None remaining as of approval. Listed for traceability:

- Backend (real Neo4j vs. in-memory) → Real Neo4j (option A).

- Data source → Hand-written minimal fixture.

- Verification depth → Constraints + type coverage + query result correctness (recommended option).

- Isolation strategy → `demo_run = true` property on every node/rel for Neo4j Community Edition compatibility.

## 9. References

- `schema/graph_schema.yaml` — canonical node + relationship definitions

- `schema/neo4j_constraints.cypher` — uniqueness + indexes

- `schema/example_queries.cypher` — queries the demo proves correct

- `localml_scheduler/storage/neo4j_store.py` — production reference impl

- `neo4j_viz/visualize.html` — JSON consumer for `export.py`

# Mini Graph DB Architecture Demo — Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> Goal: Build a small standalone Python package `mini_graph_demo/` that seeds a representative slice of the project's graph database into the running Neo4j instance, runs the example Cypher queries, and ships three verification scripts that prove the graph is correct.

> Architecture: Hand-written fixtures dataclass → Bolt driver writes 13 nodes + 21 relationships (covering every node/rel type in `schema/graph_schema.yaml`) into the live Neo4j with `mgd:` key namespace + `demo_run=true` tag for isolation. Three verifier scripts assert uniqueness constraints, type coverage, and example-query correctness.

> Tech Stack: Python 3.12 (existing `.venv` at repo root), `neo4j==6.2.0` driver, `PyYAML`, `pytest==9.0.3`. Neo4j 5.x at `bolt://127.0.0.1:7687`. No new dependencies.

## File Map

- New files (all under repo root):

- `mini_graph_demo/__init__.py` — empty package marker

- `mini_graph_demo/fixtures.py` — pure Python dataclasses + `build_fixture()` builder

- `mini_graph_demo/_driver.py` — shared `connect()` helper (env-var lookup, Bolt driver factory) — DRY across seed/queries/export/verifiers

- `mini_graph_demo/seed.py` — clear demo subgraph → apply constraints → MERGE nodes → CREATE rels

- `mini_graph_demo/queries.py` — read `schema/example_queries.cypher` → run each with fixture params → pretty-print

- `mini_graph_demo/export.py` — dump `demo_run=true` subgraph → `neo4j_viz/graph.json`

- `mini_graph_demo/verify_constraints.py` — `SHOW CONSTRAINTS` + negative-test

- `mini_graph_demo/verify_coverage.py` — parse `schema/graph_schema.yaml` → assert ≥1 of every node/rel type

- `mini_graph_demo/verify_queries.py` — run 4 example queries with assertions on row count + key fields

- `mini_graph_demo/tests/__init__.py`

- `mini_graph_demo/tests/test_fixtures.py` — pytest unit tests for the pure-data parts (no DB needed)

- `mini_graph_demo/README.md` — short usage doc

- Modified files: none.

## Reading the Schema (engineer onboarding)

- Skim before starting:

- `schema/graph_schema.yaml` — defines 6 node types (`Job`, `Hardware`, `Model`, `Technology`, `TrainingConfig`, `SingleJob`, `PackedJob`, `PackedJobMember` — `Job` is abstract, concrete records use `:Job:SingleJob` or `:Job:PackedJob`) and 9 relationship types.

- `schema/neo4j_constraints.cypher` — 6 uniqueness constraints + several indexes.

- `schema/example_queries.cypher` — 4 example queries the demo must satisfy.

- `localml_scheduler/storage/neo4j_store.py` — production driver-usage reference. Note the env-var name `LOCALML_SCHEDULER_NEO4J_PASSWORD`.

## Environment Bootstrap (do once before Task 1)

- Verify Neo4j is up: `ss -tlnp | grep 7687` (port should be LISTEN).

- Verify password env var is set: `echo $LOCALML_SCHEDULER_NEO4J_PASSWORD | head -c 1` should print something.

- Verify venv has deps: `.venv/bin/python -c "import neo4j, yaml, pytest"` exits 0.

- If any check fails, fix before proceeding. Do not skip.

## Task 1: Project Skeleton + Package Marker

- Files:

- Create: `mini_graph_demo/__init__.py`

- Create: `mini_graph_demo/tests/__init__.py`

- [ ] Step 1: Create directory layout

```bash
mkdir -p mini_graph_demo/tests
touch mini_graph_demo/__init__.py mini_graph_demo/tests/__init__.py
```

- [ ] Step 2: Verify package importable

Run: `.venv/bin/python -c "import mini_graph_demo; print('ok')"`

Expected: `ok`

- [ ] Step 3: Commit

```bash
git add mini_graph_demo/__init__.py mini_graph_demo/tests/__init__.py
git commit -m "feat(mini-graph-demo): add package skeleton"
```

## Task 2: fixtures.py — Pure Data Module

- Files:

- Create: `mini_graph_demo/fixtures.py`

- Create: `mini_graph_demo/tests/test_fixtures.py`

- [ ] Step 1: Write the failing test

Create `mini_graph_demo/tests/test_fixtures.py`:

```python
"""Unit tests for mini_graph_demo.fixtures (no Neo4j required)."""
from __future__ import annotations

from mini_graph_demo.fixtures import (
    KEY_PREFIX,
    build_fixture,
)


def test_key_prefix_is_mgd():
    assert KEY_PREFIX == "mgd:"


def test_fixture_node_counts():
    fx = build_fixture()
    assert len(fx.hardware) == 2
    assert len(fx.models) == 3
    assert len(fx.technologies) == 4
    assert len(fx.training_configs) == 3
    assert len(fx.single_jobs) == 2
    assert len(fx.packed_jobs) == 1
    assert len(fx.packed_members) == 2


def test_fixture_relationship_counts():
    fx = build_fixture()
    assert len(fx.single_trains_model) == 2
    assert len(fx.single_uses_config) == 2
    assert len(fx.job_used_hardware) == 3
    assert len(fx.job_uses_technology) >= 4
    assert len(fx.has_packed_member) == 2
    assert len(fx.member_trains_model) == 2
    assert len(fx.member_uses_config) == 2
    assert len(fx.member_uses_technology) == 2
    assert len(fx.member_baseline_single_job) == 2


def test_all_node_keys_are_prefixed():
    fx = build_fixture()
    for node in (
        list(fx.hardware) + list(fx.models) + list(fx.technologies)
        + list(fx.training_configs) + list(fx.single_jobs)
        + list(fx.packed_jobs) + list(fx.packed_members)
    ):
        assert node.primary_key().startswith(KEY_PREFIX), node


def test_rtx_5090_capability_tags():
    fx = build_fixture()
    rtx = next(h for h in fx.hardware if "rtx_5090" in h.hardware_key)
    expected_subset = {
        "tensor_cores_5gen", "fp4", "fp8", "fp8_e4m3",
        "fp8_e5m2", "sm_120", "pcie5_x16",
    }
    assert expected_subset.issubset(set(rtx.technology_keys))


def test_h100_capability_tags():
    fx = build_fixture()
    h100 = next(h for h in fx.hardware if "h100" in h.hardware_key)
    expected_subset = {
        "tensor_cores_4gen", "fp8", "fp8_e4m3", "fp8_e5m2",
        "sm_90", "nvlink",
    }
    assert expected_subset.issubset(set(h100.technology_keys))
```

- [ ] Step 2: Run the test to verify it fails

Run: `.venv/bin/python -m pytest mini_graph_demo/tests/test_fixtures.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'mini_graph_demo.fixtures'`

- [ ] Step 3: Implement fixtures.py

Create `mini_graph_demo/fixtures.py`:

```python
"""Hand-written minimal fixture covering all node and relationship types
defined in schema/graph_schema.yaml. Pure Python — no I/O, no network.

All primary keys are prefixed with `mgd:` (mini-graph-demo) so MERGE
cannot collide with user-authored data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

KEY_PREFIX = "mgd:"


def _k(suffix: str) -> str:
    return KEY_PREFIX + suffix


@dataclass
class Hardware:
    hardware_key: str
    vendor: str
    product_name: str
    architecture: str
    compute_capability: str
    total_vram_mb: int
    vram_type: str
    technology_keys: list[str] = field(default_factory=list)

    def primary_key(self) -> str:
        return self.hardware_key


@dataclass
class Model:
    model_key: str
    model_name: str
    framework: str
    model_family: str
    architecture_type: str
    parameter_count: int

    def primary_key(self) -> str:
        return self.model_key


@dataclass
class Technology:
    technology_key: str
    name: str
    category: str

    def primary_key(self) -> str:
        return self.technology_key


@dataclass
class TrainingConfig:
    config_key: str
    input_signature: str
    batch_size: int
    precision: str
    optimizer: str
    learning_rate: float

    def primary_key(self) -> str:
        return self.config_key


@dataclass
class SingleJob:
    job_id: str
    profile_key: str
    purpose: str
    status: str
    hardware_set_key: str
    run_scope: str
    model_key: str
    config_key: str
    peak_vram_mb: int
    observed_avg_step_time_ms: float
    resolved_batch_size: int
    max_safe_batch_size: int | None = None
    confidence: float = 1.0

    def primary_key(self) -> str:
        return self.job_id


@dataclass
class PackedJob:
    job_id: str
    profile_key: str
    purpose: str
    status: str
    hardware_set_key: str
    run_scope: str
    packing_group_key: str
    packing_strategy: str
    compatible: bool
    slowdown_ratio: float
    throughput_efficiency: float
    peak_vram_mb: int

    def primary_key(self) -> str:
        return self.job_id


@dataclass
class PackedJobMember:
    member_id: str
    model_key: str
    config_key: str
    status: str
    resolved_batch_size: int
    observed_avg_step_time_ms: float
    slowdown_vs_single: float

    def primary_key(self) -> str:
        return self.member_id


@dataclass
class JobUsedHardwareEdge:
    job_id: str
    hardware_key: str
    resource_role: str
    allocation_scope: str
    memory_cap_mb: int


@dataclass
class JobUsesTechnologyEdge:
    job_id: str
    technology_key: str
    role: str


@dataclass
class MiniGraphFixture:
    hardware: list[Hardware]
    models: list[Model]
    technologies: list[Technology]
    training_configs: list[TrainingConfig]
    single_jobs: list[SingleJob]
    packed_jobs: list[PackedJob]
    packed_members: list[PackedJobMember]
    single_trains_model: list[tuple[str, str]]
    single_uses_config: list[tuple[str, str]]
    job_used_hardware: list[JobUsedHardwareEdge]
    job_uses_technology: list[JobUsesTechnologyEdge]
    has_packed_member: list[tuple[str, str]]
    member_trains_model: list[tuple[str, str]]
    member_uses_config: list[tuple[str, str]]
    member_uses_technology: list[tuple[str, str]]
    member_baseline_single_job: list[tuple[str, str]]


def build_fixture() -> MiniGraphFixture:
    hw_rtx = Hardware(
        hardware_key=_k("nvidia.rtx_5090.cu128"),
        vendor="nvidia",
        product_name="GeForce RTX 5090",
        architecture="blackwell",
        compute_capability="12.0",
        total_vram_mb=32768,
        vram_type="gddr7",
        technology_keys=[
            "tensor_cores_5gen", "fp4", "fp8", "fp8_e4m3",
            "fp8_e5m2", "sm_120", "pcie5_x16",
        ],
    )
    hw_h100 = Hardware(
        hardware_key=_k("nvidia.h100_sxm.cu128"),
        vendor="nvidia",
        product_name="H100 SXM",
        architecture="hopper",
        compute_capability="9.0",
        total_vram_mb=81920,
        vram_type="hbm3",
        technology_keys=[
            "tensor_cores_4gen", "fp8", "fp8_e4m3", "fp8_e5m2",
            "sm_90", "nvlink",
        ],
    )

    m_rn50 = Model(
        model_key=_k("resnet50.timm"),
        model_name="resnet50",
        framework="pytorch",
        model_family="resnet",
        architecture_type="cnn",
        parameter_count=25_557_032,
    )
    m_vit = Model(
        model_key=_k("vit_base_patch16_224.timm"),
        model_name="vit_base_patch16_224",
        framework="pytorch",
        model_family="vit",
        architecture_type="transformer",
        parameter_count=86_567_656,
    )
    m_llama = Model(
        model_key=_k("llama2-7b.hf"),
        model_name="llama-2-7b",
        framework="pytorch",
        model_family="llama",
        architecture_type="transformer",
        parameter_count=6_738_415_616,
    )

    t_amp = Technology(_k("pytorch_amp"), "torch.amp", "precision_optimization")
    t_bf16 = Technology(_k("bf16_autocast"), "bf16 autocast", "precision_optimization")
    t_tf32 = Technology(_k("tf32_matmul"), "TF32 matmul", "tensor_core_enablement")
    t_cg = Technology(_k("cuda_graphs"), "CUDA Graphs", "launch_overhead_reduction")

    cfg_rn50 = TrainingConfig(
        config_key=_k("cfg-rn50-bs256-bf16"),
        input_signature="image=224x224",
        batch_size=256,
        precision="bf16",
        optimizer="sgd",
        learning_rate=0.1,
    )
    cfg_vit = TrainingConfig(
        config_key=_k("cfg-vit-bs128-bf16"),
        input_signature="image=224x224",
        batch_size=128,
        precision="bf16",
        optimizer="adamw",
        learning_rate=1e-4,
    )
    cfg_llama = TrainingConfig(
        config_key=_k("cfg-llama-bs4-bf16"),
        input_signature="seq_len=2048",
        batch_size=4,
        precision="bf16",
        optimizer="adamw",
        learning_rate=2e-5,
    )

    job_001 = SingleJob(
        job_id=_k("job-001"),
        profile_key=_k("profile-rn50-rtx5090-baseline"),
        purpose="baseline_benchmark",
        status="succeeded",
        hardware_set_key=hw_rtx.hardware_key,
        run_scope="full_epoch",
        model_key=m_rn50.model_key,
        config_key=cfg_rn50.config_key,
        peak_vram_mb=22000,
        observed_avg_step_time_ms=145.0,
        resolved_batch_size=256,
        max_safe_batch_size=256,
        confidence=0.95,
    )
    job_002 = SingleJob(
        job_id=_k("job-002"),
        profile_key=_k("profile-vit-rtx5090-bsprobe"),
        purpose="batch_size_probe",
        status="succeeded",
        hardware_set_key=hw_rtx.hardware_key,
        run_scope="fixed_steps",
        model_key=m_vit.model_key,
        config_key=cfg_vit.config_key,
        peak_vram_mb=28000,
        observed_avg_step_time_ms=210.0,
        resolved_batch_size=128,
        max_safe_batch_size=192,
        confidence=0.85,
    )

    job_003 = PackedJob(
        job_id=_k("job-003"),
        profile_key=_k("profile-rn50-vit-rtx5090-mps"),
        purpose="packed_benchmark",
        status="succeeded",
        hardware_set_key=hw_rtx.hardware_key,
        run_scope="fixed_steps",
        packing_group_key=_k("pack-rn50+vit"),
        packing_strategy="mps",
        compatible=True,
        slowdown_ratio=1.18,
        throughput_efficiency=0.85,
        peak_vram_mb=30500,
    )

    member_a = PackedJobMember(
        member_id=_k("member-003-a"),
        model_key=m_rn50.model_key,
        config_key=cfg_rn50.config_key,
        status="succeeded",
        resolved_batch_size=256,
        observed_avg_step_time_ms=170.0,
        slowdown_vs_single=1.17,
    )
    member_b = PackedJobMember(
        member_id=_k("member-003-b"),
        model_key=m_vit.model_key,
        config_key=cfg_vit.config_key,
        status="succeeded",
        resolved_batch_size=128,
        observed_avg_step_time_ms=248.0,
        slowdown_vs_single=1.18,
    )

    return MiniGraphFixture(
        hardware=[hw_rtx, hw_h100],
        models=[m_rn50, m_vit, m_llama],
        technologies=[t_amp, t_bf16, t_tf32, t_cg],
        training_configs=[cfg_rn50, cfg_vit, cfg_llama],
        single_jobs=[job_001, job_002],
        packed_jobs=[job_003],
        packed_members=[member_a, member_b],
        single_trains_model=[
            (job_001.job_id, m_rn50.model_key),
            (job_002.job_id, m_vit.model_key),
        ],
        single_uses_config=[
            (job_001.job_id, cfg_rn50.config_key),
            (job_002.job_id, cfg_vit.config_key),
        ],
        job_used_hardware=[
            JobUsedHardwareEdge(job_001.job_id, hw_rtx.hardware_key,
                                "primary_accelerator", "whole_device", 31744),
            JobUsedHardwareEdge(job_002.job_id, hw_rtx.hardware_key,
                                "primary_accelerator", "whole_device", 31744),
            JobUsedHardwareEdge(job_003.job_id, hw_rtx.hardware_key,
                                "primary_accelerator", "shared_device", 31744),
        ],
        job_uses_technology=[
            JobUsesTechnologyEdge(job_001.job_id, t_amp.technology_key, "precision"),
            JobUsesTechnologyEdge(job_001.job_id, t_bf16.technology_key, "precision"),
            JobUsesTechnologyEdge(job_001.job_id, t_tf32.technology_key, "optimization"),
            JobUsesTechnologyEdge(job_001.job_id, t_cg.technology_key, "optimization"),
        ],
        has_packed_member=[
            (job_003.job_id, member_a.member_id),
            (job_003.job_id, member_b.member_id),
        ],
        member_trains_model=[
            (member_a.member_id, m_rn50.model_key),
            (member_b.member_id, m_vit.model_key),
        ],
        member_uses_config=[
            (member_a.member_id, cfg_rn50.config_key),
            (member_b.member_id, cfg_vit.config_key),
        ],
        member_uses_technology=[
            (member_a.member_id, t_amp.technology_key),
            (member_b.member_id, t_amp.technology_key),
        ],
        member_baseline_single_job=[
            (member_a.member_id, job_001.job_id),
            (member_b.member_id, job_002.job_id),
        ],
    )
```

- [ ] Step 4: Run the test to verify it passes

Run: `.venv/bin/python -m pytest mini_graph_demo/tests/test_fixtures.py -v`

Expected: 6 tests PASS.

- [ ] Step 5: Commit

```bash
git add mini_graph_demo/fixtures.py mini_graph_demo/tests/test_fixtures.py
git commit -m "feat(mini-graph-demo): add fixtures module with pure-data nodes/edges"
```

## Task 3: `_driver.py` — Shared Bolt Connection Helper

- Files:

- Create: `mini_graph_demo/_driver.py`

- [ ] Step 1: Implement the helper

Create `mini_graph_demo/_driver.py`:

```python
"""Shared Bolt driver factory.

Centralised so seed/queries/export/verifiers all read the same env var and
endpoint. If the env var is missing we fail fast with a one-line message.
"""
from __future__ import annotations

import os
import sys

from neo4j import GraphDatabase, Driver

URI = "bolt://127.0.0.1:7687"
USERNAME = "neo4j"
PASSWORD_ENV = "LOCALML_SCHEDULER_NEO4J_PASSWORD"
DATABASE = "neo4j"


def connect() -> Driver:
    password = os.environ.get(PASSWORD_ENV)
    if not password:
        sys.stderr.write(
            f"error: env var {PASSWORD_ENV} not set; "
            f"`export {PASSWORD_ENV}=<password>` and retry\n"
        )
        sys.exit(2)
    driver = GraphDatabase.driver(URI, auth=(USERNAME, password))
    driver.verify_connectivity()
    return driver


def session(driver: Driver):
    return driver.session(database=DATABASE)
```

- [ ] Step 2: Smoke-test the helper

Run:

```bash
.venv/bin/python -c "
from mini_graph_demo._driver import connect, session
d = connect()
with session(d) as s:
    rec = s.run('RETURN 1 AS x').single()
    assert rec['x'] == 1
print('driver ok')
d.close()
"
```

Expected: `driver ok`. (If env var unset, exit 2 with helpful message.)

- [ ] Step 3: Commit

```bash
git add mini_graph_demo/_driver.py
git commit -m "feat(mini-graph-demo): add shared Bolt driver helper"
```

## Task 4: seed.py — Write the Fixture into Neo4j

- Files:

- Create: `mini_graph_demo/seed.py`

- [ ] Step 1: Implement seed.py

Create `mini_graph_demo/seed.py`:

```python
"""Seed the demo subgraph into Neo4j.

Re-runnable. Wipes only prior `demo_run=true` nodes/rels then re-creates the
full fixture. Applies constraints from schema/neo4j_constraints.cypher first.
"""
from __future__ import annotations

from pathlib import Path

from mini_graph_demo._driver import connect, session
from mini_graph_demo.fixtures import (
    KEY_PREFIX,
    MiniGraphFixture,
    build_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONSTRAINTS_FILE = REPO_ROOT / "schema" / "neo4j_constraints.cypher"


def _apply_constraints(s) -> None:
    text = CONSTRAINTS_FILE.read_text(encoding="utf-8")
    for stmt in (st.strip() for st in text.split(";")):
        if stmt:
            s.run(stmt).consume()


def _clear_demo(s) -> None:
    s.run("MATCH (n {demo_run: true}) DETACH DELETE n").consume()


def _seed_nodes(s, fx: MiniGraphFixture) -> int:
    n = 0
    for h in fx.hardware:
        s.run(
            "MERGE (n:Hardware {hardware_key: $k}) "
            "SET n += $props, n.demo_run = true",
            k=h.hardware_key,
            props={
                "vendor": h.vendor,
                "product_name": h.product_name,
                "architecture": h.architecture,
                "compute_capability": h.compute_capability,
                "total_vram_mb": h.total_vram_mb,
                "vram_type": h.vram_type,
                "technology_keys": h.technology_keys,
                "hardware_kind": "gpu",
            },
        ).consume()
        n += 1
    for m in fx.models:
        s.run(
            "MERGE (n:Model {model_key: $k}) "
            "SET n += $props, n.demo_run = true",
            k=m.model_key,
            props={
                "model_name": m.model_name,
                "framework": m.framework,
                "model_family": m.model_family,
                "architecture_type": m.architecture_type,
                "parameter_count": m.parameter_count,
            },
        ).consume()
        n += 1
    for t in fx.technologies:
        s.run(
            "MERGE (n:Technology {technology_key: $k}) "
            "SET n.name = $name, n.category = $cat, n.demo_run = true",
            k=t.technology_key, name=t.name, cat=t.category,
        ).consume()
        n += 1
    for c in fx.training_configs:
        s.run(
            "MERGE (n:TrainingConfig {config_key: $k}) "
            "SET n += $props, n.demo_run = true",
            k=c.config_key,
            props={
                "input_signature": c.input_signature,
                "batch_size": c.batch_size,
                "precision": c.precision,
                "optimizer": c.optimizer,
                "learning_rate": c.learning_rate,
            },
        ).consume()
        n += 1
    for sj in fx.single_jobs:
        s.run(
            "MERGE (n:Job:SingleJob {job_id: $k}) "
            "SET n += $props, n.demo_run = true",
            k=sj.job_id,
            props={
                "profile_key": sj.profile_key,
                "purpose": sj.purpose,
                "status": sj.status,
                "hardware_set_key": sj.hardware_set_key,
                "run_scope": sj.run_scope,
                "model_key": sj.model_key,
                "config_key": sj.config_key,
                "peak_vram_mb": sj.peak_vram_mb,
                "observed_avg_step_time_ms": sj.observed_avg_step_time_ms,
                "resolved_batch_size": sj.resolved_batch_size,
                "max_safe_batch_size": sj.max_safe_batch_size,
                "confidence": sj.confidence,
            },
        ).consume()
        n += 1
    for pj in fx.packed_jobs:
        s.run(
            "MERGE (n:Job:PackedJob {job_id: $k}) "
            "SET n += $props, n.demo_run = true",
            k=pj.job_id,
            props={
                "profile_key": pj.profile_key,
                "purpose": pj.purpose,
                "status": pj.status,
                "hardware_set_key": pj.hardware_set_key,
                "run_scope": pj.run_scope,
                "packing_group_key": pj.packing_group_key,
                "packing_strategy": pj.packing_strategy,
                "compatible": pj.compatible,
                "slowdown_ratio": pj.slowdown_ratio,
                "throughput_efficiency": pj.throughput_efficiency,
                "peak_vram_mb": pj.peak_vram_mb,
            },
        ).consume()
        n += 1
    for mb in fx.packed_members:
        s.run(
            "MERGE (n:PackedJobMember {member_id: $k}) "
            "SET n += $props, n.demo_run = true",
            k=mb.member_id,
            props={
                "model_key": mb.model_key,
                "config_key": mb.config_key,
                "status": mb.status,
                "resolved_batch_size": mb.resolved_batch_size,
                "observed_avg_step_time_ms": mb.observed_avg_step_time_ms,
                "slowdown_vs_single": mb.slowdown_vs_single,
            },
        ).consume()
        n += 1
    return n


def _seed_relationships(s, fx: MiniGraphFixture) -> int:
    n = 0
    for job_id, model_key in fx.single_trains_model:
        s.run(
            "MATCH (j:SingleJob {job_id: $j}), (m:Model {model_key: $m}) "
            "MERGE (j)-[r:SINGLE_TRAINS_MODEL]->(m) SET r.demo_run = true",
            j=job_id, m=model_key,
        ).consume()
        n += 1
    for job_id, config_key in fx.single_uses_config:
        s.run(
            "MATCH (j:SingleJob {job_id: $j}), (c:TrainingConfig {config_key: $c}) "
            "MERGE (j)-[r:SINGLE_USES_CONFIG]->(c) SET r.demo_run = true",
            j=job_id, c=config_key,
        ).consume()
        n += 1
    for e in fx.job_used_hardware:
        s.run(
            "MATCH (j:Job {job_id: $j}), (h:Hardware {hardware_key: $h}) "
            "MERGE (j)-[r:JOB_USED_HARDWARE]->(h) "
            "SET r.resource_role = $role, r.allocation_scope = $scope, "
            "    r.memory_cap_mb = $cap, r.demo_run = true",
            j=e.job_id, h=e.hardware_key,
            role=e.resource_role, scope=e.allocation_scope, cap=e.memory_cap_mb,
        ).consume()
        n += 1
    for e in fx.job_uses_technology:
        s.run(
            "MATCH (j:Job {job_id: $j}), (t:Technology {technology_key: $t}) "
            "MERGE (j)-[r:JOB_USES_TECHNOLOGY]->(t) "
            "SET r.role = $role, r.demo_run = true",
            j=e.job_id, t=e.technology_key, role=e.role,
        ).consume()
        n += 1
    for pj_id, mb_id in fx.has_packed_member:
        s.run(
            "MATCH (p:PackedJob {job_id: $p}), (m:PackedJobMember {member_id: $m}) "
            "MERGE (p)-[r:HAS_PACKED_MEMBER]->(m) SET r.demo_run = true",
            p=pj_id, m=mb_id,
        ).consume()
        n += 1
    for mb_id, model_key in fx.member_trains_model:
        s.run(
            "MATCH (mb:PackedJobMember {member_id: $mb}), (m:Model {model_key: $m}) "
            "MERGE (mb)-[r:MEMBER_TRAINS_MODEL]->(m) SET r.demo_run = true",
            mb=mb_id, m=model_key,
        ).consume()
        n += 1
    for mb_id, cfg_key in fx.member_uses_config:
        s.run(
            "MATCH (mb:PackedJobMember {member_id: $mb}), (c:TrainingConfig {config_key: $c}) "
            "MERGE (mb)-[r:MEMBER_USES_CONFIG]->(c) SET r.demo_run = true",
            mb=mb_id, c=cfg_key,
        ).consume()
        n += 1
    for mb_id, tech_key in fx.member_uses_technology:
        s.run(
            "MATCH (mb:PackedJobMember {member_id: $mb}), (t:Technology {technology_key: $t}) "
            "MERGE (mb)-[r:MEMBER_USES_TECHNOLOGY]->(t) SET r.demo_run = true",
            mb=mb_id, t=tech_key,
        ).consume()
        n += 1
    for mb_id, sj_id in fx.member_baseline_single_job:
        s.run(
            "MATCH (mb:PackedJobMember {member_id: $mb}), (sj:SingleJob {job_id: $sj}) "
            "MERGE (mb)-[r:MEMBER_BASELINE_SINGLE_JOB]->(sj) SET r.demo_run = true",
            mb=mb_id, sj=sj_id,
        ).consume()
        n += 1
    return n


def main() -> int:
    driver = connect()
    try:
        with session(driver) as s:
            _clear_demo(s)
            _apply_constraints(s)
            fx = build_fixture()
            n_nodes = _seed_nodes(s, fx)
            n_rels = _seed_relationships(s, fx)
            print(f"seeded {n_nodes} nodes, {n_rels} relationships")
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] Step 2: Run seed.py

Run: `.venv/bin/python -m mini_graph_demo.seed`

Expected: `seeded 13 nodes, 21 relationships`. Exit code 0.

- [ ] Step 3: Manual smoke check — count nodes

Run:

```bash
.venv/bin/python -c "
from mini_graph_demo._driver import connect, session
d = connect()
with session(d) as s:
    rows = list(s.run('MATCH (n {demo_run: true}) RETURN count(n) AS c'))
    print('demo nodes:', rows[0]['c'])
    rows = list(s.run('MATCH ()-[r {demo_run: true}]->() RETURN count(r) AS c'))
    print('demo rels:', rows[0]['c'])
d.close()
"
```

Expected: `demo nodes: 13`, `demo rels: 21`.

- [ ] Step 4: Verify re-runnability — run seed.py again

Run: `.venv/bin/python -m mini_graph_demo.seed`

Expected: same output, no errors, idempotent. Manual count still 13/21.

- [ ] Step 5: Commit

```bash
git add mini_graph_demo/seed.py
git commit -m "feat(mini-graph-demo): seed full fixture into Neo4j"
```

## Task 5: verify_constraints.py

- Files:

- Create: `mini_graph_demo/verify_constraints.py`

- [ ] Step 1: Implement verify_constraints.py

Create `mini_graph_demo/verify_constraints.py`:

```python
"""Verify uniqueness constraints from schema/neo4j_constraints.cypher are
present AND enforced. Exit 0 on success, non-zero on any failure.

Assumes seed.py has already run.
"""
from __future__ import annotations

import sys

from neo4j.exceptions import ConstraintError

from mini_graph_demo._driver import connect, session

REQUIRED_CONSTRAINTS = {
    ("Job", "job_id"),
    ("Hardware", "hardware_key"),
    ("Model", "model_key"),
    ("TrainingConfig", "config_key"),
    ("Technology", "technology_key"),
    ("PackedJobMember", "member_id"),
}


def _present_constraints(s) -> set[tuple[str, str]]:
    rows = list(s.run("SHOW CONSTRAINTS"))
    present: set[tuple[str, str]] = set()
    for row in rows:
        d = row.data()
        labels = d.get("labelsOrTypes") or []
        props = d.get("properties") or []
        if d.get("type", "").startswith("UNIQUENESS") and labels and props:
            present.add((labels[0], props[0]))
    return present


def _check_present(s) -> list[str]:
    missing = REQUIRED_CONSTRAINTS - _present_constraints(s)
    return [f"missing UNIQUENESS constraint on ({lbl}, {prop})" for lbl, prop in missing]


def _check_negative_test(s) -> str | None:
    """Try to CREATE a duplicate job_id; expect ConstraintError."""
    try:
        s.run(
            "CREATE (:Job:SingleJob {job_id: 'mgd:job-001', demo_run: true})"
        ).consume()
    except ConstraintError:
        return None
    return ("negative test failed: duplicate job_id 'mgd:job-001' was accepted "
            "(expected ConstraintError)")


def main() -> int:
    driver = connect()
    failures: list[str] = []
    try:
        with session(driver) as s:
            failures.extend(_check_present(s))
            err = _check_negative_test(s)
            if err:
                failures.append(err)
    finally:
        driver.close()

    if failures:
        for line in failures:
            sys.stderr.write(f"FAIL: {line}\n")
        return 1
    print("verify_constraints: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] Step 2: Run verify_constraints.py

Run: `.venv/bin/python -m mini_graph_demo.verify_constraints`

Expected: `verify_constraints: PASS`. Exit 0.

- [ ] Step 3: Commit

```bash
git add mini_graph_demo/verify_constraints.py
git commit -m "feat(mini-graph-demo): add constraint verification with negative test"
```

## Task 6: verify_coverage.py

- Files:

- Create: `mini_graph_demo/verify_coverage.py`

- [ ] Step 1: Implement verify_coverage.py

Create `mini_graph_demo/verify_coverage.py`:

```python
"""Verify the seeded graph instantiates every node label and every
relationship type defined in schema/graph_schema.yaml.

Exit 0 if all rows pass; non-zero otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from mini_graph_demo._driver import connect, session

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_SCHEMA = REPO_ROOT / "schema" / "graph_schema.yaml"


def _schema_labels_and_rels() -> tuple[list[str], list[str]]:
    spec = yaml.safe_load(GRAPH_SCHEMA.read_text(encoding="utf-8"))
    nodes = spec.get("node_types", {})
    # Skip abstract labels (e.g. Job is abstract — only used as additional label)
    labels = [name for name, body in nodes.items() if not body.get("abstract")]
    rels = list(spec.get("relationship_types", {}).keys())
    return labels, rels


def _count_nodes(s, label: str) -> int:
    rec = s.run(
        f"MATCH (n:{label} {{demo_run: true}}) RETURN count(n) AS c"
    ).single()
    return rec["c"]


def _count_rels(s, rel_type: str) -> int:
    rec = s.run(
        f"MATCH ()-[r:{rel_type} {{demo_run: true}}]->() RETURN count(r) AS c"
    ).single()
    return rec["c"]


def main() -> int:
    labels, rels = _schema_labels_and_rels()
    rows: list[tuple[str, str, int, int, bool]] = []
    driver = connect()
    try:
        with session(driver) as s:
            for lbl in labels:
                obs = _count_nodes(s, lbl)
                rows.append(("node", lbl, 1, obs, obs >= 1))
            for rel in rels:
                obs = _count_rels(s, rel)
                rows.append(("rel", rel, 1, obs, obs >= 1))
    finally:
        driver.close()

    width_type = max(len(r[1]) for r in rows)
    print(f"{'kind':<5} | {'type':<{width_type}} | {'min':<3} | {'obs':<3} | pass")
    for kind, t, mn, obs, ok in rows:
        print(f"{kind:<5} | {t:<{width_type}} | {mn:<3} | {obs:<3} | "
              f"{'yes' if ok else 'NO'}")

    failed = [r for r in rows if not r[4]]
    if failed:
        sys.stderr.write(f"verify_coverage: {len(failed)} type(s) missing coverage\n")
        return 1
    print("verify_coverage: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] Step 2: Run verify_coverage.py

Run: `.venv/bin/python -m mini_graph_demo.verify_coverage`

Expected: a table where every row says `yes`, ending with `verify_coverage: PASS`. Exit 0.

- [ ] Step 3: Commit

```bash
git add mini_graph_demo/verify_coverage.py
git commit -m "feat(mini-graph-demo): add coverage verification across node/rel types"
```

## Task 7: queries.py — Pretty-print example queries

- Files:

- Create: `mini_graph_demo/queries.py`

- [ ] Step 1: Implement queries.py

Create `mini_graph_demo/queries.py`:

```python
"""Run schema/example_queries.cypher with fixture parameter values.

Read-only. Pretty-prints each query's result table.
"""
from __future__ import annotations

import re
from pathlib import Path

from mini_graph_demo._driver import connect, session

REPO_ROOT = Path(__file__).resolve().parents[1]
QUERIES_FILE = REPO_ROOT / "schema" / "example_queries.cypher"

PARAMS = {
    "model_key": "mgd:resnet50.timm",
    "hardware_key": "mgd:nvidia.rtx_5090.cu128",
    "model_family": "resnet",
    "job_id": "mgd:job-001",
}


def _split_queries(text: str) -> list[str]:
    # Strip // comment lines and split on `;` outside strings (queries do not
    # use embedded semicolons in string literals).
    cleaned: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//"):
            continue
        cleaned.append(line)
    body = "\n".join(cleaned)
    return [q.strip() for q in body.split(";") if q.strip()]


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("  (no rows)")
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  " + " | ".join(c.ljust(widths[c]) for c in cols))
    print("  " + "-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  " + " | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def main() -> int:
    queries = _split_queries(QUERIES_FILE.read_text(encoding="utf-8"))
    driver = connect()
    try:
        with session(driver) as s:
            for i, q in enumerate(queries, 1):
                print(f"\n=== Query {i} ===")
                print(re.sub(r"^", "  ", q, flags=re.MULTILINE))
                print("--- result ---")
                rows = [rec.data() for rec in s.run(q, **PARAMS)]
                _print_table(rows)
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] Step 2: Run queries.py

Run: `.venv/bin/python -m mini_graph_demo.queries`

Expected: 4 query blocks printed, each with at least 1 result row (Query 1 returns the resnet50 baseline, Query 2 the vit batch_size_probe, Query 3 the packed combo, Query 4 the retrieval keys for job-001).

- [ ] Step 3: Commit

```bash
git add mini_graph_demo/queries.py
git commit -m "feat(mini-graph-demo): add example-query runner with pretty-printed results"
```

## Task 8: verify_queries.py

- Files:

- Create: `mini_graph_demo/verify_queries.py`

- [ ] Step 1: Implement verify_queries.py

Create `mini_graph_demo/verify_queries.py`:

```python
"""Verify each example query returns the expected row count and field values.

Exit 0 if all queries pass; non-zero otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

from mini_graph_demo._driver import connect, session
from mini_graph_demo.queries import PARAMS, _split_queries

REPO_ROOT = Path(__file__).resolve().parents[1]
QUERIES_FILE = REPO_ROOT / "schema" / "example_queries.cypher"


def _expectation_q1(rows: list[dict]) -> str | None:
    if len(rows) != 1:
        return f"expected 1 row, got {len(rows)}"
    r = rows[0]
    if r.get("j.resolved_batch_size") != 256:
        return f"expected resolved_batch_size=256, got {r.get('j.resolved_batch_size')}"
    if r.get("j.peak_vram_mb") is None:
        return "peak_vram_mb is None"
    return None


def _expectation_q2(rows: list[dict]) -> str | None:
    if len(rows) != 1:
        return f"expected 1 row, got {len(rows)}"
    mx = rows[0].get("j.max_safe_batch_size")
    if not (isinstance(mx, int) and mx > 0):
        return f"max_safe_batch_size not > 0, got {mx!r}"
    return None


def _expectation_q3(rows: list[dict]) -> str | None:
    if len(rows) != 1:
        return f"expected 1 row, got {len(rows)}"
    members = rows[0].get("members") or []
    if len(members) != 2:
        return f"expected 2 members, got {len(members)}"
    return None


def _expectation_q4(rows: list[dict]) -> str | None:
    if len(rows) != 1:
        return f"expected 1 row, got {len(rows)}"
    r = rows[0]
    if not r.get("hardware_key"):
        return "hardware_key empty"
    tech = r.get("technology_keys") or r.get("technologies") or []
    # Query 4 may return either a collected list of technology keys or
    # rolled-up structure; accept any non-empty collection.
    if not tech:
        return "no technology keys returned"
    return None


EXPECTATIONS = [_expectation_q1, _expectation_q2, _expectation_q3, _expectation_q4]


def main() -> int:
    queries = _split_queries(QUERIES_FILE.read_text(encoding="utf-8"))
    if len(queries) < 4:
        sys.stderr.write(
            f"FAIL: example_queries.cypher has {len(queries)} queries, expected >= 4\n"
        )
        return 1

    driver = connect()
    failed = 0
    try:
        with session(driver) as s:
            for i, q in enumerate(queries[:4]):
                rows = [rec.data() for rec in s.run(q, **PARAMS)]
                err = EXPECTATIONS[i](rows)
                if err:
                    print(f"Query {i+1}: FAIL: {err}")
                    failed += 1
                else:
                    print(f"Query {i+1}: PASS")
    finally:
        driver.close()

    if failed:
        return 1
    print("verify_queries: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] Step 2: Run verify_queries.py

Run: `.venv/bin/python -m mini_graph_demo.verify_queries`

Expected: `Query 1: PASS`, `Query 2: PASS`, `Query 3: PASS`, `Query 4: PASS`, `verify_queries: PASS`. Exit 0.

If a query fails because the example_queries.cypher returns column names different from what the expectation function reads, fix the expectation function to read the actual column name (read the cypher line in `schema/example_queries.cypher` to see the exact `RETURN` clause). Do NOT modify the cypher file.

- [ ] Step 3: Commit

```bash
git add mini_graph_demo/verify_queries.py
git commit -m "feat(mini-graph-demo): add example-query verification"
```

## Task 9: export.py — JSON dump for the visualizer

- Files:

- Create: `mini_graph_demo/export.py`

- [ ] Step 1: Inspect the visualizer's expected format

Run: `head -80 neo4j_viz/visualize.html` and look for the `fetch('graph.json')` call and any `nodes[...].id`, `edges[...].source` references. If the format is unclear or absent, write a generic `{nodes: [...], edges: [...]}` shape; the visualizer can be adjusted later.

- [ ] Step 2: Implement export.py

Create `mini_graph_demo/export.py`:

```python
"""Dump the demo subgraph (demo_run=true) to neo4j_viz/graph.json.

Output schema:
  {
    "nodes": [{"id": <internal_id>, "key": <primary_key>, "labels": [...], "props": {...}}, ...],
    "edges": [{"source": <node_id>, "target": <node_id>, "type": "...", "props": {...}}, ...]
  }
"""
from __future__ import annotations

import json
from pathlib import Path

from mini_graph_demo._driver import connect, session

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "neo4j_viz" / "graph.json"

PRIMARY_KEY_FIELDS = (
    "job_id", "hardware_key", "model_key", "config_key",
    "technology_key", "member_id",
)


def _node_key(props: dict) -> str:
    for f in PRIMARY_KEY_FIELDS:
        if f in props and props[f] is not None:
            return str(props[f])
    return ""


def main() -> int:
    driver = connect()
    try:
        with session(driver) as s:
            node_rows = list(s.run(
                "MATCH (n {demo_run: true}) "
                "RETURN id(n) AS nid, labels(n) AS labels, properties(n) AS props"
            ))
            edge_rows = list(s.run(
                "MATCH (a {demo_run: true})-[r {demo_run: true}]->(b {demo_run: true}) "
                "RETURN id(a) AS src, id(b) AS dst, type(r) AS rtype, "
                "properties(r) AS props"
            ))

        nodes = [
            {
                "id": rec["nid"],
                "key": _node_key(rec["props"]),
                "labels": rec["labels"],
                "props": rec["props"],
            }
            for rec in node_rows
        ]
        edges = [
            {
                "source": rec["src"],
                "target": rec["dst"],
                "type": rec["rtype"],
                "props": rec["props"],
            }
            for rec in edge_rows
        ]

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(
            json.dumps({"nodes": nodes, "edges": edges}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"exported {len(nodes)} nodes, {len(edges)} edges -> {OUTPUT_PATH}")
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] Step 3: Run export.py

Run: `.venv/bin/python -m mini_graph_demo.export`

Expected: `exported 13 nodes, 21 edges -> .../neo4j_viz/graph.json`. File exists and is valid JSON.

- [ ] Step 4: Validate the JSON

Run:

```bash
.venv/bin/python -c "
import json, pathlib
p = pathlib.Path('neo4j_viz/graph.json')
data = json.loads(p.read_text())
assert len(data['nodes']) == 13, len(data['nodes'])
assert len(data['edges']) == 21, len(data['edges'])
print('graph.json ok')
"
```

Expected: `graph.json ok`.

- [ ] Step 5: Commit

```bash
git add mini_graph_demo/export.py neo4j_viz/graph.json
git commit -m "feat(mini-graph-demo): export demo subgraph to neo4j_viz/graph.json"
```

## Task 10: README + final end-to-end run

- Files:

- Create: `mini_graph_demo/README.md`

- [ ] Step 1: Write the README

Create `mini_graph_demo/README.md`:

````markdown
# mini_graph_demo

- Small, standalone demonstration of the graph database architecture defined under `schema/`.

- Seeds 13 nodes + 21 relationships into the running Neo4j instance, covers every node and relationship type in `schema/graph_schema.yaml`, and ships three verification scripts.

## Prerequisites

- Neo4j 5.x running at `bolt://127.0.0.1:7687`.

- `export LOCALML_SCHEDULER_NEO4J_PASSWORD=<password>`.

- Repo-root venv with `neo4j` and `pyyaml` (already present at `.venv`).

## Usage

```bash
# Seed the demo subgraph (re-runnable, idempotent)
.venv/bin/python -m mini_graph_demo.seed

# Pretty-print example queries
.venv/bin/python -m mini_graph_demo.queries

# Run verifications (each exits non-zero on failure)
.venv/bin/python -m mini_graph_demo.verify_constraints
.venv/bin/python -m mini_graph_demo.verify_coverage
.venv/bin/python -m mini_graph_demo.verify_queries

# Optional: dump to neo4j_viz/graph.json for the visualizer
.venv/bin/python -m mini_graph_demo.export
```

## Isolation

- All primary keys are prefixed with `mgd:` and every node/rel sets `demo_run = true`.

- The cleanup query in `seed.py` only deletes nodes that have `demo_run = true`, so unrelated data in the local Neo4j is untouched.

## Files

- `fixtures.py` — pure-data dataclasses + `build_fixture()`

- `_driver.py` — Bolt connection helper

- `seed.py` / `queries.py` / `export.py` — write / read / dump

- `verify_constraints.py` / `verify_coverage.py` / `verify_queries.py` — three verifiers

- `tests/test_fixtures.py` — pytest unit tests for the pure-data parts
````

- [ ] Step 2: Run the full pipeline end-to-end

Run:

```bash
.venv/bin/python -m mini_graph_demo.seed && \
.venv/bin/python -m mini_graph_demo.verify_constraints && \
.venv/bin/python -m mini_graph_demo.verify_coverage && \
.venv/bin/python -m mini_graph_demo.verify_queries && \
.venv/bin/python -m mini_graph_demo.export
```

Expected: every step prints PASS / its expected line, total exit code 0.

- [ ] Step 3: Run pytest unit tests

Run: `.venv/bin/python -m pytest mini_graph_demo/tests/ -v`

Expected: 6 tests PASS.

- [ ] Step 4: Commit

```bash
git add mini_graph_demo/README.md
git commit -m "docs(mini-graph-demo): add README with usage instructions"
```

## Self-Review Notes

- Spec §3.1 directory layout: covered by Tasks 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.

- Spec §3.3 isolation strategy (mgd: prefix + demo_run): enforced in Task 2 (fixtures), Task 4 (seed wipes only `demo_run=true`), Tasks 5/6/7/8/9 scope every query by `demo_run`.

- Spec §4.1 node/rel coverage (13 nodes, 21 rels, all 9 rel types): wired in Task 2 fixture and asserted by Task 6 verify_coverage.

- Spec §4.2 seed.py responsibilities: covered step-by-step in Task 4 (env var via _driver, wipe, constraints, MERGE, CREATE, summary).

- Spec §4.5 negative constraint test: explicit in Task 5.

- Spec §4.6 type coverage from yaml: explicit in Task 6 — parses `schema/graph_schema.yaml`, skips abstract `Job` label.

- Spec §4.7 query-result assertions: explicit in Task 8 with per-query expectation functions.

- Spec §5 prerequisites + usage: documented in Task 10 README.

- Placeholder scan: no TBD / TODO / "implement later" present. Every step includes the actual code.

- Type consistency: `MiniGraphFixture` field names used in Task 4 match those defined in Task 2. `connect()` / `session()` signatures match between `_driver.py` and call sites.

- Open risk: Task 8 expectation functions read column names like `j.resolved_batch_size`. If `schema/example_queries.cypher` uses aliases the expectation must match — Task 8 Step 2 includes guidance to read the cypher's `RETURN` clause and align the expectation reader if needed. No cypher edits required.

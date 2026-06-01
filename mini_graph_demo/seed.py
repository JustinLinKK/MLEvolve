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
CONSTRAINTS_FILE = REPO_ROOT / "schema" / "job_evidence" / "neo4j_constraints.cypher"


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

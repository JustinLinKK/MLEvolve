from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from agents.lesson_context import LessonPromptContext, apply_lesson_context_to_node, lesson_context_instructions
from engine.execution import validate_executed_node
from engine.search_node import SearchNode
from lesson_profile_database.builder import LessonBuilder
from lesson_profile_database.benchmark import benchmark_retrieval
from lesson_profile_database.client import LessonProfileClient
from lesson_profile_database.config import LessonProfileSettings
from lesson_profile_database.evidence import bounded_parent_diff, build_evidence_packet, redact_text
from lesson_profile_database.identity import build_profile_identity, canonical_model_family
from lesson_profile_database.models import ProfileIdentity
from lesson_profile_database.registry import LessonProfileRegistry
from lesson_profile_database.vector_store import LessonVectorStore, _PAYLOAD_INDEXES
from lesson_profile_database.worker import LessonBuilderWorker


CODE = """import torch
MODEL_FAMILY = "resnet50"
MODEL_NAME = "resnet50"
BATCH_SIZE = 32
EPOCHS = 4
IMG_SIZE = 224
NUM_WORKERS = 2
GRADIENT_ACCUMULATION_STEPS = 1
"""

HARDWARE = {
    "hardware_key": "host-a10-runtime-key",
    "gpu_name": "NVIDIA A10",
    "total_vram_mb": 23000,
    "compute_capability": "8.6",
    "cuda_runtime": "12.1",
    "torch_version": "2.5.1",
}


class FakeEmbedding:
    dimension = 4

    def __init__(self):
        self.calls = 0

    def encode(self, texts, show_progress_bar=False):
        del show_progress_bar
        self.calls += 1
        return np.asarray([[float(len(text) % 7), 1.0, 0.0, 0.5] for text in texts], dtype=np.float32)


class FakeQdrant:
    def __init__(self):
        self.exists = False
        self.indexes = []
        self.points = []

    def collection_exists(self, name):
        del name
        return self.exists

    def create_collection(self, **kwargs):
        self.exists = True
        self.collection_args = kwargs

    def delete_collection(self, name):
        del name
        self.exists = False
        self.points.clear()

    def create_payload_index(self, **kwargs):
        self.indexes.append(kwargs["field_name"])

    def upsert(self, **kwargs):
        by_id = {str(point.id): point for point in self.points}
        for point in kwargs["points"]:
            by_id[str(point.id)] = point
        self.points = list(by_id.values())

    @staticmethod
    def _matches(payload, query_filter):
        for condition in getattr(query_filter, "must", []) or []:
            actual = payload.get(condition.key)
            match = condition.match
            expected = getattr(match, "value", None)
            choices = getattr(match, "any", None)
            if choices is not None:
                if actual not in choices:
                    return False
            elif isinstance(actual, list):
                if expected not in actual:
                    return False
            elif actual != expected:
                return False
        return True

    def set_payload(self, **kwargs):
        for point in self.points:
            if self._matches(point.payload, kwargs["points"]):
                point.payload.update(kwargs["payload"])

    def query_points(self, **kwargs):
        matches = [
            SimpleNamespace(payload=point.payload, score=0.9)
            for point in self.points
            if self._matches(point.payload, kwargs["query_filter"])
        ]
        return SimpleNamespace(points=matches[: kwargs["limit"]])


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.sorted_sets = {}
        self.expirations = {}

    def ping(self):
        return True

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, ex=None):
        self.values[key] = value
        self.expirations[key] = ex

    def zadd(self, key, values):
        self.sorted_sets.setdefault(key, {}).update(values)

    def zrem(self, key, *members):
        for member in members:
            self.sorted_sets.setdefault(key, {}).pop(member, None)

    def zcard(self, key):
        return len(self.sorted_sets.get(key, {}))

    def zrange(self, key, start, end):
        members = [item[0] for item in sorted(self.sorted_sets.get(key, {}).items(), key=lambda item: item[1])]
        if end == -1:
            return members[start:]
        return members[start:end + 1]

    def delete(self, *keys):
        for key in keys:
            self.values.pop(key, None)
            self.sorted_sets.pop(key, None)


def identity() -> ProfileIdentity:
    result = build_profile_identity(
        code=CODE,
        hardware=HARDWARE,
        backend="exclusive",
        task_description="image classification",
    )
    assert result is not None
    return result


def evidence(profile_identity: ProfileIdentity, *, node_id="node-1", run_id="run-1", outcome="valid", stage="draft"):
    return {
        "schema_version": "lesson-observation-v1",
        "run_id": run_id,
        "node_id": node_id,
        "stage": stage,
        "generation_strategy": "",
        "source_node_ids": [],
        "identity": profile_identity.to_dict(),
        "outcome": outcome,
        "validation": {"is_buggy": outcome != "valid", "is_valid": outcome == "valid", "metric": 0.81, "metric_maximize": True},
        "code": {
            "normalized_signature": "sig",
            "structural": {"hash": "structural-one", "operators": ["call:Conv2d"]},
            "introspection": {
                "model_key": "resnet50",
                "proposed_batch_size": 32,
                "proposed_epochs": 4,
                "gradient_accumulation_steps": 1,
                "num_workers": 2,
                "uses_amp": True,
            },
        },
        "parent_code": {},
        "delta": {
            "unified_diff": "--- parent.py\n+++ child.py\n+layer = Conv2d(64, 64, 3)",
            "added_lines": 1,
            "removed_lines": 0,
            "material_groups": 1,
            "change_scope": "one_layer",
            "diff_truncated": False,
        },
        "artifacts": {"review_issues": []},
        "scheduler_measurements": {"resolved_batch_size": 32, "runtime_seconds": 12.5, "peak_vram_mb": 1000},
        "evidence_refs": [f"node:{node_id}", f"run:{run_id}"],
    }


def summary_generator(packet, draft):
    del draft
    stage_to_type = {
        "draft": "family_baseline",
        "improve": "modification",
        "debug": "verified_fix",
        "evolution": "branch_trajectory",
        "fusion": "transfer",
        "fusion_draft": "cross_branch_consensus",
    }
    return {
        "baseline_summary": "Validated family hardware baseline.",
        "lesson_summaries": [{
            "lesson_type": stage_to_type.get(packet.get("stage"), "family_baseline"),
            "lesson": "Reuse this validated pattern with current constraints.",
            "evidence_refs": [packet["evidence_refs"][0]],
        }],
    }


def make_client(tmp_path: Path, *, redis=None, qdrant_enabled=False, qdrant=None, embedding=None):
    settings = LessonProfileSettings(runtime_root=str(tmp_path))
    settings.qdrant.enabled = qdrant_enabled
    return LessonProfileClient(
        settings,
        redis_client=redis,
        qdrant_client=qdrant,
        embedding_model=embedding,
        summary_generator=summary_generator,
    )


def enqueue(client, profile_identity, *, node_id="node-1", run_id="run-1", outcome="valid", stage="draft"):
    return client.registry.enqueue_observation(
        identity=profile_identity,
        evidence=evidence(profile_identity, node_id=node_id, run_id=run_id, outcome=outcome, stage=stage),
        outcome=outcome,
        run_id=run_id,
        node_id=node_id,
        extractor_version=client.settings.builder.extractor_version,
    )


def test_identity_is_strict_canonical_and_uncertain_family_cold_starts():
    first = identity()
    second = identity()
    assert first.profile_key == second.profile_key
    assert first.model_family == "resnet"
    assert first.architecture_type == "cnn"
    assert first.framework_major == "pytorch-2"
    assert first.cuda_major == "cuda-12"
    assert canonical_model_family("cnn") == (None, 0.0)
    assert build_profile_identity(
        code="import torch\nMODEL_FAMILY='cnn'",
        hardware=HARDWARE,
        backend="exclusive",
        task_description="image classification",
    ) is None
    assert build_profile_identity(
        code=CODE,
        hardware=HARDWARE,
        backend=None,
        task_description="image classification",
    ) is None


def test_evidence_redacts_secrets_bounds_diff_and_requires_references():
    assert "secret-value" not in redact_text("API_KEY=secret-value", limit=100)
    node = SimpleNamespace(
        id="node-a",
        parent=None,
        code=CODE,
        stage="draft",
        plan="API_KEY=secret-value\n" + "x" * 5000,
        prompt_input="Task: classify images\nAPI_KEY=secret-value\nunrestricted prose",
        pipeline_decision={},
        hardware_decision={},
        stage_note_board=[],
        review_issues=[],
        review_history=[],
        bug_report="",
        fix_report="",
        term_out="",
        metric=SimpleNamespace(value=0.8, maximize=True),
        is_buggy=False,
        is_valid=True,
        generation_strategy="",
        source_node_ids=[],
    )
    packet = build_evidence_packet(
        node=node,
        identity=identity().to_dict(),
        outcome="valid",
        run_id="run-a",
        task_description="image classification",
    )
    serialized = json.dumps(packet)
    assert "secret-value" not in serialized
    assert "unrestricted prose" not in serialized
    assert packet["prompt"]["sha256"]
    assert packet["evidence_refs"] == ["node:node-a", "run:run-a"]


def test_structural_delta_distinguishes_controlled_and_multi_change():
    parent = CODE + "\nlayer = torch.nn.Linear(4, 2)\n"
    child = parent + "extra = torch.nn.Conv2d(4, 4, 3)\n"
    controlled = bounded_parent_diff(parent, child)
    assert controlled["change_scope"] == "one_layer"
    assert controlled["change_action"] == "add"
    assert controlled["layer_type"] == "conv2d"
    multi = bounded_parent_diff(parent, child.replace("IMG_SIZE = 224", "IMG_SIZE = 512"))
    assert multi["change_scope"] == "multi_change"
    assert multi["controlled"] is False


def test_concurrent_duplicate_claim_and_expired_lease_recovery(tmp_path):
    client = make_client(tmp_path)
    client.registry.initialize()
    profile_identity = identity()
    results = []

    def claim():
        results.append(enqueue(client, profile_identity))

    threads = [threading.Thread(target=claim) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(bool(item["inserted"]) for item in results) == 1
    first = client.registry.lease_next_job(worker_id="one", lease_seconds=300)
    assert first is not None
    with client.registry.transaction() as connection:
        connection.execute("UPDATE builder_jobs SET lease_expires_at=?", (time.time() - 1,))
    recovered = client.registry.lease_next_job(worker_id="two", lease_seconds=300)
    assert recovered["job_id"] == first["job_id"]
    assert recovered["lease_owner"] == "two"


def test_end_to_end_publication_cache_hit_and_compatible_downgrade(tmp_path):
    redis = FakeRedis()
    qdrant = FakeQdrant()
    embedding = FakeEmbedding()
    client = make_client(
        tmp_path,
        redis=redis,
        qdrant_enabled=True,
        qdrant=qdrant,
        embedding=embedding,
    )
    client.registry.initialize()
    exact = identity()
    enqueue(client, exact)
    assert client.worker.process_once()
    active = client.registry.active_revision(exact.profile_key)
    assert active is not None and active["state"] == "active"
    assert active["maturity"] == "provisional"
    assert qdrant.points

    first = client.get_family_hardware_profile(agent_role="draft", identity=exact, code=CODE)
    view = first["family_hardware_profile"]
    assert view["match_level"] == "exact"
    assert view["source"] == "sqlite_qdrant"
    assert view["revision"] == 1
    assert len(view["relevant_lessons"]) <= 3
    assert len(json.dumps(first)) <= client.settings.max_prompt_chars

    original_profile = client.registry.profile
    client.registry.profile = lambda key: (_ for _ in ()).throw(AssertionError("SQLite called on cache hit"))
    second = client.get_family_hardware_profile(agent_role="draft", identity=exact, code=CODE)
    assert second["family_hardware_profile"]["source"] == "redis"
    client.registry.profile = original_profile
    assert all(value == 300 for value in redis.expirations.values())

    different = exact.to_dict()
    different["hardware_key"] = "different-runtime-hardware-key"
    different["runtime_class"] = "pytorch-2-cuda-12-minor-drift"
    different["profile_key"] = "different-exact-key"
    compatible = client.get_family_hardware_profile(agent_role="draft", identity=different, code=CODE)
    assert compatible["family_hardware_profile"]["match_level"] == "compatible"
    assert any("Advisory compatible" in warning for warning in compatible["family_hardware_profile"]["warnings"])

    report = benchmark_retrieval(client, identity=exact.to_dict(), agent_role="draft", iterations=3)
    assert report["acceptance"] is True
    assert report["warm_io_calls"] == {"sqlite": 0, "embedding": 0, "qdrant": 0}


def test_negative_result_cache_avoids_sqlite_and_profile_publish_invalidates_it(tmp_path):
    redis = FakeRedis()
    client = make_client(tmp_path, redis=redis)
    client.registry.initialize()
    missing = identity().to_dict()
    missing["profile_key"] = "missing-profile"
    first = client.get_family_hardware_profile(agent_role="debug", identity=missing, error="oom")
    assert first["family_hardware_profile"]["match_level"] == "none"
    original_profile = client.registry.profile
    client.registry.profile = lambda key: (_ for _ in ()).throw(AssertionError("negative cache missed"))
    second = client.get_family_hardware_profile(agent_role="debug", identity=missing, error="oom")
    assert second["family_hardware_profile"]["source"] == "redis"
    client.registry.profile = original_profile
    assert redis.values
    client.invalidate_profile("another-profile")
    assert not redis.values


def test_distinct_runs_promote_stable_and_rollback_is_immutable(tmp_path):
    client = make_client(tmp_path)
    client.registry.initialize()
    profile_identity = identity()
    for index in range(3):
        enqueue(client, profile_identity, node_id=f"node-{index}", run_id=f"run-{index}")
        assert client.worker.process_once()
    profile = client.registry.profile(profile_identity.profile_key)
    assert profile["maturity"] == "stable"
    revisions = client.registry.list_revisions(profile_identity.profile_key)
    assert len(revisions) == 3
    first_baseline = revisions[-1]["baseline"]
    publication = client.registry.rollback(profile_identity.profile_key, 1)
    client.vector_store.upsert_publication(publication["payload"])
    activated = client.registry.activate_publication(publication["outbox_id"])
    assert activated["revision_number"] == 4
    assert client.registry.active_revision(profile_identity.profile_key)["baseline"] == first_baseline
    assert len(client.registry.list_revisions(profile_identity.profile_key)) == 4


def test_all_agent_roles_receive_only_their_projection_and_provenance(tmp_path):
    client = make_client(tmp_path)
    client.registry.initialize()
    exact = identity()
    stages = ["draft", "improve", "debug", "evolution", "fusion", "fusion_draft"]
    for index, stage in enumerate(stages):
        enqueue(client, exact, node_id=f"role-node-{index}", run_id=f"role-run-{index}", stage=stage)
        client.worker.process_once()
    expected = {
        "draft": {"family_baseline"},
        "improve": {"modification"},
        "debug": {"verified_fix"},
        "evolution": {"branch_trajectory"},
        "fusion": {"transfer"},
        "aggregation": {"cross_branch_consensus"},
        "review": {"implementation_contract"},
    }
    for role, allowed in expected.items():
        result = client.get_family_hardware_profile(agent_role=role, identity=exact, code=CODE)
        view = result["family_hardware_profile"]
        assert view["match_level"] == "exact"
        assert view["maturity"] == "stable"
        assert len(view["relevant_lessons"]) <= 3
        assert {item["lesson_type"] for item in view["relevant_lessons"]} <= allowed

    review = client.get_family_hardware_profile(agent_role="review", identity=exact, code=CODE)
    context = LessonPromptContext("review", review, "section")
    node = SimpleNamespace(pipeline_decision={"evidence": {}})
    apply_lesson_context_to_node(node, context)
    assert node.lesson_profile_key == exact.profile_key
    assert node.lesson_profile_revision == 6
    assert node.pipeline_decision["evidence"]["lesson_profile_used"] is True
    assert lesson_context_instructions(context)


def test_failure_cannot_create_baseline_and_controlled_contradiction_is_preserved(tmp_path):
    client = make_client(tmp_path)
    client.registry.initialize()
    profile_identity = identity()
    enqueue(client, profile_identity, node_id="failed-first", outcome="missing_submission")
    client.worker.process_once()
    assert client.registry.profile(profile_identity.profile_key) is None

    enqueue(client, profile_identity, node_id="success", run_id="run-success")
    client.worker.process_once()
    enqueue(client, profile_identity, node_id="failed-later", run_id="run-failure", outcome="zero_metric")
    client.worker.process_once()
    conflicts = client.registry.list_conflicts(profile_identity.profile_key)
    assert conflicts
    assert client.registry.profile(profile_identity.profile_key)["maturity"] == "conflicted"


def test_qdrant_collection_indexes_and_deterministic_point_ids(tmp_path):
    from qdrant_client import models

    settings = LessonProfileSettings(runtime_root=str(tmp_path))
    fake = FakeQdrant()
    embedding = FakeEmbedding()
    store = LessonVectorStore(
        settings,
        qdrant_client=fake,
        qdrant_models=models,
        embedding_model=embedding,
    )
    initialized = store.ensure_collection()
    assert initialized["ok"]
    assert set(fake.indexes) == set(_PAYLOAD_INDEXES)
    assert store.point_id("profile", 1, "lesson", "id") == store.point_id("profile", 1, "lesson", "id")
    payload = {
        "profile_key": identity().profile_key,
        "revision_number": 1,
        "identity": identity().to_dict(),
        "baseline": {"model_summary": "known good"},
        "trust": {"confidence": 0.5, "evidence_refs": ["node:n"]},
        "maturity": "provisional",
        "lessons": [],
    }
    store.upsert_publication(payload)
    assert len(fake.points) == 1
    assert embedding.calls == 1


def test_similar_search_removes_numeric_defaults_and_code(tmp_path):
    client = make_client(tmp_path)
    client.registry.initialize()
    exact = identity()
    enqueue(client, exact, stage="improve")
    client.worker.process_once()
    revision = client.registry.active_revision(exact.profile_key)
    lesson_id = client.registry.lessons_for_revision(exact.profile_key, revision["revision_number"], role="improve")[0]["lesson_id"]
    client.vector_store.search = lambda **kwargs: [{
        "profile_key": exact.profile_key,
        "revision": revision["revision_number"],
        "record_id": lesson_id,
        "lesson_type": "modification",
        "maturity": "provisional",
        "content": {
            "lesson": "Use batch 128 safely as inspiration",
            "physical_batch_size": 128,
            "implementation_example": {"code": "unsafe()"},
        },
        "confidence": 0.4,
        "evidence_refs": ["node:n"],
    }]
    result = client.search_lesson_profiles(query="add a block", agent_role="improve")
    serialized = json.dumps(result)
    assert result[0]["match_level"] == "similar"
    assert "physical_batch_size" not in serialized
    assert "implementation_example" not in serialized
    assert "unsafe()" not in serialized
    assert "batch 128" not in serialized

    missing = exact.to_dict()
    missing["profile_key"] = "no-exact-or-compatible-profile"
    missing["accelerator_key"] = "different-accelerator"
    similar_view = client.get_family_hardware_profile(
        agent_role="improve",
        identity=missing,
        code="add a convolution block",
    )
    assert similar_view["family_hardware_profile"]["match_level"] == "similar"

    client.invalidate_profile(exact.profile_key)
    client.vector_store.search = lambda **kwargs: (_ for _ in ()).throw(ConnectionError("offline"))
    fallback = client.search_lesson_profiles(query="validated pattern fallback", agent_role="improve")
    assert fallback
    assert fallback[0]["source"] == "sqlite_fallback"


def test_validation_hook_runs_only_from_final_validator_and_invalid_is_advisory(tmp_path):
    calls = []
    lesson_client = SimpleNamespace(
        enqueue_validated_node=lambda agent, node, outcome: calls.append((node.id, outcome)) or {"ok": True}
    )
    cfg = SimpleNamespace(workspace_dir=tmp_path)
    agent = SimpleNamespace(
        cfg=cfg,
        lesson_profile_client=lesson_client,
        branch_successful_nodes={},
        pipeline_logger=None,
    )
    invalid = SearchNode(code=CODE, plan="p", stage="draft", is_buggy=True)
    validate_executed_node(agent, invalid)
    assert calls == [(invalid.id, "skipped_buggy")]

    valid = SearchNode(
        code=CODE,
        plan="p",
        stage="draft",
        is_buggy=False,
        is_valid=True,
        metric=SimpleNamespace(value=0.8, maximize=True),
    )
    submission = tmp_path / "submission" / f"submission_{valid.id}.csv"
    submission.parent.mkdir(parents=True)
    submission.write_text("id,pred\n1,0\n")
    validate_executed_node(agent, valid)
    assert calls[-1] == (valid.id, "valid")


def test_unsupported_llm_numeric_claim_fails_job_without_activation(tmp_path):
    settings = LessonProfileSettings(runtime_root=str(tmp_path))
    settings.qdrant.enabled = False
    registry = LessonProfileRegistry(settings)
    registry.initialize()
    profile_identity = identity()
    registry.enqueue_observation(
        identity=profile_identity,
        evidence=evidence(profile_identity),
        outcome="valid",
        run_id="run-1",
        node_id="node-1",
        extractor_version=settings.builder.extractor_version,
    )
    builder = LessonBuilder(
        settings,
        registry,
        summary_generator=lambda packet, draft: {
            "baseline_summary": "Use batch 999.",
            "lesson_summaries": [],
        },
    )
    worker = LessonBuilderWorker(settings, registry, builder, LessonVectorStore(settings))
    worker.process_once()
    assert registry.profile(profile_identity.profile_key) is None
    assert registry.status()["builder_jobs"].get("queued") == 1


def test_outbox_retry_reuses_frozen_revision_without_rebuilding(tmp_path):
    calls = {"summary": 0, "qdrant": 0}

    def counted_summary(packet, draft):
        calls["summary"] += 1
        return summary_generator(packet, draft)

    class FlakyVector:
        def upsert_publication(self, payload):
            del payload
            calls["qdrant"] += 1
            if calls["qdrant"] == 1:
                raise ConnectionError("qdrant unavailable")
            return {"ok": True}

    settings = LessonProfileSettings(runtime_root=str(tmp_path))
    settings.builder.retry_delay_seconds = 0
    registry = LessonProfileRegistry(settings)
    registry.initialize()
    builder = LessonBuilder(settings, registry, summary_generator=counted_summary)
    worker = LessonBuilderWorker(settings, registry, builder, FlakyVector())
    exact = identity()
    registry.enqueue_observation(
        identity=exact,
        evidence=evidence(exact),
        outcome="valid",
        run_id="run-1",
        node_id="node-1",
        extractor_version=settings.builder.extractor_version,
    )
    worker.process_once()
    assert registry.active_revision(exact.profile_key) is None
    assert registry.status()["qdrant_outbox"]["pending"] == 1
    worker.process_once()
    assert registry.active_revision(exact.profile_key) is not None
    assert calls == {"summary": 1, "qdrant": 2}


def test_late_publication_cannot_replace_a_newer_active_revision(tmp_path):
    settings = LessonProfileSettings(runtime_root=str(tmp_path))
    registry = LessonProfileRegistry(settings)
    registry.initialize()
    exact = identity()
    publications = []
    for index in range(2):
        observation = registry.enqueue_observation(
            identity=exact,
            evidence=evidence(exact, node_id=f"node-{index}", run_id=f"run-{index}"),
            outcome="valid",
            run_id=f"run-{index}",
            node_id=f"node-{index}",
            extractor_version=settings.builder.extractor_version,
        )
        publications.append(registry.prepare_revision(
            identity=exact.to_dict(),
            observation_id=observation["observation_id"],
            baseline={"model_summary": f"revision {index + 1}"},
            trust={"confidence": 0.5, "evidence_refs": [f"node:node-{index}"]},
            maturity="provisional",
            lessons=[],
            builder_model="fake",
            builder_prompt_version="v1",
            extractor_version="v1",
        ))
    registry.activate_publication(publications[1]["outbox_id"])
    late = registry.activate_publication(publications[0]["outbox_id"])
    assert late["superseded_publication"] == 1
    assert registry.active_revision(exact.profile_key)["revision_number"] == 2

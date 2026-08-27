from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from agents import code_review_agent
from config import SchedulerBridgeConfig
from engine.search_node import SearchNode
from localml_scheduler.backend_mode import normalize_packing_backend
from localml_scheduler.client import SchedulerClient
from localml_scheduler.hardware_knowledge.feature_filter import PIPELINE_STAGES
from localml_scheduler.storage.neo4j_store import Neo4jStateStore


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_deleted_compatibility_modules_stay_absent() -> None:
    removed_paths = (
        "hardware_knowledge_graph/feature_filter.py",
        "hardware_knowledge_graph/records.py",
        "hardware_knowledge_graph/store.py",
        "localml_scheduler/migrations/backend_mode_v2.py",
        "schema/job_evidence/property_graph_schema.yaml",
    )

    assert all(not (REPO_ROOT / relative_path).exists() for relative_path in removed_paths)


def test_runtime_exposes_only_canonical_contracts() -> None:
    scheduler_fields = {field.name for field in fields(SchedulerBridgeConfig)}
    search_node_fields = {field.name for field in fields(SearchNode)}

    assert "settings_path" not in scheduler_fields
    assert "active_profile_key" not in search_node_fields
    assert not hasattr(code_review_agent, "run")
    assert PIPELINE_STAGES == (
        "model_design",
        "datatype_precision",
        "training_evaluation",
    )

    for method_name in (
        "search_hardware_features",
        "get_hardware_feature_context",
        "get_hardware_optimization_context",
        "migrate_backend_modes",
    ):
        assert not hasattr(SchedulerClient, method_name)

    for method_name in (
        "submit_job",
        "get_job",
        "upsert_runtime_profile",
    ):
        assert not hasattr(Neo4jStateStore, method_name)

    with pytest.raises(ValueError, match="Unsupported packing backend"):
        normalize_packing_backend("mps")

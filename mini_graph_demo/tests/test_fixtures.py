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

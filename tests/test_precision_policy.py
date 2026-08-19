from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.hardware_context import HardwarePromptContext, compact_optimization_context
from agents.precision_validation import validate_training_precision
from localml_scheduler.hardware_knowledge.feature_filter import (
    query_hardware_features,
    query_hardware_node,
)
from localml_scheduler.client import SchedulerClient
from schema.build_cypher_from_json import emit_cypher
from utils.precision_policy import precision_feature_visibility, resolve_precision_policy


ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "schema" / "hardware_knowledge_graph.json"

BASE = {"fp32", "disabled"}


@pytest.mark.parametrize(
    ("architecture", "normal", "aggressive"),
    [
        ("volta", BASE | {"fp16_amp"}, BASE | {"fp16_amp"}),
        ("turing", BASE | {"fp16_amp"}, BASE | {"fp16_amp"}),
        ("ampere", BASE | {"tf32", "bf16_amp", "fp16_amp"}, BASE | {"tf32", "bf16_amp", "fp16_amp"}),
        ("ada_lovelace", BASE | {"tf32", "bf16_amp", "fp16_amp"}, BASE | {"tf32", "bf16_amp", "fp16_amp", "fp8_te"}),
        ("hopper", BASE | {"tf32", "bf16_amp", "fp16_amp"}, BASE | {"tf32", "bf16_amp", "fp16_amp", "fp8_te"}),
        (
            "blackwell",
            BASE | {"tf32", "bf16_amp", "fp16_amp"},
            BASE | {"tf32", "bf16_amp", "fp16_amp", "fp8_te", "mxfp8_te", "nvfp4_te"},
        ),
    ],
)
def test_full_architecture_mode_matrix(
    architecture: str, normal: set[str], aggressive: set[str]
) -> None:
    normal_policy = resolve_precision_policy(
        {"architecture": architecture, "datatypes": ["int8", "fp4", "fp64"]},
        mode="normal",
    )
    aggressive_policy = resolve_precision_policy(
        {"architecture": architecture, "datatypes": ["int8", "fp4", "fp64"]},
        mode="aggressive",
    )

    assert set(normal_policy.allowed_policies) == normal
    assert set(aggressive_policy.allowed_policies) == aggressive
    assert "fp6" not in normal_policy.allowed_policies
    assert "fp6" not in aggressive_policy.allowed_policies
    assert precision_feature_visibility("int8", normal_policy) == "integer_indicator"
    assert precision_feature_visibility("fp4", aggressive_policy) == "hidden"
    assert precision_feature_visibility("fp64", aggressive_policy) == "hidden"


def test_aggressive_low_precision_is_permitted_but_not_recommended() -> None:
    policy = resolve_precision_policy({"architecture": "blackwell"}, mode="aggressive")

    assert {"fp8", "mxfp8", "nvfp4"} <= set(policy.permitted_features)
    assert not ({"fp8", "mxfp8", "nvfp4"} & set(policy.recommended_features))
    assert precision_feature_visibility("fp8", policy) == "permitted"
    assert precision_feature_visibility("mxfp8", policy) == "permitted"
    assert precision_feature_visibility("nvfp4", policy) == "permitted"


def test_invalid_precision_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="precision_optimization_mode"):
        resolve_precision_policy({"architecture": "blackwell"}, mode="turbo")


@pytest.mark.parametrize(
    ("gpu", "mode", "present", "absent"),
    [
        ("NVIDIA Tesla V100 32GB", "normal", {"fp16"}, {"bf16", "tf32", "fp8", "mxfp8", "nvfp4", "fp64"}),
        ("NVIDIA T4", "normal", {"fp16", "int8", "int4"}, {"bf16", "tf32", "fp8", "mxfp8", "nvfp4"}),
        ("GeForce RTX 4090", "normal", {"fp16", "bf16", "tf32", "int8"}, {"fp8", "mxfp8", "nvfp4", "fp4"}),
        ("GeForce RTX 4090", "aggressive", {"fp8", "fp8_e4m3", "fp8_e5m2"}, {"mxfp8", "nvfp4", "fp4"}),
        ("NVIDIA H100 PCIe 80GB", "aggressive", {"fp8", "fp8_e4m3", "fp8_e5m2"}, {"mxfp8", "nvfp4", "fp4"}),
        ("GeForce RTX 5090", "normal", {"fp16", "bf16", "tf32", "int8"}, {"fp8", "mxfp8", "nvfp4", "fp4", "fp64"}),
        ("GeForce RTX 5090", "aggressive", {"fp8", "mxfp8", "nvfp4", "int8"}, {"fp4", "fp64"}),
    ],
)
def test_static_hardware_filter_obeys_architecture_and_mode(
    gpu: str, mode: str, present: set[str], absent: set[str]
) -> None:
    result = query_hardware_features(
        gpu,
        "datatype_precision",
        precision_mode=mode,
    )
    features = {item["feature_id"]: item for item in result["features"]}

    assert result["found"] is True
    assert present <= set(features)
    assert not (absent & set(features))
    for integer in {"int8", "int4"} & set(features):
        assert features[integer]["recommended"] is False
        assert features[integer]["prompt_visibility"] == "integer_indicator"
    for opt_in in {"fp8", "mxfp8", "nvfp4"} & set(features):
        assert features[opt_in]["recommended"] is False
        assert features[opt_in]["prompt_visibility"] == "permitted"


def test_compaction_retains_opt_in_formats_and_integer_capability_indicators() -> None:
    stage_context = SchedulerClient._stage_feature_context_from_static_graph(
        hardware_name="GeForce RTX 5090",
        stages=["datatype_precision"],
        limit=20,
        precision_mode="aggressive",
    )
    compact = compact_optimization_context(
        {"stage_hardware_features": stage_context}
    )
    features = {
        feature["feature_id"]: feature
        for stage in compact["stage_hardware_features"]["stages"]
        for feature in stage.get("features", [])
    }

    assert {"fp8", "mxfp8", "nvfp4", "int8"} <= set(features)
    assert features["fp8"]["prompt_visibility"] == "permitted"
    assert features["mxfp8"]["recommended"] is False
    assert features["nvfp4"]["recommended"] is False
    assert features["int8"]["prompt_visibility"] == "integer_indicator"


def test_normal_mode_removes_aggressive_recommendation_text_from_stage_node() -> None:
    normal = query_hardware_node(
        "GeForce RTX 5090",
        "datatype_precision",
        precision_mode="normal",
    )
    aggressive = query_hardware_node(
        "GeForce RTX 5090",
        "datatype_precision",
        precision_mode="aggressive",
    )

    normal_text = " ".join(normal.get("recommended_patterns", [])).lower()
    aggressive_text = " ".join(aggressive.get("recommended_patterns", [])).lower()
    assert "mxfp8" not in normal_text
    assert "nvfp4" not in normal_text
    assert "transformer engine fp8" not in normal_text
    assert "mxfp8" in aggressive_text
    assert "nvfp4" in aggressive_text


def test_hardware_graph_format_metadata_and_cypher_mirror_are_complete() -> None:
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]}
    feature_ids = {
        node["properties"].get("feature_id")
        for node in graph["nodes"]
        if node.get("label") == "Feature"
    }

    assert {"feat:mxfp8", "feat:nvfp4", "feat:int4"} <= set(nodes)
    assert {
        "hw:nvidia.volta.tesla_v100_32gb.spec",
        "hw:nvidia.turing.tesla_t4.spec",
    } <= set(nodes)
    assert "fp6" not in feature_ids
    assert all(str(edge.get("to") or "").lower() != "feat:fp6" for edge in graph["edges"])
    assert all(
        "fp6" not in {str(value).lower() for value in node.get("properties", {}).get("datatypes", [])}
        for node in graph["nodes"]
        if node.get("label") == "Hardware"
    )

    metadata_fields = {
        "min_compute_capability",
        "native_tensor_core_evidence",
        "training_backend",
        "optimization_modes",
        "prompt_visibility",
        "model_shape_limitations",
    }
    for feature_id in ("amp", "fp16", "bf16", "tf32", "fp8", "fp8_e4m3", "fp8_e5m2", "mxfp8", "nvfp4"):
        assert metadata_fields <= set(nodes[f"feat:{feature_id}"]["properties"])

    assert nodes["feat:fp4"]["properties"]["prompt_visibility"] == "hidden"
    assert nodes["feat:fp64"]["properties"]["prompt_visibility"] == "hidden"
    assert nodes["feat:int8"]["properties"]["prompt_visibility"] == "capability_indicator"
    assert nodes["feat:int4"]["properties"]["prompt_visibility"] == "capability_indicator"
    assert "HYBRID" in nodes["feat:fp8_e5m2"]["properties"]["description"]

    cypher = (ROOT / "schema" / "hardware_knowledge_graph.cypher").read_text(encoding="utf-8")
    assert cypher == emit_cypher(graph["nodes"], graph["edges"])


def _agent(mode: str) -> SimpleNamespace:
    return SimpleNamespace(acfg=SimpleNamespace(precision_optimization_mode=mode))


def _context(architecture: str) -> HardwarePromptContext:
    return HardwarePromptContext(
        compact_context={
            "hardware_context": {
                "hardware": {
                    "architecture": architecture,
                    "compute_capability": "10.0" if architecture == "blackwell" else "8.9",
                }
            }
        }
    )


def test_deterministic_validation_rejects_generic_fp4_and_normal_mode_fp8() -> None:
    generic = validate_training_precision(
        _agent("aggressive"),
        'PRECISION = "fp4"\n',
        context=_context("blackwell"),
    )
    fp8 = validate_training_precision(
        _agent("normal"),
        "import transformer_engine.pytorch as te\nwith te.fp8_autocast(enabled=True):\n    pass\n",
        context=_context("ada_lovelace"),
    )

    assert generic and generic[0].severity == "critical"
    assert generic[0].category == "datatype_precision"
    assert fp8 and "normal mode" in fp8[0].evidence


def test_aggressive_fp8_and_nvfp4_require_explicit_transformer_engine_recipe() -> None:
    fp8_code = (
        "import transformer_engine.pytorch as te\n"
        "from transformer_engine.common.recipe import DelayedScaling, Format\n"
        "model = te.TransformerLayer(hidden_size=128, ffn_hidden_size=512, num_attention_heads=4)\n"
        "recipe = DelayedScaling(fp8_format=Format.HYBRID)\n"
        "with te.fp8_autocast(enabled=True, fp8_recipe=recipe):\n    loss = model(x)\n"
    )
    nvfp4_code = (
        "import transformer_engine.pytorch as te\n"
        "from transformer_engine.common.recipe import NVFP4BlockScaling\n"
        "model = te.TransformerLayer(hidden_size=128, ffn_hidden_size=512, num_attention_heads=4)\n"
        "recipe = NVFP4BlockScaling()\n"
        "with te.fp8_autocast(enabled=True, fp8_recipe=recipe):\n    loss = model(x)\n"
    )

    assert validate_training_precision(_agent("aggressive"), fp8_code, context=_context("ada_lovelace")) == ()
    assert validate_training_precision(_agent("aggressive"), nvfp4_code, context=_context("blackwell")) == ()
    assert validate_training_precision(
        _agent("aggressive"),
        "import transformer_engine.pytorch as te\nwith te.fp8_autocast(enabled=True):\n    pass\n",
        context=_context("ada_lovelace"),
    )
    assert validate_training_precision(
        _agent("aggressive"),
        'PRECISION = "nvfp4"\n',
        context=_context("blackwell"),
    )

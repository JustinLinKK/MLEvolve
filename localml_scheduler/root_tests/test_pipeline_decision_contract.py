from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.prompts.pipeline_decision import (
    PIPELINE_STAGE_ORDER,
    apply_pipeline_decision_to_node,
    build_pipeline_decision,
    format_pipeline_decision_prompt_section,
)
from engine.search_node import Journal, SearchNode
from utils.serialize import dumps_json, loads_json


def _agent(
    task_desc: str = "image classification with train_images and labels",
    *,
    enabled: bool = True,
    precision_mode: str = "normal",
    target_profile: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        task_desc=task_desc,
        acfg=SimpleNamespace(
            code=SimpleNamespace(temp=0.0),
            pipeline_decision_enabled=enabled,
            precision_optimization_mode=precision_mode,
        ),
        cfg=SimpleNamespace(preflight=SimpleNamespace(target_profile=target_profile)),
    )


def _decision_payload(**evidence_overrides):
    evidence = {
        "hardware_context_used": True,
        "evidence_refs": ["graph:job:1"],
        "confidence": 0.9,
        "missing_evidence": [],
    }
    evidence.update(evidence_overrides)
    return {
        "model_design": {
            "modality": "image",
            "target_type": "classification",
            "shape_constraints": ["images are 2D tensors"],
            "family": "compact_cnn",
            "alternatives_considered": ["vision_transformer"],
            "loss": "cross entropy",
            "output_interface": "class logits for the submission labels",
            "reason": "A CNN matches image labels and is safe for the environment.",
            "hardware_fit": "tensor-core-friendly if AMP is supported",
        },
        "datatype_precision": {
            "precision_policy": "bf16_amp",
            "precision_model_adaptation": "none",
            "fallback_policy": "fall back to fp32 on unsupported AMP",
            "reason": "BF16 is safe when hardware evidence confirms support.",
        },
        "training_evaluation": {
            "optimizer": "AdamW",
            "scheduler": "none",
            "batch_size_policy": "scheduler_recommended",
            "dataloader_policy": "pin_memory and non_blocking on CUDA",
            "fallbacks": ["halve batch size on OOM"],
            "metrics_to_log": ["elapsed_seconds", "peak_vram_mb", "resolved_batch_size"],
            "advanced_optimizer_used": True,
            "reason": "Stable default for image classification.",
        },
        "evidence": evidence,
    }


def test_pipeline_prompt_contract_preserves_required_order() -> None:
    section = format_pipeline_decision_prompt_section(_decision_payload())

    assert PIPELINE_STAGE_ORDER == ("model_design", "datatype_precision", "training_evaluation")
    order_tokens = ["1. Model design", "2. Datatype/quantization", "3. Training"]
    positions = [section.index(token) for token in order_tokens]
    assert positions == sorted(positions)
    assert "model_design -> datatype_precision -> training_evaluation" in section


def test_missing_graph_or_predictor_evidence_uses_safe_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        "agents.prompts.pipeline_decision.generate",
        lambda **_: json.dumps(_decision_payload(evidence_refs=["hallucinated:ref"], confidence=0.99)),
    )
    empty_context = SimpleNamespace(
        compact_context={
            "hardware_context": {},
            "graph_evidence": {"exact_profiles": [], "similar_profiles": [], "packed_profiles": []},
            "recommendations": [],
            "risk_flags": [],
            "evidence_refs": [],
            "confidence": 0.0,
        }
    )

    decision = build_pipeline_decision(
        _agent(),
        stage="draft",
        data_preview="train_images/*.png with label column",
        hardware_contexts=[empty_context],
    )

    assert decision["evidence"]["hardware_context_used"] is False
    assert decision["evidence"]["evidence_refs"] == []
    assert "predictor/graph evidence not available" in decision["evidence"]["missing_evidence"]
    assert "hardware/profile evidence not available" in decision["evidence"]["missing_evidence"]
    assert decision["model_design"]["hardware_fit"] == "none"
    assert decision["training_evaluation"]["advanced_optimizer_used"] is False
    assert decision["training_evaluation"]["batch_size_policy"] == "fixed"
    assert decision["datatype_precision"]["precision_policy"] == "disabled"
    assert decision["datatype_precision"]["precision_model_adaptation"] == "none"


def test_pipeline_decision_persists_on_search_node_round_trip(monkeypatch) -> None:
    monkeypatch.setattr(
        "agents.prompts.pipeline_decision.generate",
        lambda **_: json.dumps(_decision_payload(evidence_refs=["made_up:ref"], confidence=0.1)),
    )
    graph_context = SimpleNamespace(
        compact_context={
            "hardware_context": {"found": True, "summary": "RTX Test"},
            "graph_evidence": {
                "exact_profiles": [{"resolved_batch_size": 16, "ref": "graph:job:1"}],
                "similar_profiles": [],
                "packed_profiles": [],
            },
            "evidence_refs": ["graph:job:1"],
            "confidence": 0.8,
        }
    )

    decision = build_pipeline_decision(
        _agent(),
        stage="draft",
        data_preview="train_images/*.png with label column",
        hardware_contexts=[graph_context],
    )
    root = SearchNode(code="", plan="root", stage="root")
    node = SearchNode(code="BATCH_SIZE = 16", plan="draft", parent=root, stage="draft")
    apply_pipeline_decision_to_node(node, decision)

    journal = Journal()
    journal.append(root)
    journal.append(node)
    loaded = loads_json(dumps_json(journal), Journal)
    loaded_node = loaded.nodes[1]

    section = format_pipeline_decision_prompt_section(decision)
    assert "model_design -> datatype_precision -> training_evaluation" in section
    assert loaded_node.pipeline_decision["evidence"]["evidence_refs"] == ["graph:job:1"]
    assert loaded_node.pipeline_decision["evidence"]["confidence"] == 0.8


def test_baseline_style_decision_section_has_no_hardware_evidence(monkeypatch) -> None:
    monkeypatch.setattr(
        "agents.prompts.pipeline_decision.generate",
        lambda **_: json.dumps(_decision_payload(evidence_refs=["hallucinated:ref"])),
    )

    decision = build_pipeline_decision(
        _agent("tabular regression from train.csv"),
        stage="draft",
        data_preview="train.csv with numeric feature columns and target value",
        hardware_contexts=[],
    )
    section = format_pipeline_decision_prompt_section(decision)

    assert "# Pipeline Decision Contract" in section
    assert "# Hardware/Profile Optimization Context" not in section
    assert decision["evidence"]["hardware_context_used"] is False
    assert decision["evidence"]["evidence_refs"] == []


def test_invalid_policy_values_and_string_boolean_are_normalized(monkeypatch) -> None:
    payload = _decision_payload(confidence=2.5)
    payload["training_evaluation"]["advanced_optimizer_used"] = "false"
    payload["training_evaluation"]["batch_size_policy"] = "invented_policy"
    payload["datatype_precision"]["precision_policy"] = "fp8_magic"
    monkeypatch.setattr(
        "agents.prompts.pipeline_decision.generate",
        lambda **_: json.dumps(payload),
    )
    graph_context = SimpleNamespace(
        compact_context={
            "hardware_context": {"found": True, "summary": "RTX Test"},
            "graph_evidence": {"exact_profiles": [{"resolved_batch_size": 16}]},
            "evidence_refs": ["graph:job:1"],
            "confidence": 2.5,
        }
    )

    decision = build_pipeline_decision(
        _agent(),
        stage="draft",
        data_preview="train_images/*.png with label column",
        hardware_contexts=[graph_context],
    )

    assert decision["training_evaluation"]["advanced_optimizer_used"] is False
    assert decision["training_evaluation"]["batch_size_policy"] == "scheduler_recommended"
    assert decision["datatype_precision"]["precision_policy"] == "disabled"
    assert decision["datatype_precision"]["precision_model_adaptation"] == "none"
    assert decision["evidence"]["confidence"] == 1.0


def test_te_precision_policy_and_adapter_are_preserved_with_evidence(monkeypatch) -> None:
    payload = _decision_payload()
    payload["datatype_precision"]["precision_policy"] = "nvfp4_te"
    payload["datatype_precision"]["precision_model_adaptation"] = "replace compatible Linear layers with TE modules"
    monkeypatch.setattr(
        "agents.prompts.pipeline_decision.generate",
        lambda **_: json.dumps(payload),
    )
    graph_context = SimpleNamespace(
        compact_context={
            "hardware_context": {
                "found": True,
                "summary": "Blackwell GPU with Transformer Engine support",
                "hardware": {
                    "architecture": "blackwell",
                    "compute_capability": "10.0",
                },
            },
            "graph_evidence": {"exact_profiles": [{"precision": "nvfp4_te", "ref": "graph:te:nvfp4"}]},
            "evidence_refs": ["graph:te:nvfp4"],
            "confidence": 0.8,
        }
    )

    decision = build_pipeline_decision(
        _agent("transformer training", precision_mode="aggressive"),
        stage="draft",
        data_preview="tokenized sequences",
        hardware_contexts=[graph_context],
    )

    assert decision["datatype_precision"]["precision_policy"] == "nvfp4_te"
    assert "TE modules" in decision["datatype_precision"]["precision_model_adaptation"]


def test_explicit_preflight_profile_supplies_precision_allowlist_without_graph(monkeypatch) -> None:
    monkeypatch.setattr(
        "agents.prompts.pipeline_decision.generate",
        lambda **_: json.dumps(_decision_payload()),
    )
    profile = str(Path(__file__).parents[1] / "config" / "preflight_profiles" / "a100_80gb.yaml")

    decision = build_pipeline_decision(
        _agent(precision_mode="normal", target_profile=profile),
        stage="draft",
        data_preview="train_images/*.png with label column",
        hardware_contexts=[],
    )

    assert decision["datatype_precision"]["precision_policy"] == "bf16_amp"
    assert decision["evidence"]["hardware_context_used"] is False


@pytest.mark.parametrize("selected", ["fp8_te", "mxfp8_te", "nvfp4_te", "fp4", "fp6"])
def test_normal_mode_normalizes_low_or_unverified_precision_to_disabled(
    monkeypatch, selected: str
) -> None:
    payload = _decision_payload()
    payload["datatype_precision"]["precision_policy"] = selected
    monkeypatch.setattr(
        "agents.prompts.pipeline_decision.generate",
        lambda **_: json.dumps(payload),
    )
    context = SimpleNamespace(
        compact_context={
            "hardware_context": {
                "found": True,
                "hardware": {"architecture": "blackwell", "compute_capability": "10.0"},
            },
            "graph_evidence": {"exact_profiles": [{"ref": "graph:blackwell"}]},
            "evidence_refs": ["graph:blackwell"],
            "confidence": 0.9,
        }
    )

    decision = build_pipeline_decision(
        _agent("transformer training", precision_mode="normal"),
        stage="draft",
        data_preview="tokenized sequences",
        hardware_contexts=[context],
    )

    assert decision["datatype_precision"]["precision_policy"] == "disabled"
    assert decision["datatype_precision"]["precision_model_adaptation"] == "none"
    assert "FP32" in decision["datatype_precision"]["fallback_policy"]


@pytest.mark.parametrize(
    ("architecture", "selected", "expected"),
    [
        ("ada_lovelace", "fp8_te", "fp8_te"),
        ("ada_lovelace", "mxfp8_te", "disabled"),
        ("hopper", "fp8_te", "fp8_te"),
        ("hopper", "nvfp4_te", "disabled"),
        ("blackwell", "fp8_te", "fp8_te"),
        ("blackwell", "mxfp8_te", "mxfp8_te"),
        ("blackwell", "nvfp4_te", "nvfp4_te"),
    ],
)
def test_aggressive_pipeline_normalization_obeys_architecture(
    monkeypatch, architecture: str, selected: str, expected: str
) -> None:
    payload = _decision_payload()
    payload["datatype_precision"]["precision_policy"] = selected
    monkeypatch.setattr(
        "agents.prompts.pipeline_decision.generate",
        lambda **_: json.dumps(payload),
    )
    context = SimpleNamespace(
        compact_context={
            "hardware_context": {
                "found": True,
                "hardware": {"architecture": architecture, "compute_capability": "10.0"},
            },
            "graph_evidence": {"exact_profiles": [{"ref": f"graph:{architecture}"}]},
            "evidence_refs": [f"graph:{architecture}"],
            "confidence": 0.9,
        }
    )

    decision = build_pipeline_decision(
        _agent("transformer training", precision_mode="aggressive"),
        stage="draft",
        data_preview="tokenized sequences",
        hardware_contexts=[context],
    )

    assert decision["datatype_precision"]["precision_policy"] == expected


def test_long_pipeline_trace_remains_valid_json() -> None:
    decision = _decision_payload()
    decision["model_design"]["shape_constraints"] = ["shape-" + ("x" * 500)] * 20
    decision["model_design"]["reason"] = "reason-" + ("y" * 4000)

    section = format_pipeline_decision_prompt_section(decision, max_chars=1800)
    match = re.search(r"```json\n(.*)\n```", section, re.DOTALL)

    assert match is not None
    parsed = json.loads(match.group(1))
    assert parsed["model_design"]["modality"] == "image"
    assert "datatype" not in parsed
    assert "model" not in parsed
    assert "optimizer" not in parsed
    assert "tuning" not in parsed
    assert len(section) <= 1800
    assert section.endswith("```\n")


def test_pipeline_decision_can_be_disabled_without_calling_llm(monkeypatch) -> None:
    monkeypatch.setattr(
        "agents.prompts.pipeline_decision.generate",
        lambda **_: (_ for _ in ()).throw(AssertionError("LLM should not be called")),
    )

    decision = build_pipeline_decision(
        _agent(enabled=False),
        stage="draft",
        data_preview="train.csv",
        hardware_contexts=[],
    )

    assert decision == {}
    assert format_pipeline_decision_prompt_section(decision) == ""


def test_pipeline_decision_llm_failure_returns_explicit_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        "agents.prompts.pipeline_decision.generate",
        lambda **_: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )

    decision = build_pipeline_decision(
        _agent("tabular regression from train.csv"),
        stage="draft",
        data_preview="numeric feature columns and target value",
        hardware_contexts=[],
    )

    assert decision["model_design"]["modality"] == "tabular"
    assert decision["training_evaluation"]["batch_size_policy"] == "fixed"
    assert "pipeline decision LLM unavailable" in decision["evidence"]["missing_evidence"]


def test_child_node_inherits_pipeline_decision_by_default() -> None:
    root = SearchNode(code="", plan="root", stage="root")
    parent = SearchNode(code="x = 1", plan="draft", parent=root, stage="draft")
    parent.pipeline_decision = _decision_payload()

    child = SearchNode(code="x = 2", plan="debug", parent=parent, stage="debug")

    assert child.pipeline_decision == parent.pipeline_decision
    assert child.pipeline_decision is not parent.pipeline_decision


def test_noncanonical_four_bucket_decision_is_not_translated() -> None:
    noncanonical = {
        "datatype": {
            "modality": "image",
            "target_type": "classification",
            "shape_constraints": ["old image tensors"],
            "reason": "old datatype reason",
        },
        "model": {
            "family": "compact_cnn",
            "alternatives_considered": [],
            "reason": "old model reason",
            "hardware_fit": "none",
        },
        "optimizer": {
            "loss": "cross entropy",
            "optimizer": "AdamW",
            "scheduler": "none",
            "reason": "old optimizer reason",
            "advanced_optimizer_used": False,
        },
        "tuning": {
            "batch_size_policy": "fixed",
            "precision_policy": "disabled",
            "precision_model_adaptation": "none",
            "dataloader_policy": "safe defaults",
            "fallbacks": ["halve batch on OOM"],
            "metrics_to_log": ["elapsed_seconds"],
        },
        "evidence": {
            "hardware_context_used": False,
            "evidence_refs": [],
            "confidence": 0.0,
            "missing_evidence": [],
        },
    }

    section = format_pipeline_decision_prompt_section(noncanonical)
    payload = json.loads(re.search(r"```json\n(.*)\n```", section, re.DOTALL).group(1))

    assert list(payload) == ["model_design", "datatype_precision", "training_evaluation", "evidence"]
    assert payload["model_design"].get("family") != "compact_cnn"
    assert payload["model_design"].get("loss") != "cross entropy"
    assert payload["training_evaluation"].get("optimizer") != "AdamW"

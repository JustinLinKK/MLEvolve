from __future__ import annotations

import json
import re
from types import SimpleNamespace

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
) -> SimpleNamespace:
    return SimpleNamespace(
        task_desc=task_desc,
        acfg=SimpleNamespace(
            code=SimpleNamespace(temp=0.0),
            pipeline_decision_enabled=enabled,
        ),
        cfg=SimpleNamespace(),
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
        "datatype": {
            "modality": "image",
            "target_type": "classification",
            "shape_constraints": ["images are 2D tensors"],
            "reason": "The task uses labeled images.",
        },
        "model": {
            "family": "compact_cnn",
            "alternatives_considered": ["vision_transformer"],
            "reason": "A CNN matches image labels and is safe for the environment.",
            "hardware_fit": "tensor-core-friendly if AMP is supported",
        },
        "optimizer": {
            "loss": "cross entropy",
            "optimizer": "AdamW",
            "scheduler": "none",
            "reason": "Stable default for image classification.",
            "advanced_optimizer_used": True,
        },
        "tuning": {
            "batch_size_policy": "scheduler_recommended",
            "precision_policy": "bf16_amp",
            "precision_model_adaptation": "none",
            "dataloader_policy": "pin_memory and non_blocking on CUDA",
            "fallbacks": ["halve batch size on OOM"],
            "metrics_to_log": ["elapsed_seconds", "peak_vram_mb", "resolved_batch_size"],
        },
        "evidence": evidence,
    }


def test_pipeline_prompt_contract_preserves_required_order() -> None:
    section = format_pipeline_decision_prompt_section(_decision_payload())

    assert PIPELINE_STAGE_ORDER == ("model_design", "datatype_precision", "training_evaluation")
    order_tokens = ["1. Model design", "2. Datatype precision", "3. Training evaluation"]
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
    assert decision["model"]["hardware_fit"] == "none"
    assert decision["optimizer"]["advanced_optimizer_used"] is False
    assert decision["tuning"]["batch_size_policy"] == "fixed"
    assert decision["tuning"]["precision_policy"] == "disabled"
    assert decision["tuning"]["precision_model_adaptation"] == "none"


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
    payload["optimizer"]["advanced_optimizer_used"] = "false"
    payload["tuning"]["batch_size_policy"] = "invented_policy"
    payload["tuning"]["precision_policy"] = "fp8_magic"
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

    assert decision["optimizer"]["advanced_optimizer_used"] is False
    assert decision["tuning"]["batch_size_policy"] == "scheduler_recommended"
    assert decision["tuning"]["precision_policy"] == "disabled"
    assert decision["tuning"]["precision_model_adaptation"] == "none"
    assert decision["evidence"]["confidence"] == 1.0


def test_te_precision_policy_and_adapter_are_preserved_with_evidence(monkeypatch) -> None:
    payload = _decision_payload()
    payload["tuning"]["precision_policy"] = "nvfp4_te"
    payload["tuning"]["precision_model_adaptation"] = "replace compatible Linear layers with TE modules"
    monkeypatch.setattr(
        "agents.prompts.pipeline_decision.generate",
        lambda **_: json.dumps(payload),
    )
    graph_context = SimpleNamespace(
        compact_context={
            "hardware_context": {"found": True, "summary": "Blackwell GPU with Transformer Engine support"},
            "graph_evidence": {"exact_profiles": [{"precision": "nvfp4_te", "ref": "graph:te:nvfp4"}]},
            "evidence_refs": ["graph:te:nvfp4"],
            "confidence": 0.8,
        }
    )

    decision = build_pipeline_decision(
        _agent("transformer training"),
        stage="draft",
        data_preview="tokenized sequences",
        hardware_contexts=[graph_context],
    )

    assert decision["tuning"]["precision_policy"] == "nvfp4_te"
    assert "TE modules" in decision["tuning"]["precision_model_adaptation"]


def test_long_pipeline_trace_remains_valid_json() -> None:
    decision = _decision_payload()
    decision["datatype"]["shape_constraints"] = ["shape-" + ("x" * 500)] * 20
    decision["model"]["reason"] = "reason-" + ("y" * 4000)

    section = format_pipeline_decision_prompt_section(decision, max_chars=1800)
    match = re.search(r"```json\n(.*)\n```", section, re.DOTALL)

    assert match is not None
    parsed = json.loads(match.group(1))
    assert parsed["datatype"]["modality"] == "image"
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

    assert decision["datatype"]["modality"] == "tabular"
    assert decision["tuning"]["batch_size_policy"] == "fixed"
    assert "pipeline decision LLM unavailable" in decision["evidence"]["missing_evidence"]


def test_child_node_inherits_pipeline_decision_by_default() -> None:
    root = SearchNode(code="", plan="root", stage="root")
    parent = SearchNode(code="x = 1", plan="draft", parent=root, stage="draft")
    parent.pipeline_decision = _decision_payload()

    child = SearchNode(code="x = 2", plan="debug", parent=parent, stage="debug")

    assert child.pipeline_decision == parent.pipeline_decision
    assert child.pipeline_decision is not parent.pipeline_decision

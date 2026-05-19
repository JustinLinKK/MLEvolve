from __future__ import annotations

from types import SimpleNamespace

from agents.hardware_context import (
    EVIDENCE_NOT_LAW_RULE,
    apply_hardware_context_to_node,
    compact_optimization_context,
    format_hardware_prompt_section,
    get_hardware_context_for_stage,
)
from engine.script_introspection import introspect_training_script, normalized_mlevolve_script_signature
from engine.search_node import Journal, SearchNode
from utils.serialize import dumps_json, loads_json


def test_script_introspection_extracts_training_hints() -> None:
    code = """
import torch
import timm
MODEL_NAME = "vit_base_patch16_224"
BATCH_SIZE = 32
EPOCHS = 4
model = timm.create_model(MODEL_NAME)
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    pass
"""

    candidate = introspect_training_script(code)

    assert candidate["proposed_batch_size"] == 32
    assert candidate["proposed_epochs"] == 4
    assert candidate["model_key"] == "vit_base_patch16_224"
    assert candidate["model_family"] == "transformer"
    assert candidate["framework"] == "pytorch"
    assert candidate["uses_amp"] is True
    assert candidate["requires_gpu"] is True

    changed_batch = code.replace("BATCH_SIZE = 32", "BATCH_SIZE = 8")
    assert normalized_mlevolve_script_signature(code) == normalized_mlevolve_script_signature(changed_batch)


def test_compact_context_formats_prompt_without_raw_bloat() -> None:
    raw = {
        "hardware_context": {
            "found": True,
            "hardware": {
                "hardware_key": "hw-1",
                "gpu_name": "RTX Test",
                "total_vram_mb": 49152,
                "summary_text": "RTX Test with 49152 MiB VRAM",
            },
            "scheduler_limits": {"safe_vram_budget_mb": 30720, "max_packed_jobs_per_gpu": 2},
        },
        "graph_evidence": {
            "exact_profiles": [
                {
                    "summary_text": "resnet50 succeeded",
                    "resolved_batch_size": 16,
                    "peak_vram_mb": 12000,
                    "avg_sm_utilization_pct": 62,
                    "ref": "graph:job:1",
                }
            ],
            "similar_profiles": [],
            "packed_profiles": [],
        },
        "derived_diagnosis": {
            "profile_symptoms": ["precision_not_optimized"],
            "optimization_targets": ["enable_tensor_core"],
        },
        "vector_evidence": {
            "recipes": [
                {
                    "record_id": "recipe-1",
                    "record_type": "optimization_recipe_chunks",
                    "title": "Use BF16 autocast",
                    "recommended_patterns": ["Use torch.amp.autocast with bf16."],
                    "confidence": 0.7,
                }
            ],
            "docs": [],
            "api_symbols": [],
        },
        "recommendations": ["Use graph-recommended physical batch size 16 as the starting point."],
        "risk_flags": ["avoid fixed oversized batch size"],
        "evidence_refs": ["graph:job:1", "code_knowledge:recipe:1"],
        "confidence": 0.8,
    }

    compact = compact_optimization_context(raw)
    prompt = format_hardware_prompt_section(compact, max_chars=2000)

    assert "# Hardware/Profile Optimization Context" in prompt
    assert "RTX Test" in prompt
    assert "Use graph-recommended physical batch size 16" in prompt
    assert "precision_not_optimized" in prompt
    assert "Use BF16 autocast" in prompt
    assert EVIDENCE_NOT_LAW_RULE in prompt
    assert "raw_context" not in prompt


def test_scheduler_lookup_is_non_fatal_and_uses_get_optimization_context() -> None:
    class FakeScheduler:
        def __init__(self) -> None:
            self.calls = []
            self.settings = SimpleNamespace(
                gpu_scheduler=SimpleNamespace(
                    submission_defaults=SimpleNamespace(
                        requires_gpu=True,
                        packing_family="mlevolve_script",
                        backend_allowlist=["cuda_process"],
                        batch_probe_model_key=None,
                    )
                )
            )

        def get_optimization_context(self, *, candidate, limit):
            self.calls.append((candidate, limit))
            return {"hardware_context": {"found": False}, "confidence": 0.0}

    agent = SimpleNamespace(
        scheduler_client=FakeScheduler(),
        acfg=SimpleNamespace(
            hardware_context_enabled=True,
            hardware_context_limit=3,
            hardware_context_max_prompt_chars=1000,
        ),
        cfg=SimpleNamespace(exp_id="task-a"),
        task_desc="image classification",
        data_preview="train.csv and train_images",
    )

    context = get_hardware_context_for_stage(agent, "draft")

    assert agent.scheduler_client.calls
    candidate, limit = agent.scheduler_client.calls[0]
    assert candidate["stage"] == "draft"
    assert candidate["task_type"] == "vision_training"
    assert limit == 3
    assert context.prompt_section

    agent.scheduler_client.get_optimization_context = lambda **_: (_ for _ in ()).throw(RuntimeError("down"))
    failed = get_hardware_context_for_stage(agent, "draft")
    assert failed.prompt_section == ""


def test_search_node_hardware_fields_round_trip_in_journal() -> None:
    root = SearchNode(code="", plan="root", stage="root")
    node = SearchNode(code="BATCH_SIZE = 16", plan="draft", parent=root, stage="draft")
    apply_hardware_context_to_node(
        node,
        SimpleNamespace(
            candidate={"proposed_batch_size": 16},
            compact_context={
                "hardware_context": {"found": True, "hardware": {"gpu_name": "RTX Test"}},
                "graph_evidence": {"exact_profiles": [{"resolved_batch_size": 16}]},
                "derived_diagnosis": {"profile_symptoms": ["low_sm_utilization"], "optimization_targets": []},
                "vector_evidence": {"recipes": []},
                "risk_flags": ["avoid_oom"],
                "evidence_refs": ["graph:job:1"],
                "confidence": 0.6,
            },
        ),
    )
    journal = Journal()
    journal.append(root)
    journal.append(node)

    loaded = loads_json(dumps_json(journal), Journal)
    loaded_node = loaded.nodes[1]

    assert loaded_node.hardware_context["found"] is True
    assert loaded_node.resolved_batch_size == 16
    assert loaded_node.scheduler_risk_flags == ["avoid_oom"]
    assert loaded_node.scheduler_confidence == 0.6
    assert loaded_node.hardware_evidence_refs == ["graph:job:1"]
    assert loaded_node.parent.id == root.id

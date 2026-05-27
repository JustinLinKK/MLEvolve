from __future__ import annotations

from types import SimpleNamespace

from agents.hardware_context import (
    CONSTRAINT_PRECEDENCE_RULE,
    EVIDENCE_NOT_LAW_RULE,
    HARDWARE_BUDGET_GUARDRAIL_RULE,
    _scheduler_backend_config,
    apply_hardware_context_to_node,
    build_hardware_candidate,
    compact_optimization_context,
    format_hardware_design_brief,
    format_hardware_prompt_section,
    get_hardware_design_brief,
    get_hardware_context_for_stage,
    hardware_context_instructions,
    optimize_training_parameters_for_round,
)
from agents.coder.stepwise_coder import _hardware_reasoning_enabled, create_default_step_agents
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
	IMG_SIZE = 224
	N_FOLDS = 5
	ENSEMBLE_SIZE = 2
	TTA_STEPS = 3
	model = timm.create_model(MODEL_NAME)
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    pass
"""

    candidate = introspect_training_script(code)

    assert candidate["proposed_batch_size"] == 32
    assert candidate["proposed_epochs"] == 4
    assert candidate["input_resolution"] == 224
    assert candidate["fold_count"] == 5
    assert candidate["ensemble_count"] == 2
    assert candidate["tta_count"] == 3
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
            "backend_capabilities": {
                "mode": "auto",
                "effective_mode": "parallel_auto_pack",
                "backend_priority": ["stream_mps", "stream", "cuda_process", "exclusive"],
                "enabled_backends": ["exclusive", "stream_mps", "stream", "cuda_process"],
                "concurrent_backend_allowlist": ["stream_mps", "stream"],
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
    assert "Scheduler backend config" in prompt
    assert "stream_mps" in prompt
    assert "precision_not_optimized" in prompt
    assert "Use BF16 autocast" in prompt
    assert EVIDENCE_NOT_LAW_RULE in prompt
    assert CONSTRAINT_PRECEDENCE_RULE in prompt
    assert HARDWARE_BUDGET_GUARDRAIL_RULE in prompt
    assert "raw_context" not in prompt


def test_hardware_design_brief_ranks_model_options_without_overriding_constraints() -> None:
    compact = {
        "hardware_context": {
            "summary": "RTX 5090 with tensor cores",
            "scheduler_limits": {"safe_vram_budget_mb": 24000, "max_packed_jobs_per_gpu": 2},
        },
        "workload_type": "vision_training",
        "model_options": [
            {
                "model_family": "vision_transformer",
                "rationale": "transformer evidence uses bf16 tensor cores",
                "hardware_features": ["bf16", "tensor core"],
                "confidence": 0.8,
            }
        ],
        "recommendations": ["Prefer tensor-core-friendly shapes when task fit is acceptable."],
        "evidence_refs": ["code_knowledge:doc:tensor-core"],
        "confidence": 0.8,
    }

    prompt = format_hardware_design_brief(compact, max_chars=2000)

    assert "Hardware-Aware Model Design Brief" in prompt
    assert "vision_transformer" in prompt
    assert "tensor-core-friendly" in prompt
    assert CONSTRAINT_PRECEDENCE_RULE in prompt
    assert HARDWARE_BUDGET_GUARDRAIL_RULE in prompt


def test_scheduler_lookup_is_non_fatal_and_uses_get_optimization_context() -> None:
    class FakeScheduler:
        def __init__(self) -> None:
            self.calls = []
            self.settings = SimpleNamespace(
                gpu_scheduler=SimpleNamespace(
                    mode="auto",
                    backend_priority=["stream_mps", "stream", "cuda_process", "exclusive"],
                    concurrent_backend_allowlist=["stream_mps", "stream"],
                    submission_defaults=SimpleNamespace(
                        requires_gpu=True,
                        packing_family="mlevolve_script",
                        backend_allowlist=[],
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
    assert candidate["scheduler_mode"] == "auto"
    assert candidate["scheduler_effective_mode"] == "parallel_auto_pack"
    assert candidate["backend_preference"] == "stream_mps"
    assert limit == 3
    assert context.prompt_section

    agent.scheduler_client.get_optimization_context = lambda **_: (_ for _ in ()).throw(RuntimeError("down"))
    failed = get_hardware_context_for_stage(agent, "draft")
    assert failed.prompt_section == ""


def test_hardware_candidate_prefers_boot_resolved_auto_backend_probe() -> None:
    class FakeScheduler:
        def __init__(self) -> None:
            self.settings = SimpleNamespace(
                gpu_scheduler=SimpleNamespace(
                    mode="auto",
                    backend_priority=["stream_mps", "stream", "cuda_process", "exclusive"],
                    concurrent_backend_allowlist=["stream_mps", "stream"],
                    submission_defaults=SimpleNamespace(
                        requires_gpu=True,
                        packing_family="mlevolve_script",
                        backend_allowlist=[],
                        batch_probe_model_key=None,
                    ),
                )
            )

        def list_events(self, *, event_type=None, job_id=None):
            assert event_type == "scheduler_auto_backend_probe"
            return [
                {
                    "payload": {
                        "configured_mode": "auto",
                        "effective_scheduler_mode": "parallel_auto_pack",
                        "backend_priority": ["cuda_process", "exclusive"],
                        "concurrent_backend_allowlist": [],
                    }
                }
            ]

    agent = SimpleNamespace(
        scheduler_client=FakeScheduler(),
        acfg=SimpleNamespace(hardware_context_enabled=True),
        cfg=SimpleNamespace(exp_id="task-a"),
        task_desc="image classification",
        data_preview="train_images",
    )

    candidate = build_hardware_candidate(agent, "draft")

    assert candidate["scheduler_mode"] == "auto"
    assert candidate["scheduler_effective_mode"] == "parallel_auto_pack"
    assert candidate["backend_preference"] == "cuda_process"


def test_scheduler_backend_config_preserves_empty_auto_probe_lists() -> None:
    class FakeScheduler:
        def __init__(self) -> None:
            self.settings = SimpleNamespace(
                gpu_scheduler=SimpleNamespace(
                    mode="auto",
                    backend_priority=["stream", "cuda_process", "exclusive"],
                    concurrent_backend_allowlist=["stream"],
                )
            )

        def list_events(self, *, event_type=None, job_id=None):
            del job_id
            assert event_type == "scheduler_auto_backend_probe"
            return [
                {
                    "payload": {
                        "configured_mode": "auto",
                        "effective_scheduler_mode": "parallel_auto_pack",
                        "backend_priority": [],
                        "concurrent_backend_allowlist": [],
                    }
                }
            ]

    backend_config = _scheduler_backend_config(FakeScheduler())

    assert backend_config["backend_priority"] == []
    assert backend_config["concurrent_backend_allowlist"] == []


def test_hardware_design_brief_uses_scheduler_model_design_context() -> None:
    class FakeScheduler:
        def __init__(self) -> None:
            self.calls = []
            self.settings = SimpleNamespace(
                gpu_scheduler=SimpleNamespace(
                    mode="auto",
                    backend_priority=["stream_mps", "stream", "cuda_process", "exclusive"],
                    concurrent_backend_allowlist=["stream_mps", "stream"],
                    submission_defaults=SimpleNamespace(
                        requires_gpu=True,
                        packing_family="mlevolve_script",
                        backend_allowlist=[],
                        batch_probe_model_key=None,
                    )
                )
            )

        def get_model_design_hardware_context(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "hardware_context": {
                    "found": True,
                    "hardware": {"gpu_name": "RTX Test", "summary_text": "RTX Test"},
                },
                "workload_type": kwargs["workload_type"],
                "model_options": [
                    {
                        "model_family": "vision_transformer",
                        "rationale": "uses tensor cores efficiently",
                        "confidence": 0.7,
                    }
                ],
                "recommendations": ["Prefer bf16 autocast."],
                "evidence_refs": ["code_knowledge:doc:bf16"],
                "confidence": 0.7,
            }

    agent = SimpleNamespace(
        scheduler_client=FakeScheduler(),
        acfg=SimpleNamespace(
            hardware_context_enabled=True,
            hardware_context_limit=3,
            hardware_context_max_prompt_chars=1000,
        ),
        cfg=SimpleNamespace(exp_id="task-a"),
        task_desc="histopathologic cancer detection image classification",
        data_preview="train_images",
    )

    context = get_hardware_design_brief(agent)

    assert agent.scheduler_client.calls
    assert agent.scheduler_client.calls[0]["workload_type"] == "vision_training"
    assert "vision_transformer" in context.prompt_section
    assert context.compact_context["model_options"][0]["model_family"] == "vision_transformer"


def test_training_parameter_round_review_rewrites_safe_literals_and_records_decision() -> None:
    class FakeScheduler:
        def __init__(self) -> None:
            self.settings = SimpleNamespace(
                gpu_scheduler=SimpleNamespace(
                    mode="auto",
                    backend_priority=["stream_mps", "stream", "cuda_process", "exclusive"],
                    concurrent_backend_allowlist=["stream_mps", "stream"],
                    submission_defaults=SimpleNamespace(
                        requires_gpu=True,
                        packing_family="mlevolve_script",
                        backend_allowlist=[],
                        batch_probe_model_key=None,
                    )
                )
            )

        def plan_job_packet(self, *, candidates, limit):
            return {
                "packet_id": "packet-test",
                "jobs": [
                    {
                        "optimization_context": {
                            "hardware_context": {"found": True},
                            "graph_evidence": {
                                "exact_profiles": [
                                    {"resolved_batch_size": 24, "ref": "graph:batch"}
                                ]
                            },
                            "recommendations": [],
                            "evidence_refs": ["graph:batch"],
                            "confidence": 0.9,
                        }
                    }
                ],
                "confidence": 0.9,
            }

        def get_optimization_context(self, *, candidate, limit):
            raise AssertionError("packet context should be used")

    root = SearchNode(code="", plan="root", stage="root")
    node = SearchNode(code="BATCH_SIZE = 8\nEPOCHS = 2\n", plan="draft", parent=root, stage="draft")
    agent = SimpleNamespace(
        scheduler_client=FakeScheduler(),
        acfg=SimpleNamespace(
            hardware_context_enabled=True,
            hardware_context_limit=3,
            hardware_context_max_prompt_chars=1000,
        ),
        cfg=SimpleNamespace(exp_id="task-a"),
        task_desc="image classification",
        data_preview="train_images",
    )

    decisions = optimize_training_parameters_for_round(agent, [node])

    assert decisions
    assert "BATCH_SIZE = 24" in node.code
    assert node.hardware_decision["applied_params"] == {"batch_size": 24}
    assert node.hardware_decision["evidence_refs"] == ["graph:batch"]


def test_training_parameter_round_review_does_not_increase_epochs() -> None:
    class FakeScheduler:
        def __init__(self) -> None:
            self.settings = SimpleNamespace(
                gpu_scheduler=SimpleNamespace(
                    mode="auto",
                    backend_priority=["stream_mps", "stream", "cuda_process", "exclusive"],
                    concurrent_backend_allowlist=["stream_mps", "stream"],
                    submission_defaults=SimpleNamespace(
                        requires_gpu=True,
                        packing_family="mlevolve_script",
                        backend_allowlist=[],
                        batch_probe_model_key=None,
                    )
                )
            )

        def plan_job_packet(self, *, candidates, limit):
            return {
                "packet_id": "packet-test",
                "jobs": [
                    {
                        "optimization_context": {
                            "hardware_context": {"found": True},
                            "recommendations": ["Use epoch budget 10 based on historical profile."],
                            "evidence_refs": ["graph:epochs"],
                            "confidence": 0.9,
                        }
                    }
                ],
                "confidence": 0.9,
            }

        def get_optimization_context(self, *, candidate, limit):
            raise AssertionError("packet context should be used")

    root = SearchNode(code="", plan="root", stage="root")
    node = SearchNode(code="BATCH_SIZE = 8\nEPOCHS = 2\n", plan="draft", parent=root, stage="draft")
    agent = SimpleNamespace(
        scheduler_client=FakeScheduler(),
        acfg=SimpleNamespace(
            hardware_context_enabled=True,
            hardware_context_limit=3,
            hardware_context_max_prompt_chars=1000,
        ),
        cfg=SimpleNamespace(exp_id="task-a"),
        task_desc="image classification",
        data_preview="train_images",
    )

    decisions = optimize_training_parameters_for_round(agent, [node])

    assert decisions
    assert "EPOCHS = 2" in node.code
    assert "epochs" not in node.hardware_decision["applied_params"]


def test_hardware_prompt_text_forbids_budget_increases() -> None:
    instructions = hardware_context_instructions()
    text = "\n".join(item for values in instructions.values() for item in values)
    agents = create_default_step_agents(hardware_aware=True)
    all_guidelines = "\n".join(guideline for step_agent in agents for guideline in step_agent.guidelines)

    assert "Do not increase epochs" in text
    assert "current scheduler backend config" in text
    assert "model size" in text
    assert "Do NOT increase epochs" in all_guidelines
    assert "Allowed hardware optimizations" in all_guidelines
    assert "Scheduler-aware training" in all_guidelines


def test_hardware_context_handles_unexecuted_root_parent() -> None:
    class FakeScheduler:
        def __init__(self) -> None:
            self.calls = []
            self.settings = SimpleNamespace(
                gpu_scheduler=SimpleNamespace(
                    mode="auto",
                    backend_priority=["stream_mps", "stream", "cuda_process", "exclusive"],
                    concurrent_backend_allowlist=["stream_mps", "stream"],
                    submission_defaults=SimpleNamespace(
                        requires_gpu=True,
                        packing_family="mlevolve_script",
                        backend_allowlist=[],
                        batch_probe_model_key=None,
                    )
                )
            )

        def get_optimization_context(self, *, candidate, limit):
            self.calls.append((candidate, limit))
            return {"hardware_context": {"found": False}, "confidence": 0.0}

    root = SearchNode(code="", plan="root", stage="root")
    agent = SimpleNamespace(
        scheduler_client=FakeScheduler(),
        acfg=SimpleNamespace(
            hardware_context_enabled=True,
            hardware_context_limit=3,
            hardware_context_max_prompt_chars=1000,
        ),
        cfg=SimpleNamespace(exp_id="task-a"),
        task_desc="histopathologic cancer detection image classification",
        data_preview="train_images",
    )

    context = get_hardware_context_for_stage(
        agent,
        "code_review",
        parent_node=root,
        code="BATCH_SIZE = 16\n",
    )

    assert agent.scheduler_client.calls
    candidate, _ = agent.scheduler_client.calls[0]
    assert candidate["stage"] == "code_review"
    assert candidate["proposed_batch_size"] == 16
    assert context.candidate["task_type"] == "vision_training"


def test_baseline_mode_disables_hardware_context_and_static_guidance() -> None:
    class FakeScheduler:
        def get_optimization_context(self, *, candidate, limit):
            raise AssertionError("baseline mode should not query scheduler hardware context")

    agent = SimpleNamespace(
        scheduler_client=FakeScheduler(),
        acfg=SimpleNamespace(
            hardware_context_enabled=True,
            hardware_context_limit=3,
            hardware_context_max_prompt_chars=1000,
        ),
        cfg=SimpleNamespace(experiment=SimpleNamespace(mode="baseline"), exp_id="task-a"),
        task_desc="image classification",
        data_preview="train.csv and train_images",
    )

    context = get_hardware_context_for_stage(agent, "draft")
    agents = create_default_step_agents(hardware_aware=False)
    all_guidelines = "\n".join(guideline for step_agent in agents for guideline in step_agent.guidelines)

    assert context.prompt_section == ""
    assert "Hardware-aware" not in all_guidelines


def test_origin_mode_disables_hardware_context_and_static_guidance() -> None:
    class FakeScheduler:
        def get_optimization_context(self, *, candidate, limit):
            raise AssertionError("origin mode should not query scheduler hardware context")

    agent = SimpleNamespace(
        scheduler_client=FakeScheduler(),
        acfg=SimpleNamespace(
            hardware_context_enabled=True,
            hardware_context_limit=3,
            hardware_context_max_prompt_chars=1000,
        ),
        cfg=SimpleNamespace(experiment=SimpleNamespace(mode="origin"), exp_id="task-a"),
        task_desc="image classification",
        data_preview="train.csv and train_images",
    )

    context = get_hardware_context_for_stage(agent, "draft")
    agents = create_default_step_agents(hardware_aware=_hardware_reasoning_enabled(agent))
    all_guidelines = "\n".join(guideline for step_agent in agents for guideline in step_agent.guidelines)

    assert context.prompt_section == ""
    assert "Hardware-aware" not in all_guidelines


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

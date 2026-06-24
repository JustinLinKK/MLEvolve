from __future__ import annotations

from types import SimpleNamespace

from agents.hardware_context import (
    CONSTRAINT_PRECEDENCE_RULE,
    EVIDENCE_NOT_LAW_RULE,
    HARDWARE_BUDGET_GUARDRAIL_RULE,
    _scheduler_backend_config,
    apply_hardware_context_to_node,
    apply_stepwise_hardware_decisions_to_node,
    build_stepwise_hardware_stage_sections,
    build_hardware_candidate,
    compact_optimization_context,
    format_hardware_datatype_prompt_section,
    format_hardware_design_brief,
    format_hardware_prompt_section,
    format_hardware_training_prompt_section,
    get_hardware_design_brief,
    get_hardware_context_for_stage,
    hardware_context_instructions,
    optimize_training_parameters_for_round,
)
from agents.coder.stepwise_coder import _hardware_reasoning_enabled, create_default_step_agents, stepwise_plan_and_code_query
from agents.planner.base_planner import PLANNING_ALLOWED_MODULES
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
	PRECISION_MODE = "bf16"
	LR = 1e-3
	WEIGHT_DECAY = 0.01
	GRADIENT_ACCUMULATION_STEPS = 2
	NUM_WORKERS = 4
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
    assert candidate["precision_mode"] == "bf16"
    assert candidate["learning_rate"] == 1e-3
    assert candidate["weight_decay"] == 0.01
    assert candidate["gradient_accumulation_steps"] == 2
    assert candidate["num_workers"] == 4

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


def test_stage_specific_hardware_prompts_split_precision_and_training_focus() -> None:
    raw = {
        "hardware_context": {
            "found": True,
            "hardware": {
                "hardware_key": "hw-1",
                "gpu_name": "RTX Test",
                "summary_text": "RTX Test with tensor cores",
            },
            "backend_capabilities": {"effective_mode": "parallel_auto_pack"},
            "scheduler_limits": {"safe_vram_budget_mb": 24000},
        },
        "graph_evidence": {
            "exact_profiles": [
                {
                    "summary_text": "bf16 run succeeded",
                    "resolved_batch_size": 16,
                    "precision": "bf16",
                    "runtime_seconds": 12.5,
                    "ref": "graph:bf16",
                }
            ],
            "similar_profiles": [],
            "packed_profiles": [],
        },
        "derived_diagnosis": {
            "profile_symptoms": ["precision_not_optimized", "batch_too_small"],
            "optimization_targets": ["enable_tensor_core", "increase_batch_size"],
        },
        "vector_evidence": {
            "recipes": [
                {
                    "record_id": "amp-recipe",
                    "record_type": "optimization_recipe_chunks",
                    "title": "Use BF16 autocast",
                    "recommended_patterns": ["Use torch.amp.autocast with bf16."],
                    "confidence": 0.8,
                },
                {
                    "record_id": "batch-recipe",
                    "record_type": "optimization_recipe_chunks",
                    "title": "Tune batch size",
                    "recommended_patterns": ["Use graph-recommended physical batch size 16."],
                    "confidence": 0.8,
                },
            ],
            "docs": [],
            "api_symbols": [],
        },
        "recommendations": [
            "Use torch.amp.autocast with bf16.",
            "Use graph-recommended physical batch size 16 as the starting point.",
        ],
        "risk_flags": ["avoid fixed oversized batch size", "avoid fp16 without GradScaler"],
        "evidence_refs": ["graph:bf16", "code_knowledge:amp-recipe"],
        "confidence": 0.8,
    }
    compact = compact_optimization_context(raw)

    dtype_prompt = format_hardware_datatype_prompt_section(compact, max_chars=3000)
    training_prompt = format_hardware_training_prompt_section(compact, max_chars=3000)
    sections = build_stepwise_hardware_stage_sections(
        design_context=SimpleNamespace(prompt_section="# Hardware-Aware Model Design Brief\n", compact_context={}),
        execution_context=SimpleNamespace(compact_context=compact),
        max_chars=3000,
    )

    assert "Datatype/Precision" in dtype_prompt
    assert "torch.amp.autocast with bf16" in dtype_prompt
    assert "Use graph-recommended physical batch size 16 as the starting point" not in dtype_prompt
    assert "learning rate" in dtype_prompt
    assert "Training Hyperparameter" in training_prompt
    assert "physical batch size 16" in training_prompt
    assert "consume the datatype_precision" in training_prompt
    assert "Model Design Brief" in sections["model_design"]
    assert "Datatype/Precision" in sections["datatype_precision"]
    assert "Training Hyperparameter" in sections["training_evaluation"]


def test_hardware_aware_step_agents_split_stage_order_and_boundaries() -> None:
    hardware_agents = create_default_step_agents(hardware_aware=True)
    baseline_agents = create_default_step_agents(hardware_aware=False)

    assert [agent.name for agent in hardware_agents] == [
        "data_processing_and_feature_engineering",
        "model_design",
        "datatype_precision",
        "training_evaluation",
    ]
    assert [agent.name for agent in baseline_agents] == [
        "data_processing_and_feature_engineering",
        "model_design",
        "training_evaluation",
    ]
    assert "datatype_precision" in PLANNING_ALLOWED_MODULES
    model_guidelines = "\n".join(hardware_agents[1].guidelines)
    dtype_guidelines = "\n".join(hardware_agents[2].guidelines)
    training_guidelines = "\n".join(hardware_agents[3].guidelines)

    assert "Do NOT choose AMP" in model_guidelines
    assert "datatype_precision step owns" in model_guidelines
    assert "Do NOT change model architecture" in dtype_guidelines
    assert "Do NOT redefine or reload data/features" in training_guidelines
    assert "Own training hyperparameters" in training_guidelines


def test_stepwise_generation_can_return_stage_metadata(monkeypatch) -> None:
    responses = []

    def fake_generate(**kwargs):
        responses.append(kwargs["prompt"])
        return "Plan for this step.\n```python\nVALUE = 1\n```"

    monkeypatch.setattr("agents.coder.stepwise_coder.generate", fake_generate)
    agent = SimpleNamespace(
        acfg=SimpleNamespace(
            code=SimpleNamespace(temp=0.0, model="test-model"),
            hardware_context_enabled=True,
        ),
        cfg=SimpleNamespace(experiment=SimpleNamespace(mode="hardware_aware")),
    )

    plan, code, metadata = stepwise_plan_and_code_query(
        agent_instance=agent,
        prompt_base={
            "Introduction": "intro",
            "Task description": "image classification",
            "Memory": "",
            "Instructions": {},
        },
        data_preview="train_images",
        context={
            "stage": "draft",
            "hardware_prompt_section": "# Combined hardware context\n",
            "hardware_stage_sections": {
                "model_design": "# Hardware-Aware Model Design Brief\n",
                "datatype_precision": "# Hardware-Aware Datatype/Precision Context\n",
                "training_evaluation": "# Hardware-Aware Training Hyperparameter Context\n",
            },
        },
        return_metadata=True,
    )

    assert plan
    assert "VALUE = 1" in code
    assert metadata["step_order"] == [
        "data_processing_and_feature_engineering",
        "model_design",
        "datatype_precision",
        "training_evaluation",
    ]
    assert metadata["decisions"][2]["stage"] == "datatype_precision"
    assert any("Datatype/Precision" in str(prompt) for prompt in responses)


def test_stepwise_hardware_decisions_are_stored_as_ordered_pipeline() -> None:
    root = SearchNode(code="", plan="root", stage="root")
    node = SearchNode(
        code=(
            "MODEL_NAME = 'vit_base_patch16_224'\n"
            "PRECISION_MODE = 'bf16'\n"
            "BATCH_SIZE = 16\n"
            "EPOCHS = 2\n"
            "LR = 1e-3\n"
            "WEIGHT_DECAY = 0.01\n"
            "GRADIENT_ACCUMULATION_STEPS = 2\n"
            "NUM_WORKERS = 4\n"
        ),
        plan="draft",
        parent=root,
        stage="draft",
    )
    design_context = SimpleNamespace(
        compact_context={
            "model_options": [{"model_family": "vision_transformer"}],
            "evidence_refs": ["design:1"],
            "confidence": 0.7,
        }
    )
    execution_context = SimpleNamespace(
        compact_context={
            "evidence_refs": ["exec:1"],
            "confidence": 0.8,
        }
    )
    metadata = {
        "decisions": [
            {"stage": "model_design", "plan": "Use a ViT interface.", "hardware_context_used": True},
            {"stage": "datatype_precision", "plan": "Use bf16 autocast.", "hardware_context_used": True},
            {"stage": "training_evaluation", "plan": "Tune batch and LR.", "hardware_context_used": True},
        ]
    }

    apply_stepwise_hardware_decisions_to_node(
        node,
        metadata,
        design_context=design_context,
        execution_context=execution_context,
    )

    pipeline = node.hardware_decision["pipeline"]
    assert [decision["stage"] for decision in pipeline] == [
        "model_design",
        "datatype_precision",
        "training_evaluation",
    ]
    assert pipeline[1]["chosen_params"]["precision_mode"] == "bf16"
    assert pipeline[2]["chosen_params"]["learning_rate"] == 1e-3


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


def test_hardware_design_brief_fetches_only_selected_feature_details(monkeypatch) -> None:
    class FakeScheduler:
        def __init__(self) -> None:
            self.detail_calls = []
            self.settings = SimpleNamespace(
                gpu_scheduler=SimpleNamespace(
                    mode="parallel_auto_pack",
                    backend_priority=["stream", "cuda_process", "exclusive"],
                    concurrent_backend_allowlist=["stream"],
                    submission_defaults=SimpleNamespace(
                        requires_gpu=True,
                        packing_family="mlevolve_script",
                        backend_allowlist=[],
                        batch_probe_model_key=None,
                    ),
                )
            )

        def get_model_design_hardware_context(self, **kwargs):
            return {
                "found": True,
                "hardware_context": {
                    "found": True,
                    "hardware": {"gpu_name": "NVIDIA GeForce RTX 5090", "hardware_key": "hw-current"},
                },
                "hardware_feature_index": {
                    "found": True,
                    "hardware": {"hardware_id": "nvidia.blackwell.geforce_rtx_5090.spec", "name": "GeForce RTX 5090"},
                    "features": [
                        {
                            "feature_id": "bf16",
                            "feature_name": "BF16",
                            "category": "precision_optimization",
                            "support_level": "native",
                            "performance_impact": "high",
                        },
                        {
                            "feature_id": "fp8",
                            "feature_name": "FP8",
                            "category": "precision_optimization",
                            "support_level": "supported",
                            "performance_impact": "medium",
                        },
                    ],
                    "feature_count": 2,
                    "source": "redis",
                },
                "workload_type": kwargs.get("workload_type"),
                "model_options": [
                    {
                        "model_family": "vision_transformer",
                        "rationale": "Good task fit when precision is supported.",
                        "confidence": 0.6,
                    }
                ],
                "recommendations": ["Use selected hardware details only when they apply."],
                "confidence": 0.6,
            }

        def get_hardware_feature_details(self, *, hardware_id, feature_ids, limit):
            self.detail_calls.append((hardware_id, list(feature_ids), limit))
            return {
                "found": True,
                "hardware": {"hardware_id": "nvidia.blackwell.geforce_rtx_5090.spec", "name": "GeForce RTX 5090"},
                "features": [
                    {
                        "feature_id": "bf16",
                        "feature_name": "BF16",
                        "category": "precision_optimization",
                        "support_level": "native",
                        "summary_text": "Native BF16 tensor-core path.",
                        "recommended_patterns": ["Use torch.autocast with bf16."],
                        "avoid_patterns": ["Avoid if numerically unstable."],
                    }
                ],
            }

    scheduler = FakeScheduler()
    agent = SimpleNamespace(
        scheduler_client=scheduler,
        acfg=SimpleNamespace(
            hardware_context_enabled=True,
            hardware_context_limit=4,
            hardware_context_max_prompt_chars=4000,
            code=SimpleNamespace(temp=0.7),
        ),
        cfg=SimpleNamespace(exp_id="task-a"),
        task_desc="image classification",
        data_preview="train images and labels",
    )
    monkeypatch.setattr("agents.hardware_context.generate", lambda **_: '{"feature_ids": ["bf16", "made_up"]}')

    context = get_hardware_design_brief(agent)

    assert scheduler.detail_calls == [("current", ["bf16"], 1)]
    assert "Available hardware feature keys linked to this hardware" in context.prompt_section
    assert "Selected hardware feature details for model_design" in context.prompt_section
    assert "Native BF16 tensor-core path" in context.prompt_section
    assert "fp8" in context.prompt_section
    assert "detail for fp8" not in context.prompt_section


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

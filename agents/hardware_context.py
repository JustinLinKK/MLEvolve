"""Hardware/profile context helpers for MLEvolve stage prompts."""

from __future__ import annotations

import logging
import json
import re
from dataclasses import dataclass, field
from typing import Any

from llm import generate
from engine.script_introspection import introspect_training_script

logger = logging.getLogger("MLEvolve")

HARDWARE_CONTEXT_HEADING = "# Hardware/Profile Optimization Context"
HARDWARE_DESIGN_HEADING = "# Hardware-Aware Model Design Brief"
HARDWARE_DATATYPE_HEADING = "# Hardware-Aware Datatype/Precision Context"
HARDWARE_TRAINING_HEADING = "# Hardware-Aware Training Hyperparameter Context"
EVIDENCE_NOT_LAW_RULE = (
    "Treat recommendations as empirical evidence, not hard rules. Follow high-confidence hardware/profile "
    "guidance by default; if a scoring reason requires deviating, state why and include a fallback such as "
    "smaller physical batch size, gradient accumulation, AMP, reduced resolution, fewer epochs, or checkpointing."
)
CONSTRAINT_PRECEDENCE_RULE = (
    "Hardware advice never overrides task, dataset, submission, package, model-source, or filesystem constraints. "
    "Do not invent Kaggle input paths, local checkpoints, torch hub directories, or placeholder weights; use only "
    "paths and model sources that are explicitly available in the prompt/config."
)
HARDWARE_BUDGET_GUARDRAIL_RULE = (
    "For hardware/runtime optimization, preserve the parent solution's modeling budget unless the user explicitly "
    "asks to improve score. Do not increase epochs, folds, model size, input resolution, ensemble count, TTA, "
    "dataset size, or validation workload as a hardware-only optimization."
)
_PRECISION_KEYWORDS = (
    "precision",
    "dtype",
    "amp",
    "autocast",
    "bf16",
    "bfloat16",
    "fp16",
    "float16",
    "tf32",
    "float32",
    "fp32",
    "gradscaler",
    "grad scaler",
    "tensor core",
    "matmul",
)
_TRAINING_HPARAM_KEYWORDS = (
    "batch",
    "epoch",
    "accum",
    "learning rate",
    "lr",
    "weight decay",
    "dataloader",
    "num_workers",
    "worker",
    "pin_memory",
    "persistent_workers",
    "checkpoint",
    "runtime",
    "throughput",
    "vram",
    "packing",
    "oom",
    "timeout",
)
_DATATYPE_EXCLUDE_KEYWORDS = (
    "batch size",
    "epochs",
    "epoch budget",
    "learning rate",
    "weight decay",
    "num_workers",
    "dataloader workers",
    "gradient accumulation",
    "checkpoint",
)

_STAGE_NODE_FIELD_LIMITS: dict[str, tuple[tuple[str, int], ...]] = {
    "datatype": (
        ("software_features", 6),
        ("recommended_patterns", 3),
        ("avoid_patterns", 3),
    ),
    "model": (
        ("datatypes", 4),
        ("software_features", 6),
        ("recommended_patterns", 3),
        ("avoid_patterns", 3),
    ),
    "optimizer": (
        ("recipes", 4),
        ("recommended_patterns", 3),
        ("avoid_patterns", 3),
    ),
    "tuning": (
        ("datatypes", 6),
        ("software_features", 6),
        ("recommended_patterns", 3),
        ("avoid_patterns", 3),
    ),
}
_DEFAULT_STAGE_NODE_FIELD_LIMITS: tuple[tuple[str, int], ...] = (
    ("datatypes", 4),
    ("software_features", 4),
    ("recommended_patterns", 3),
    ("avoid_patterns", 3),
)
_STAGE_DIRECT_VALUE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "datatype": (
        "data",
        "dataset",
        "dataloader",
        "decode",
        "image",
        "video",
        "wsi",
        "tile",
        "tiled",
        "chunk",
        "decomposition",
        "sequence",
        "packing",
        "channels_last",
    ),
    "model": (
        "attention",
        "sdpa",
        "flash",
        "tensor_core",
        "tensor",
        "kernel",
        "compile",
        "channels_last",
        "unet",
        "cnn",
        "transformer",
        "activation",
        "checkpoint",
        "sm_",
        "nvlink",
        "mig",
        "topology",
    ),
    "optimizer": (
        "optimizer",
        "adam",
        "muon",
        "soap",
        "ademamix",
        "loss",
        "cross_entropy",
        "cross-entropy",
        "gram_newton",
        "newton_schulz",
        "scheduler",
        "learning_rate",
        "lr",
        "weight_decay",
        "momentum",
    ),
    "tuning": (
        "bf16",
        "fp16",
        "fp8",
        "fp4",
        "fp64",
        "tf32",
        "int8",
        "amp",
        "precision",
        "autocast",
        "quant",
        "grad",
        "batch",
        "vram",
        "memory",
        "throughput",
        "parallel",
        "tensor_parallel",
    ),
}


@dataclass
class HardwarePromptContext:
    candidate: dict[str, Any] = field(default_factory=dict)
    raw_context: dict[str, Any] | None = None
    compact_context: dict[str, Any] = field(default_factory=dict)
    prompt_section: str = ""

    @property
    def found(self) -> bool:
        return bool(self.prompt_section.strip())


def hardware_context_instructions(context: HardwarePromptContext | None = None) -> dict[str, list[str]]:
    if context is not None and not context.found:
        return {}
    return {
        "Hardware/Profile reasoning rule": [
            EVIDENCE_NOT_LAW_RULE,
            CONSTRAINT_PRECEDENCE_RULE,
            HARDWARE_BUDGET_GUARDRAIL_RULE,
            "Use the current scheduler backend config from the hardware context as the execution contract; do not hardcode CUDA process, CUDA stream, MPS, or any other backend in generated code.",
            "Prefer scheduler-compatible code with configurable batch size, precision, dataloader workers, checkpoints, and runtime logging when using GPU training.",
        ]
    }


def get_hardware_design_brief(agent: Any) -> HardwarePromptContext:
    """Build the architecture-selection hardware brief used before draft code is written."""
    if not _hardware_context_enabled(agent):
        return HardwarePromptContext()
    scheduler_client = getattr(agent, "scheduler_client", None)
    if scheduler_client is None:
        return HardwarePromptContext()

    try:
        workload_type = infer_workload_type(getattr(agent, "task_desc", ""), getattr(agent, "data_preview", ""))
        candidate = build_hardware_candidate(
            agent,
            "model_design",
            extra_candidate={
                "workload_type": workload_type,
                "task_type": workload_type,
                "design_stage": "model_family_selection",
                "candidate_families": _default_model_families_for_workload(workload_type),
            },
        )
    except Exception as exc:
        logger.debug("Hardware model-design candidate construction failed: %s", exc)
        return HardwarePromptContext()

    try:
        limit = _safe_int(getattr(agent.acfg, "hardware_context_limit", 8), default=8)
        if hasattr(scheduler_client, "get_model_design_hardware_context"):
            raw_context = scheduler_client.get_model_design_hardware_context(
                workload_type=candidate.get("workload_type"),
                task_type=candidate.get("task_type"),
                candidate_families=list(candidate.get("candidate_families") or []),
                hardware_key="current",
                limit=limit,
            )
        else:
            raw_context = scheduler_client.get_optimization_context(candidate=candidate, limit=limit)
    except Exception as exc:
        logger.debug("Hardware model-design context lookup failed: %s", exc)
        return HardwarePromptContext(candidate=candidate)

    raw_context = dict(raw_context or {})
    initial_compact = compact_model_design_context(raw_context)
    selected_feature_ids = _select_hardware_feature_ids_for_design(agent, candidate, initial_compact)
    if selected_feature_ids and hasattr(scheduler_client, "get_hardware_feature_details"):
        try:
            raw_context["selected_hardware_feature_ids"] = selected_feature_ids
            raw_context["selected_hardware_feature_details"] = scheduler_client.get_hardware_feature_details(
                hardware_id="current",
                feature_ids=selected_feature_ids,
                limit=max(1, len(selected_feature_ids)),
            )
        except Exception as exc:
            logger.debug("Hardware feature detail lookup failed: %s", exc)

    compact = compact_model_design_context(raw_context)
    max_chars = _safe_int(getattr(agent.acfg, "hardware_context_max_prompt_chars", 3500), default=3500)
    prompt_section = format_hardware_design_brief(compact, max_chars=max_chars)
    return HardwarePromptContext(
        candidate=candidate,
        raw_context=raw_context,
        compact_context=compact,
        prompt_section=prompt_section,
    )


def get_hardware_context_for_stage(
    agent: Any,
    stage: str,
    *,
    parent_node: Any | None = None,
    code: str | None = None,
    extra_candidate: dict[str, Any] | None = None,
) -> HardwarePromptContext:
    """Build prompt-ready hardware context. Failure is non-fatal by design."""
    if not _hardware_context_enabled(agent):
        return HardwarePromptContext()

    scheduler_client = getattr(agent, "scheduler_client", None)
    if scheduler_client is None:
        return HardwarePromptContext()

    try:
        candidate = build_hardware_candidate(
            agent,
            stage,
            parent_node=parent_node,
            code=code,
            extra_candidate=extra_candidate,
        )
    except Exception as exc:
        logger.debug("Hardware/profile candidate construction failed for stage %s: %s", stage, exc)
        return HardwarePromptContext()
    if not candidate:
        return HardwarePromptContext(candidate=candidate)

    try:
        limit = _safe_int(getattr(agent.acfg, "hardware_context_limit", 8), default=8)
        raw_context = scheduler_client.get_optimization_context(candidate=candidate, limit=limit)
    except Exception as exc:
        logger.debug("Hardware/profile context lookup failed for stage %s: %s", stage, exc)
        return HardwarePromptContext(candidate=candidate)

    compact = compact_optimization_context(raw_context)
    max_chars = _safe_int(getattr(agent.acfg, "hardware_context_max_prompt_chars", 3500), default=3500)
    prompt_section = format_hardware_prompt_section_for_stage(compact, stage=stage, max_chars=max_chars)
    return HardwarePromptContext(
        candidate=candidate,
        raw_context=raw_context,
        compact_context=compact,
        prompt_section=prompt_section,
    )


def refresh_node_hardware_context(agent: Any, node: Any) -> HardwarePromptContext:
    context = get_hardware_context_for_stage(
        agent,
        getattr(node, "stage", "unknown"),
        parent_node=getattr(node, "parent", None),
        code=getattr(node, "code", None),
    )
    apply_hardware_context_to_node(node, context)
    return context


def apply_hardware_context_to_node(node: Any, context: HardwarePromptContext | None) -> None:
    if context is None or not context.compact_context:
        return
    compact = context.compact_context
    node.hardware_context = compact.get("hardware_context")
    node.graph_evidence = compact.get("graph_evidence")
    node.derived_diagnosis = compact.get("derived_diagnosis")
    node.vector_evidence = compact.get("vector_evidence")
    node.scheduler_risk_flags = list(compact.get("risk_flags") or [])
    node.scheduler_confidence = float(compact.get("confidence") or 0.0)
    node.hardware_evidence_refs = list(compact.get("evidence_refs") or [])
    node.resolved_batch_size = _recommended_batch_size(compact) or context.candidate.get("proposed_batch_size")
    node.estimated_runtime_seconds = _runtime_seconds(compact)
    node.peak_vram_mb = _peak_vram_mb(compact)
    node.backend_name = _backend_name(compact)


def apply_hardware_design_brief_to_node(node: Any, context: HardwarePromptContext | None) -> None:
    if context is None or not context.compact_context:
        return
    compact = context.compact_context
    _store_hardware_decision(node, {
        "stage": "model_design",
        "rationale": "Hardware-aware model design brief was provided before draft generation.",
        "chosen_params": {},
        "original_params": {},
        "model_options": compact.get("model_options") or [],
        "recommendations": compact.get("recommendations") or [],
        "evidence_refs": list(compact.get("evidence_refs") or []),
        "confidence": float(compact.get("confidence") or 0.0),
        "fallback_reason": None if compact.get("model_options") else "no model-family hardware evidence found",
    })


def build_stepwise_hardware_stage_sections(
    *,
    design_context: HardwarePromptContext | None,
    execution_context: HardwarePromptContext | None,
    max_chars: int = 3500,
) -> dict[str, str]:
    """Return focused hardware sections for the internal stepwise agents."""
    design_section = design_context.prompt_section if design_context is not None else ""
    compact = execution_context.compact_context if execution_context is not None else {}
    generic_section = format_hardware_prompt_section(compact, max_chars=max_chars) if compact else ""
    datatype_section = format_hardware_datatype_prompt_section(compact, max_chars=max_chars) if compact else ""
    training_section = format_hardware_training_prompt_section(compact, max_chars=max_chars) if compact else ""
    data_feature_section = _format_stage_specific_hardware_features(compact, ("datatype",), max_chars=max_chars)
    model_feature_section = _format_stage_specific_hardware_features(compact, ("model",), max_chars=max_chars)
    precision_feature_section = _format_stage_specific_hardware_features(compact, ("tuning",), max_chars=max_chars)
    training_feature_section = _format_stage_specific_hardware_features(
        compact,
        ("optimizer", "tuning"),
        max_chars=max_chars,
    )
    return {
        "data_processing_and_feature_engineering": data_feature_section or generic_section,
        "model_design": _join_prompt_sections((design_section, model_feature_section), max_chars=max_chars)
        or design_section
        or generic_section,
        "datatype_precision": _join_prompt_sections((datatype_section, precision_feature_section), max_chars=max_chars)
        or generic_section,
        "training_evaluation": _join_prompt_sections((training_section, training_feature_section), max_chars=max_chars)
        or generic_section,
    }


def _join_prompt_sections(sections: tuple[str, ...], *, max_chars: int) -> str:
    text = "\n".join(section.strip() for section in sections if section and section.strip()).strip()
    if not text:
        return ""
    if len(text) > max_chars:
        text = text[: max(0, max_chars - 48)].rstrip() + "\n... [hardware context truncated]"
    return text + "\n"


def _format_stage_specific_hardware_features(
    compact: dict[str, Any],
    stage_filters: tuple[str, ...],
    *,
    max_chars: int,
) -> str:
    stage_context = _filter_stage_hardware_features(
        compact.get("stage_hardware_features") or {},
        stage_filters,
    )
    lines = _format_stage_hardware_features(stage_context)
    if not lines:
        return ""
    text = "\n".join([HARDWARE_CONTEXT_HEADING, *lines]).strip()
    if len(text) > max_chars:
        text = text[: max(0, max_chars - 48)].rstrip() + "\n... [stage hardware context truncated]"
    return text + "\n"


def _filter_stage_hardware_features(
    stage_context: dict[str, Any],
    stage_filters: tuple[str, ...],
) -> dict[str, Any]:
    if not stage_context or not stage_context.get("found"):
        return {}
    allowed = {str(stage).strip().lower() for stage in stage_filters if str(stage).strip()}
    stages = [
        dict(stage)
        for stage in list(stage_context.get("stages") or [])
        if str(stage.get("stage") or "").strip().lower() in allowed
    ]
    if not stages:
        return {}
    feature_ids: list[str] = []
    feature_count = 0
    for stage in stages:
        feature_count += int(stage.get("feature_count") or 0)
        for feature in list(stage.get("features") or []):
            feature_id = str(feature.get("feature_id") or "").strip()
            if feature_id and feature_id not in feature_ids:
                feature_ids.append(feature_id)
    filtered = dict(stage_context)
    filtered["stages"] = stages
    filtered["stage_filter"] = [stage.get("stage") for stage in stages if stage.get("stage")]
    filtered["feature_ids"] = feature_ids
    filtered["feature_count"] = feature_count
    return {key: value for key, value in filtered.items() if value not in (None, "", [], {})}


def apply_stepwise_hardware_decisions_to_node(
    node: Any,
    metadata: dict[str, Any] | None,
    *,
    design_context: HardwarePromptContext | None,
    execution_context: HardwarePromptContext | None,
) -> None:
    """Store an ordered hardware-aware generation record on a search node."""
    if not metadata:
        return
    step_decisions = list(metadata.get("decisions") or [])
    if not step_decisions:
        return

    generated_candidate = introspect_training_script(getattr(node, "code", "") or "")
    design_compact = design_context.compact_context if design_context is not None else {}
    execution_compact = execution_context.compact_context if execution_context is not None else {}
    pipeline: list[dict[str, Any]] = []
    for item in step_decisions:
        stage = str(item.get("stage") or "")
        if stage not in {"model_design", "datatype_precision", "training_evaluation"}:
            continue
        compact = design_compact if stage == "model_design" else execution_compact
        decision = {
            "stage": stage,
            "rationale": _stage_decision_rationale(stage),
            "step_plan": item.get("plan") or "",
            "chosen_params": _stage_chosen_params(stage, generated_candidate),
            "original_params": {},
            "evidence_refs": list(compact.get("evidence_refs") or []),
            "confidence": float(compact.get("confidence") or 0.0),
            "fallback_reason": None if item.get("hardware_context_used") else "no hardware context was available for this internal step",
        }
        if stage == "model_design":
            decision["model_options"] = list(design_compact.get("model_options") or [])
        pipeline.append(decision)
    if pipeline:
        _set_hardware_decision_pipeline(node, pipeline)


def optimize_training_parameters_for_round(agent: Any, nodes: list[Any]) -> list[dict[str, Any]]:
    """Use scheduler evidence to safely tune generated training parameters before a round submission."""
    if not nodes or not _hardware_context_enabled(agent):
        return []
    scheduler_client = getattr(agent, "scheduler_client", None)
    if scheduler_client is None:
        return []

    candidates: list[dict[str, Any]] = []
    for node in nodes:
        try:
            candidates.append(
                build_hardware_candidate(
                    agent,
                    "pre_submit_training_review",
                    parent_node=getattr(node, "parent", None),
                    code=getattr(node, "code", "") or "",
                    extra_candidate={"node_id": str(getattr(node, "id", ""))},
                )
            )
        except Exception as exc:
            logger.debug("Skipping hardware training review candidate for node %s: %s", getattr(node, "id", ""), exc)
            candidates.append({})

    packet_context: dict[str, Any] = {}
    if hasattr(scheduler_client, "plan_job_packet"):
        try:
            limit = _safe_int(getattr(agent.acfg, "hardware_context_limit", 8), default=8)
            packet_context = scheduler_client.plan_job_packet(candidates=candidates, limit=limit)
        except Exception as exc:
            logger.debug("Hardware packet planning failed; falling back to per-node context: %s", exc)
            packet_context = {}

    packet_jobs = list(packet_context.get("jobs") or [])
    decisions: list[dict[str, Any]] = []
    for idx, node in enumerate(nodes):
        candidate = candidates[idx] if idx < len(candidates) else {}
        raw_context = None
        if idx < len(packet_jobs):
            raw_context = packet_jobs[idx].get("optimization_context")
        if raw_context is None:
            try:
                limit = _safe_int(getattr(agent.acfg, "hardware_context_limit", 8), default=8)
                raw_context = scheduler_client.get_optimization_context(candidate=candidate, limit=limit)
            except Exception as exc:
                logger.debug("Hardware training review lookup failed for node %s: %s", getattr(node, "id", ""), exc)
                raw_context = {}

        compact = compact_optimization_context(raw_context)
        original_code = getattr(node, "code", "") or ""
        original_params = {
            "batch_size": candidate.get("proposed_batch_size"),
            "epochs": candidate.get("proposed_epochs"),
        }
        recommended_epochs = _recommended_epochs(compact)
        original_epochs = _safe_int(candidate.get("proposed_epochs"), default=None)
        if recommended_epochs is not None and (original_epochs is None or recommended_epochs > original_epochs):
            recommended_epochs = None
        chosen_params = {
            "batch_size": _recommended_batch_size(compact),
            "epochs": recommended_epochs,
        }
        updated_code, applied = _rewrite_training_params(original_code, chosen_params)
        if applied:
            node.code = updated_code
        decision = {
            "stage": "training_parameter_review",
            "rationale": "Reviewed graph/probe/packing evidence before scheduler round submission.",
            "original_params": {key: value for key, value in original_params.items() if value is not None},
            "chosen_params": {key: value for key, value in chosen_params.items() if value is not None},
            "applied_params": applied,
            "evidence_refs": list(compact.get("evidence_refs") or packet_context.get("evidence_refs") or []),
            "confidence": float(compact.get("confidence") or packet_context.get("confidence") or 0.0),
            "fallback_reason": None if applied else "no safe literal training-parameter assignment found or no stronger evidence available",
            "packet_id": packet_context.get("packet_id"),
        }
        previous_decision = getattr(node, "hardware_decision", None)
        if previous_decision:
            decision["previous_decision"] = _latest_decision_without_pipeline(previous_decision)
        _store_hardware_decision(node, decision)
        decisions.append(decision)
    return decisions


def build_hardware_candidate(
    agent: Any,
    stage: str,
    *,
    parent_node: Any | None = None,
    code: str | None = None,
    extra_candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    code_text = code
    if code_text is None and parent_node is not None:
        code_text = getattr(parent_node, "code", "") or ""
    code_text = code_text or ""

    candidate = introspect_training_script(code_text)
    workload_type = infer_workload_type(getattr(agent, "task_desc", ""), getattr(agent, "data_preview", ""))
    candidate.update(
        {
            "stage": stage,
            "task_type": workload_type,
            "workload_type": workload_type,
            "framework": candidate.get("framework") or "pytorch",
        }
    )

    if parent_node is not None:
        if candidate.get("proposed_batch_size") is None and getattr(parent_node, "resolved_batch_size", None) is not None:
            candidate["proposed_batch_size"] = parent_node.resolved_batch_size
        if getattr(parent_node, "estimated_runtime_seconds", None) is not None:
            candidate["notes"] = f"parent_estimated_runtime_seconds={parent_node.estimated_runtime_seconds}"
        hints = _execution_resource_hints(_safe_node_term_out(parent_node))
        for key, value in hints.items():
            candidate.setdefault(key, value)

    scheduler_defaults = _scheduler_submission_defaults(getattr(agent, "scheduler_client", None))
    if scheduler_defaults is not None:
        candidate.setdefault("requires_gpu", bool(getattr(scheduler_defaults, "requires_gpu", True)))
        candidate.setdefault("packing_family", getattr(scheduler_defaults, "packing_family", None))
        backend_allowlist = list(getattr(scheduler_defaults, "backend_allowlist", []) or [])
        if backend_allowlist:
            candidate.setdefault("backend_preference", backend_allowlist[0])
        if not candidate.get("model_key") and getattr(scheduler_defaults, "batch_probe_model_key", None):
            candidate["model_key"] = getattr(scheduler_defaults, "batch_probe_model_key")
    scheduler_backend_config = _scheduler_backend_config(getattr(agent, "scheduler_client", None))
    if scheduler_backend_config:
        candidate.setdefault("scheduler_mode", scheduler_backend_config.get("mode"))
        candidate.setdefault("scheduler_effective_mode", scheduler_backend_config.get("effective_mode"))
        if not candidate.get("backend_preference"):
            backend_priority = [
                backend_name
                for backend_name in list(scheduler_backend_config.get("backend_priority") or [])
                if backend_name != "exclusive"
            ]
            if backend_priority:
                candidate["backend_preference"] = backend_priority[0]

    if not candidate.get("model_key"):
        exp_id = getattr(getattr(agent, "cfg", None), "exp_id", None) or "mlevolve"
        candidate["model_key"] = f"mlevolve-task:{exp_id}"

    if not candidate.get("packing_family") and candidate.get("model_family"):
        candidate["packing_family"] = candidate["model_family"]

    if extra_candidate:
        candidate.update({key: value for key, value in extra_candidate.items() if value is not None})

    return {key: value for key, value in candidate.items() if value is not None and value != ""}


def compact_optimization_context(raw_context: dict[str, Any] | None) -> dict[str, Any]:
    raw_context = raw_context or {}
    compact = {
        "hardware_context": _compact_hardware_context(raw_context.get("hardware_context") or {}),
        "graph_evidence": _compact_graph_evidence(raw_context.get("graph_evidence") or {}),
        "derived_diagnosis": _compact_diagnosis(raw_context.get("derived_diagnosis") or {}),
        "stage_hardware_features": _compact_stage_hardware_features(raw_context.get("stage_hardware_features") or {}),
        "vector_evidence": _compact_vector_evidence(raw_context.get("vector_evidence") or {}),
        "recommendations": _clean_string_list(raw_context.get("recommendations") or [], limit=8),
        "risk_flags": _clean_string_list(raw_context.get("risk_flags") or [], limit=8),
        "evidence_refs": _clean_string_list(raw_context.get("evidence_refs") or [], limit=16),
        "confidence": round(float(raw_context.get("confidence") or 0.0), 3),
    }
    return compact


def _select_hardware_feature_ids_for_design(
    agent: Any,
    candidate: dict[str, Any],
    compact: dict[str, Any],
    *,
    max_features: int = 4,
) -> list[str]:
    feature_index = list((compact.get("hardware_feature_index") or {}).get("features") or [])
    if not feature_index:
        return []
    available_ids = [str(item.get("feature_id") or "").strip() for item in feature_index if item.get("feature_id")]
    available_set = set(available_ids)
    if not available_set:
        return []
    feature_lines = []
    for item in feature_index[:32]:
        feature_id = item.get("feature_id")
        if not feature_id:
            continue
        feature_lines.append(
            "- {feature_id}: {name}; category={category}; support={support}; impact={impact}".format(
                feature_id=feature_id,
                name=item.get("feature_name") or "",
                category=item.get("category") or "",
                support=item.get("support_level") or "",
                impact=item.get("performance_impact") or "",
            )
        )
    if not feature_lines:
        return []
    prompt = (
        "Select only the hardware feature IDs that are directly useful for the model_design stage.\n"
        "Return a JSON object exactly like {\"feature_ids\": [\"bf16\"]}. "
        f"Choose at most {max_features} IDs. Choose zero if none are relevant.\n\n"
        f"Task description:\n{getattr(agent, 'task_desc', '')}\n\n"
        f"Data preview:\n{getattr(agent, 'data_preview', '')}\n\n"
        f"Workload: {candidate.get('workload_type') or candidate.get('task_type') or compact.get('workload_type')}\n\n"
        "Available feature index:\n"
        + "\n".join(feature_lines)
    )
    try:
        response = generate(
            prompt=prompt,
            temperature=0.0,
            cfg=agent.cfg,
        )
    except Exception as exc:
        logger.debug("Hardware feature selector failed: %s", exc)
        return []
    return _parse_selected_feature_ids(response, available_set=available_set, max_features=max_features)


def _parse_selected_feature_ids(response: str, *, available_set: set[str], max_features: int) -> list[str]:
    text = str(response or "").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return []
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    raw_ids = payload.get("feature_ids") if isinstance(payload, dict) else []
    if not isinstance(raw_ids, list):
        return []
    selected: list[str] = []
    for item in raw_ids:
        feature_id = str(item or "").strip()
        if feature_id in available_set and feature_id not in selected:
            selected.append(feature_id)
        if len(selected) >= max(0, int(max_features)):
            break
    return selected


def compact_model_design_context(raw_context: dict[str, Any] | None) -> dict[str, Any]:
    raw_context = raw_context or {}
    model_options = []
    for item in list(raw_context.get("model_options") or raw_context.get("ranked_options") or [])[:6]:
        option = _pick(
            dict(item),
            (
                "model_family",
                "model_key",
                "summary",
                "rationale",
                "hardware_features",
                "expected_benefits",
                "risks",
                "score",
                "confidence",
            ),
        )
        if option:
            model_options.append(option)
    feature_index_raw = raw_context.get("hardware_feature_index") or {}
    feature_index = {
        "found": bool(feature_index_raw.get("found")),
        "hardware": _pick(dict(feature_index_raw.get("hardware") or {}), ("hardware_id", "name", "name_key", "architecture")),
        "features": [
            _compact_hardware_feature_index_item(item)
            for item in list(feature_index_raw.get("features") or [])[:32]
        ],
        "feature_count": feature_index_raw.get("feature_count"),
        "stage_filter": feature_index_raw.get("stage_filter"),
        "source": feature_index_raw.get("source"),
    }
    feature_index = {key: value for key, value in feature_index.items() if value not in (None, {}, [], "")}
    selected_details = raw_context.get("selected_hardware_feature_details") or {}
    selected_features = [
        _compact_hardware_feature_detail(item)
        for item in list(selected_details.get("features") or [])[:6]
    ]
    compact = {
        "hardware_context": _compact_hardware_context(raw_context.get("hardware_context") or {}),
        "hardware_feature_index": feature_index,
        "selected_hardware_feature_ids": _clean_string_list(raw_context.get("selected_hardware_feature_ids") or [], limit=8),
        "selected_hardware_features": selected_features,
        "workload_type": raw_context.get("workload_type"),
        "model_options": model_options,
        "recommendations": _clean_string_list(raw_context.get("recommendations") or [], limit=8),
        "risk_flags": _clean_string_list(raw_context.get("risk_flags") or [], limit=8),
        "evidence_refs": _clean_string_list(raw_context.get("evidence_refs") or [], limit=16),
        "confidence": round(float(raw_context.get("confidence") or 0.0), 3),
    }
    if not compact["model_options"] and raw_context.get("recommendations"):
        compact["model_options"] = [
            {
                "model_family": "baseline_compatible",
                "rationale": "No ranked architecture evidence was available; keep baseline-style architecture choice and apply only safe training optimizations.",
                "confidence": compact["confidence"],
            }
        ]
    return {key: value for key, value in compact.items() if value not in (None, {}, [], "")}


def format_hardware_design_brief(compact: dict[str, Any], *, max_chars: int = 3500) -> str:
    if not compact:
        return ""
    lines = [HARDWARE_DESIGN_HEADING]
    hardware = compact.get("hardware_context") or {}
    if hardware:
        lines.append(f"- Hardware: {hardware.get('summary') or 'current hardware'}")
        backend = hardware.get("backend_capabilities") or {}
        if backend:
            backend_bits = _format_kv(
                backend,
                (
                    "mode",
                    "effective_mode",
                    "backend_priority",
                    "enabled_backends",
                    "concurrent_groups_enabled",
                    "concurrent_backend_allowlist",
                ),
            )
            if backend_bits:
                lines.append(f"- Scheduler backend config: {backend_bits}")
        limits = hardware.get("scheduler_limits") or {}
        if limits:
            limit_bits = _format_kv(
                limits,
                ("safe_vram_budget_mb", "max_packed_jobs_per_gpu", "mode", "effective_mode", "backend_priority"),
            )
            if limit_bits:
                lines.append(f"- Scheduler limits: {limit_bits}")
    workload = compact.get("workload_type")
    if workload:
        lines.append(f"- Workload: {workload}")
    options = list(compact.get("model_options") or [])
    if options:
        lines.append("- Candidate model-family options:")
        for option in options[:5]:
            family = option.get("model_family") or option.get("model_key") or "candidate"
            rationale = option.get("rationale") or option.get("summary") or ""
            benefits = option.get("expected_benefits") or option.get("hardware_features") or []
            risk = option.get("risks") or []
            details = _format_kv(option, ("score", "confidence"))
            suffix = f" ({details})" if details else ""
            benefit_text = f"; benefits={benefits}" if benefits else ""
            risk_text = f"; risks={risk}" if risk else ""
            lines.append(f"  - {family}{suffix}: {_short(rationale, 180)}{benefit_text}{risk_text}")
    feature_index = compact.get("hardware_feature_index") or {}
    feature_keys = list(feature_index.get("features") or [])
    if feature_keys:
        lines.append("- Available hardware feature keys linked to this hardware:")
        for feature in feature_keys[:16]:
            details = _format_kv(
                feature,
                ("category", "support_level", "performance_impact", "recommended", "confidence"),
            )
            suffix = f" ({details})" if details else ""
            name = feature.get("feature_name") or ""
            lines.append(f"  - {feature.get('feature_id')}: {name}{suffix}")
        if len(feature_keys) > 16:
            lines.append(f"  - ... {len(feature_keys) - 16} more feature key(s) omitted from the prompt")
    selected_features = list(compact.get("selected_hardware_features") or [])
    if selected_features:
        lines.append("- Selected hardware feature details for model_design:")
        for feature in selected_features[:4]:
            feature_id = feature.get("feature_id") or "feature"
            name = feature.get("feature_name") or feature.get("title") or ""
            summary = feature.get("summary_text") or feature.get("detail_text") or ""
            details = _format_kv(feature, ("category", "support_level", "performance_impact", "confidence"))
            suffix = f" ({details})" if details else ""
            lines.append(f"  - {feature_id}: {name}{suffix}: {_short(summary, 220)}")
            for pattern in feature.get("recommended_patterns") or []:
                lines.append(f"    recommended: {_short(pattern, 220)}")
            for pattern in feature.get("avoid_patterns") or []:
                lines.append(f"    avoid: {_short(pattern, 220)}")
            if feature.get("sample_code"):
                lines.append(f"    sample: {_short(feature['sample_code'], 260)}")
    recommendations = compact.get("recommendations") or []
    if recommendations:
        lines.append("- Design recommendations:")
        lines.extend(f"  - {item}" for item in recommendations)
    risks = compact.get("risk_flags") or []
    if risks:
        lines.append("- Design risk flags:")
        lines.extend(f"  - {item}" for item in risks)
    refs = compact.get("evidence_refs") or []
    if refs:
        lines.append(f"- Evidence refs: {', '.join(refs[:8])}")
    lines.append(f"- Confidence: {compact.get('confidence', 0.0)}")
    lines.append(
        "- Decision rule: Choose the architecture that best satisfies the task metric while using hardware features "
        "to reduce training time and improve GPU utilization. If hardware evidence is weak, keep a conservative "
        "baseline-compatible architecture."
    )
    lines.append(
        "- Feature-detail rule: The feature key list is only an index. Use detailed guidance only from the selected "
        "hardware feature details above; do not assume unexpanded feature keys have undocumented behavior."
    )
    lines.append(f"- Constraint rule: {CONSTRAINT_PRECEDENCE_RULE}")
    lines.append(f"- Budget guardrail: {HARDWARE_BUDGET_GUARDRAIL_RULE}")
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[: max(0, max_chars - 48)].rstrip() + "\n... [hardware design brief truncated]"
    return text + "\n"


def format_hardware_prompt_section(compact: dict[str, Any], *, max_chars: int = 3500) -> str:
    if not compact:
        return ""

    lines = [HARDWARE_CONTEXT_HEADING]
    hardware = compact.get("hardware_context") or {}
    if hardware:
        lines.append(f"- Hardware: {hardware.get('summary') or 'current hardware'}")
        backend = hardware.get("backend_capabilities") or {}
        if backend:
            backend_bits = _format_kv(
                backend,
                (
                    "mode",
                    "effective_mode",
                    "backend_priority",
                    "enabled_backends",
                    "concurrent_groups_enabled",
                    "concurrent_backend_allowlist",
                ),
            )
            if backend_bits:
                lines.append(f"- Scheduler backend config: {backend_bits}")
        limits = hardware.get("scheduler_limits") or {}
        if limits:
            limit_bits = _format_kv(
                limits,
                ("safe_vram_budget_mb", "max_packed_jobs_per_gpu", "mode", "effective_mode", "backend_priority"),
            )
            if limit_bits:
                lines.append(f"- Scheduler limits: {limit_bits}")

    stage_hardware_lines = _format_stage_hardware_features(compact.get("stage_hardware_features") or {})
    if stage_hardware_lines:
        lines.extend(stage_hardware_lines)

    diagnosis = compact.get("derived_diagnosis") or {}
    symptoms = diagnosis.get("profile_symptoms") or []
    targets = diagnosis.get("optimization_targets") or []
    if symptoms or targets:
        lines.append(f"- Diagnosis: symptoms={symptoms or ['none']} targets={targets or ['none']}")

    recommendations = compact.get("recommendations") or []
    if recommendations:
        lines.append("- Recommendations:")
        lines.extend(f"  - {item}" for item in recommendations)

    risks = compact.get("risk_flags") or []
    if risks:
        lines.append("- Risk flags:")
        lines.extend(f"  - {item}" for item in risks)

    graph_evidence = compact.get("graph_evidence") or {}
    graph_lines = _format_evidence_group("Graph evidence", graph_evidence)
    if graph_lines:
        lines.extend(graph_lines)

    vector_evidence = compact.get("vector_evidence") or {}
    vector_lines = _format_evidence_group("Code knowledge", vector_evidence)
    if vector_lines:
        lines.extend(vector_lines)

    refs = compact.get("evidence_refs") or []
    if refs:
        lines.append(f"- Evidence refs: {', '.join(refs[:8])}")
    lines.append(f"- Confidence: {compact.get('confidence', 0.0)}")
    lines.append(f"- Rule: {EVIDENCE_NOT_LAW_RULE}")
    lines.append(f"- Constraint rule: {CONSTRAINT_PRECEDENCE_RULE}")
    lines.append(f"- Budget guardrail: {HARDWARE_BUDGET_GUARDRAIL_RULE}")

    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[: max(0, max_chars - 40)].rstrip() + "\n... [hardware context truncated]"
    return text + "\n"


def format_hardware_prompt_section_for_stage(
    compact: dict[str, Any],
    *,
    stage: str,
    max_chars: int = 3500,
) -> str:
    stage_key = str(stage or "").strip().lower()
    if stage_key in {"datatype_precision", "precision", "dtype"}:
        return format_hardware_datatype_prompt_section(compact, max_chars=max_chars)
    if stage_key in {"training_evaluation", "pre_submit_training_review", "training_hyperparameters"}:
        return format_hardware_training_prompt_section(compact, max_chars=max_chars)
    return format_hardware_prompt_section(compact, max_chars=max_chars)


def format_hardware_datatype_prompt_section(compact: dict[str, Any], *, max_chars: int = 3500) -> str:
    if not compact:
        return ""

    lines = [HARDWARE_DATATYPE_HEADING]
    hardware = compact.get("hardware_context") or {}
    _append_hardware_summary(lines, hardware)

    diagnosis = _filter_diagnosis(compact.get("derived_diagnosis") or {}, _PRECISION_KEYWORDS)
    symptoms = diagnosis.get("profile_symptoms") or []
    targets = diagnosis.get("optimization_targets") or []
    if symptoms or targets:
        lines.append(f"- Precision diagnosis: symptoms={symptoms or ['none']} targets={targets or ['none']}")

    recommendations = _filter_string_list(
        compact.get("recommendations") or [],
        include_keywords=_PRECISION_KEYWORDS,
        exclude_keywords=_DATATYPE_EXCLUDE_KEYWORDS,
        limit=8,
    )
    if recommendations:
        lines.append("- Precision recommendations:")
        lines.extend(f"  - {item}" for item in recommendations)

    risks = _filter_string_list(
        compact.get("risk_flags") or [],
        include_keywords=_PRECISION_KEYWORDS,
        exclude_keywords=(),
        limit=8,
    )
    if risks:
        lines.append("- Precision risk flags:")
        lines.extend(f"  - {item}" for item in risks)

    vector_evidence = _filter_evidence_groups(
        compact.get("vector_evidence") or {},
        include_keywords=_PRECISION_KEYWORDS,
        exclude_keywords=_DATATYPE_EXCLUDE_KEYWORDS,
    )
    vector_lines = _format_evidence_group("Precision code knowledge", vector_evidence)
    if vector_lines:
        lines.extend(vector_lines)

    refs = compact.get("evidence_refs") or []
    if refs:
        lines.append(f"- Evidence refs: {', '.join(refs[:8])}")
    lines.append(f"- Confidence: {compact.get('confidence', 0.0)}")
    lines.append(
        "- Stage boundary: Choose only tensor datatype and precision policy here: DEVICE, USE_AMP, AMP_DTYPE, "
        "USE_TF32, GradScaler, autocast helper, fallback behavior, and precision logging."
    )
    lines.append(
        "- Out of scope for this stage: model architecture, loss, features, batch size, epochs, learning rate, "
        "dataloader workers, gradient accumulation, checkpoint cadence, validation metric, and submission logic."
    )
    lines.append(f"- Rule: {EVIDENCE_NOT_LAW_RULE}")
    lines.append(f"- Constraint rule: {CONSTRAINT_PRECEDENCE_RULE}")
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[: max(0, max_chars - 48)].rstrip() + "\n... [datatype context truncated]"
    return text + "\n"


def format_hardware_training_prompt_section(compact: dict[str, Any], *, max_chars: int = 3500) -> str:
    if not compact:
        return ""

    lines = [HARDWARE_TRAINING_HEADING]
    hardware = compact.get("hardware_context") or {}
    _append_hardware_summary(lines, hardware)

    diagnosis = _filter_diagnosis(compact.get("derived_diagnosis") or {}, _TRAINING_HPARAM_KEYWORDS)
    symptoms = diagnosis.get("profile_symptoms") or []
    targets = diagnosis.get("optimization_targets") or []
    if symptoms or targets:
        lines.append(f"- Training diagnosis: symptoms={symptoms or ['none']} targets={targets or ['none']}")

    recommendations = _filter_string_list(
        compact.get("recommendations") or [],
        include_keywords=_TRAINING_HPARAM_KEYWORDS,
        exclude_keywords=(),
        limit=8,
    )
    if recommendations:
        lines.append("- Training hyperparameter recommendations:")
        lines.extend(f"  - {item}" for item in recommendations)

    risks = _filter_string_list(
        compact.get("risk_flags") or [],
        include_keywords=_TRAINING_HPARAM_KEYWORDS,
        exclude_keywords=(),
        limit=8,
    )
    if risks:
        lines.append("- Training risk flags:")
        lines.extend(f"  - {item}" for item in risks)

    graph_evidence = compact.get("graph_evidence") or {}
    graph_lines = _format_evidence_group("Training graph evidence", graph_evidence)
    if graph_lines:
        lines.extend(graph_lines)

    vector_evidence = _filter_evidence_groups(
        compact.get("vector_evidence") or {},
        include_keywords=_TRAINING_HPARAM_KEYWORDS,
        exclude_keywords=(),
    )
    vector_lines = _format_evidence_group("Training code knowledge", vector_evidence)
    if vector_lines:
        lines.extend(vector_lines)

    refs = compact.get("evidence_refs") or []
    if refs:
        lines.append(f"- Evidence refs: {', '.join(refs[:8])}")
    lines.append(f"- Confidence: {compact.get('confidence', 0.0)}")
    lines.append(
        "- Stage boundary: Optimize training hyperparameters here: physical batch size, effective batch size, "
        "gradient accumulation, epochs, learning rate, weight decay, scheduler, dataloader workers, checkpointing, "
        "runtime logging, validation, and submission."
    )
    lines.append(
        "- Precision boundary: consume the datatype_precision variables/utilities. Do not pick a new AMP dtype here "
        "unless the prior policy is impossible to use and a safe fallback is needed."
    )
    lines.append(f"- Rule: {EVIDENCE_NOT_LAW_RULE}")
    lines.append(f"- Constraint rule: {CONSTRAINT_PRECEDENCE_RULE}")
    lines.append(f"- Budget guardrail: {HARDWARE_BUDGET_GUARDRAIL_RULE}")
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[: max(0, max_chars - 48)].rstrip() + "\n... [training context truncated]"
    return text + "\n"


def infer_workload_type(task_desc: Any, data_preview: str | None = None) -> str:
    text = f"{task_desc or ''} {data_preview or ''}".lower()
    if any(token in text for token in ("image", "jpg", "jpeg", "png", "dicom", "vision", "segmentation")):
        return "vision_training"
    if any(token in text for token in ("text", "nlp", "token", "language", "transformer")):
        return "transformer_training"
    if any(token in text for token in ("audio", "wav", "spectrogram", "sound")):
        return "audio_training"
    if any(token in text for token in ("tabular", "csv", "categorical", "xgboost", "lightgbm")):
        return "tabular_training"
    return "mlevolve_training"


def _hardware_context_enabled(agent: Any) -> bool:
    cfg = getattr(agent, "cfg", None)
    experiment = getattr(cfg, "experiment", None)
    mode = str(getattr(experiment, "mode", "") or "").strip().lower().replace("-", "_")
    if mode in {"origin", "baseline"}:
        return False
    acfg = getattr(agent, "acfg", None)
    return bool(getattr(acfg, "hardware_context_enabled", True))


def _safe_node_term_out(node: Any | None) -> str:
    if node is None:
        return ""
    try:
        return str(getattr(node, "term_out", "") or "")
    except Exception as exc:
        logger.debug("Skipping parent execution output in hardware context: %s", exc)
        return ""


def _scheduler_submission_defaults(scheduler_client: Any | None) -> Any | None:
    if scheduler_client is None:
        return None
    settings = getattr(scheduler_client, "settings", None)
    gpu_scheduler = getattr(settings, "gpu_scheduler", None)
    return getattr(gpu_scheduler, "submission_defaults", None)


def _scheduler_backend_config(scheduler_client: Any | None) -> dict[str, Any]:
    if scheduler_client is None:
        return {}
    settings = getattr(scheduler_client, "settings", None)
    gpu_scheduler = getattr(settings, "gpu_scheduler", None)
    if gpu_scheduler is None:
        return {}
    probe_payload = _latest_auto_backend_probe_payload(scheduler_client)
    mode = getattr(gpu_scheduler, "mode", None)
    if probe_payload:
        mode = probe_payload.get("configured_mode", mode)
    try:
        from localml_scheduler.config import effective_scheduler_mode

        effective_mode = probe_payload.get("effective_scheduler_mode") if probe_payload else effective_scheduler_mode(mode)
    except Exception:
        effective_mode = mode
    return {
        "mode": mode,
        "effective_mode": effective_mode,
        "backend_priority": _probe_payload_list_or_setting(probe_payload, "backend_priority", gpu_scheduler),
        "concurrent_backend_allowlist": _probe_payload_list_or_setting(
            probe_payload,
            "concurrent_backend_allowlist",
            gpu_scheduler,
        ),
    }


def _probe_payload_list_or_setting(probe_payload: dict[str, Any], key: str, settings: Any) -> list[Any]:
    if probe_payload and key in probe_payload:
        return list(probe_payload.get(key) or [])
    return list(getattr(settings, key, []) or [])


def _latest_auto_backend_probe_payload(scheduler_client: Any | None) -> dict[str, Any]:
    if scheduler_client is None:
        return {}
    events: list[dict[str, Any]] = []
    list_events = getattr(scheduler_client, "list_events", None)
    if callable(list_events):
        try:
            events = list(list_events(event_type="scheduler_auto_backend_probe"))
        except Exception:
            events = []
    if not events:
        store = getattr(scheduler_client, "store", None)
        store_list_events = getattr(store, "list_events", None)
        if callable(store_list_events):
            try:
                events = list(store_list_events(event_type="scheduler_auto_backend_probe"))
            except Exception:
                events = []
    if not events:
        return {}
    payload = events[-1].get("payload") or {}
    return dict(payload) if isinstance(payload, dict) else {}


def _compact_hardware_context(context: dict[str, Any]) -> dict[str, Any]:
    hardware = dict(context.get("hardware") or {})
    toolkit = dict(context.get("toolkit") or {})
    backend = dict(context.get("backend_capabilities") or {})
    limits = dict(context.get("scheduler_limits") or {})
    summary = hardware.get("summary_text") or hardware.get("gpu_name") or hardware.get("hardware_key")
    compact = {
        "found": bool(context.get("found")),
        "summary": _short(summary, 220),
        "hardware": _pick(
            hardware,
            ("hardware_key", "gpu_name", "total_vram_mb", "compute_capability", "toolkit_name", "toolkit_version", "torch_version"),
        ),
        "toolkit": _pick(toolkit, ("toolkit_name", "toolkit_version", "torch_version")),
        "backend_capabilities": _pick(
            backend,
            (
                "mode",
                "effective_mode",
                "backend_priority",
                "enabled_backends",
                "stream_mps_available",
                "stream_available",
                "mps_available",
                "cuda_process_available",
                "concurrent_groups_enabled",
                "concurrent_backend_allowlist",
            ),
        ),
        "scheduler_limits": _pick(
            limits,
            ("safe_vram_budget_mb", "max_packed_jobs_per_gpu", "mode", "effective_mode", "backend_priority", "concurrent_backend_allowlist"),
        ),
    }
    return {key: value for key, value in compact.items() if value not in (None, {}, [], "")}


def _compact_graph_evidence(graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "exact_profiles": [_compact_profile(item) for item in list(graph.get("exact_profiles") or [])[:2]],
        "similar_profiles": [_compact_profile(item) for item in list(graph.get("similar_profiles") or [])[:2]],
        "packed_profiles": [_compact_profile(item) for item in list(graph.get("packed_profiles") or [])[:2]],
    }


def _compact_profile(item: dict[str, Any]) -> dict[str, Any]:
    data = dict(item.get("data") or item)
    compact = _pick(
        data,
        (
            "summary_text",
            "match_reason",
            "status",
            "purpose",
            "model_key",
            "model_family",
            "batch_size",
            "resolved_batch_size",
            "epochs",
            "estimated_total_runtime_seconds",
            "runtime_seconds",
            "peak_vram_mb",
            "peak_vram_mib",
            "avg_sm_utilization_pct",
            "throughput_samples_per_second",
            "precision",
            "precision_mode",
            "uses_amp",
            "backend_name",
            "failure_reason",
        ),
    )
    if item.get("ref"):
        compact["ref"] = item.get("ref")
    if item.get("match_reason"):
        compact["match_reason"] = item.get("match_reason")
    return {key: _short(value, 220) if isinstance(value, str) else value for key, value in compact.items() if value not in (None, "", [], {})}


def _compact_diagnosis(diagnosis: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "profile_symptoms": _clean_string_list(diagnosis.get("profile_symptoms") or [], limit=8),
        "optimization_targets": _clean_string_list(diagnosis.get("optimization_targets") or [], limit=8),
    }


def _filter_stage_direct_values(stage_name: str, list_key: str, values: Any) -> list[str]:
    cleaned = _clean_string_list(values or [], limit=32)
    if not cleaned or list_key in {"recommended_patterns", "avoid_patterns"}:
        return cleaned
    keywords = _STAGE_DIRECT_VALUE_KEYWORDS.get(stage_name)
    if not keywords:
        return cleaned
    filtered: list[str] = []
    for value in cleaned:
        text = value.lower().replace("-", "_").replace(" ", "_")
        if any(keyword.replace("-", "_").replace(" ", "_") in text for keyword in keywords):
            filtered.append(value)
    return filtered


def _compact_stage_hardware_features(stage_context: dict[str, Any]) -> dict[str, Any]:
    if not stage_context:
        return {}
    stages: list[dict[str, Any]] = []
    for item in list(stage_context.get("stages") or [])[:4]:
        node = dict(item.get("node") or {})
        stage_name = str(item.get("stage") or node.get("stage_filter") or "").strip()
        field_limits = _STAGE_NODE_FIELD_LIMITS.get(stage_name, _DEFAULT_STAGE_NODE_FIELD_LIMITS)
        compact_node = {"stage_filter": node.get("stage_filter")} if node.get("stage_filter") else {}
        for list_key, item_limit in field_limits:
            if list_key in node:
                compact_node[list_key] = _clean_string_list(
                    _filter_stage_direct_values(stage_name, list_key, node.get(list_key) or []),
                    limit=item_limit,
                )

        features: list[dict[str, Any]] = []
        omitted_not_recommended: list[str] = []
        for raw_feature in list(item.get("features") or [])[:8]:
            feature = _compact_stage_hardware_feature(feature=raw_feature)
            if not feature:
                continue
            feature_id = str(feature.get("feature_id") or feature.get("name") or "feature").strip()
            if feature.get("recommended") is False:
                if feature_id and feature_id not in omitted_not_recommended:
                    omitted_not_recommended.append(feature_id)
                continue
            features.append(feature)
        compact_stage = {
            "stage": stage_name or compact_node.get("stage_filter"),
            "node": {key: value for key, value in compact_node.items() if value not in (None, "", [], {})},
            "features": features[:4],
            "feature_count": item.get("feature_count"),
            "shown_feature_count": len(features[:4]),
            "omitted_not_recommended": omitted_not_recommended[:6],
        }
        stages.append({key: value for key, value in compact_stage.items() if value not in (None, "", [], {})})
    feature_ids = _clean_string_list(
        [feature.get("feature_id") for feature in stage_context.get("features") or [] if feature.get("feature_id")],
        limit=24,
    )
    compact = {
        "found": bool(stage_context.get("found")),
        "stage_filter": stage_context.get("stage_filter"),
        "hardware": _pick(dict(stage_context.get("hardware") or {}), ("node_id", "gpu_name", "architecture", "vram_MB", "compute_capability")),
        "stages": stages,
        "feature_ids": feature_ids,
        "feature_count": stage_context.get("feature_count"),
        "source": stage_context.get("source"),
        "reason": stage_context.get("reason"),
    }
    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


def _compact_stage_hardware_feature(feature: dict[str, Any]) -> dict[str, Any]:
    compact = _pick(
        dict(feature),
        (
            "feature_id",
            "name",
            "feature_name",
            "category",
            "support_level",
            "recommended",
            "verified",
            "performance_impact",
            "recommendation_scope",
            "limitations",
            "notes",
            "usage",
            "recommended_patterns",
            "avoid_patterns",
            "example_code",
        ),
    )
    feature_id = str(compact.get("feature_id") or feature.get("feature_id") or "").lower()
    is_optimizer = compact.get("category") == "optimizer" or any(
        token in feature_id for token in ("optimizer", "muon", "adamw", "soap", "ademamix")
    )
    shortened: dict[str, Any] = {}
    for key, value in compact.items():
        if value in (None, "", [], {}):
            continue
        if isinstance(value, str):
            if key == "example_code":
                limit = 520
            elif is_optimizer and key in {"usage", "limitations", "notes"}:
                limit = 260
            else:
                limit = 220
            shortened[key] = _short(value, limit)
        elif isinstance(value, list):
            item_limit = 220 if is_optimizer and key in {"recommended_patterns", "avoid_patterns"} else 180
            item_count = 4 if is_optimizer and key in {"recommended_patterns", "avoid_patterns"} else 3
            shortened[key] = [_short(entry, item_limit) if isinstance(entry, str) else entry for entry in value[:item_count]]
        else:
            shortened[key] = value
    return shortened


def _compact_vector_evidence(vector: dict[str, Any]) -> dict[str, Any]:
    return {
        "recipes": [_compact_code_knowledge(item) for item in list(vector.get("recipes") or [])[:2]],
        "docs": [_compact_code_knowledge(item) for item in list(vector.get("docs") or [])[:2]],
        "api_symbols": [_compact_code_knowledge(item) for item in list(vector.get("api_symbols") or [])[:2]],
    }


def _compact_hardware_feature_index_item(item: dict[str, Any]) -> dict[str, Any]:
    compact = _pick(
        dict(item),
        (
            "feature_id",
            "feature_name",
            "category",
            "support_level",
            "recommended",
            "performance_impact",
            "pipeline_stage",
            "frameworks",
            "tags",
            "confidence",
        ),
    )
    return {
        key: _short(value, 120) if isinstance(value, str) else value
        for key, value in compact.items()
        if value not in (None, "", [], {})
    }


def _compact_hardware_feature_detail(item: dict[str, Any]) -> dict[str, Any]:
    compact = _pick(
        dict(item),
        (
            "feature_id",
            "feature_name",
            "title",
            "category",
            "support_level",
            "recommended",
            "performance_impact",
            "summary_text",
            "detail_text",
            "software_requirements",
            "recommended_patterns",
            "avoid_patterns",
            "sample_code",
            "confidence",
            "evidence_ref",
        ),
    )
    shortened: dict[str, Any] = {}
    for key, value in compact.items():
        if value in (None, "", [], {}):
            continue
        if isinstance(value, str):
            shortened[key] = _short(value, 260 if key != "sample_code" else 400)
        elif isinstance(value, list):
            shortened[key] = [_short(item, 220) if isinstance(item, str) else item for item in value[:4]]
        else:
            shortened[key] = value
    return shortened


def _compact_code_knowledge(item: dict[str, Any]) -> dict[str, Any]:
    compact = _pick(
        item,
        (
            "record_id",
            "record_type",
            "title",
            "summary_text",
            "api_symbol",
            "api_symbols",
            "recommended_patterns",
            "avoid_patterns",
            "confidence",
        ),
    )
    for key in ("summary_text", "title"):
        if key in compact:
            compact[key] = _short(compact[key], 220)
    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


def _format_stage_hardware_features(stage_context: dict[str, Any]) -> list[str]:
    if not stage_context or not stage_context.get("found"):
        return []
    lines = ["- Stage-filtered hardware knowledge:"]
    stages = list(stage_context.get("stages") or [])
    if not stages and stage_context.get("feature_ids"):
        lines.append(
            f"  - stage={stage_context.get('stage_filter')}: feature_ids={stage_context.get('feature_ids')}"
        )
        return lines
    for stage in stages[:4]:
        stage_name = stage.get("stage") or "pipeline"
        is_optimizer_stage = str(stage_name).strip().lower() == "optimizer"
        node = stage.get("node") or {}
        direct_bits = _format_kv(
            node,
            (
                "datatypes",
                "software_features",
                "recipes",
                "recommended_patterns",
                "avoid_patterns",
            ),
        )
        suffix = f"; {direct_bits}" if direct_bits else ""
        lines.append(f"  - {stage_name}{suffix}")
        audit_bits: list[str] = []
        feature_count = stage.get("feature_count")
        shown_count = stage.get("shown_feature_count")
        if feature_count is not None:
            audit_bits.append(f"features_shown={shown_count or 0}/{feature_count}")
        elif shown_count is not None:
            audit_bits.append(f"features_shown={shown_count}")
        omitted = stage.get("omitted_not_recommended") or []
        if omitted:
            audit_bits.append(f"omitted_not_recommended={omitted}")
        if audit_bits:
            lines.append(f"    - filter_audit: {', '.join(audit_bits)}")
        for feature in list(stage.get("features") or [])[:4]:
            feature_id = feature.get("feature_id") or "feature"
            name = feature.get("name") or feature.get("feature_name") or ""
            details = _format_kv(
                feature,
                (
                    "category",
                    "support_level",
                    "recommended",
                    "verified",
                    "performance_impact",
                    "recommendation_scope",
                ),
            )
            detail_text = f" ({details})" if details else ""
            summary_keys = (
                ("usage", "notes", "limitations")
                if is_optimizer_stage
                else ("limitations", "usage", "notes")
            )
            summary_parts: list[str] = []
            for key in summary_keys:
                summary = feature.get(key)
                if summary:
                    text = _short(summary, 220 if is_optimizer_stage else 160)
                    if text not in summary_parts:
                        summary_parts.append(f"{key}: {text}")
            summary_text = f": {'; '.join(summary_parts[:3])}" if summary_parts else ""
            lines.append(f"    - {feature_id}: {name}{detail_text}{summary_text}")
            if is_optimizer_stage:
                for pattern in list(feature.get("recommended_patterns") or [])[:3]:
                    lines.append(f"      recommended: {_short(pattern, 220)}")
                for pattern in list(feature.get("avoid_patterns") or [])[:2]:
                    lines.append(f"      avoid: {_short(pattern, 220)}")
                if feature_id == "muon_optimizer" and feature.get("example_code"):
                    lines.append(f"      example: {_short(feature['example_code'], 360)}")
    source = stage_context.get("source")
    feature_count = stage_context.get("feature_count")
    if source or feature_count is not None:
        lines.append(f"  - source={source or 'unknown'}; linked_features={feature_count}")
    return lines


def _format_evidence_group(label: str, groups: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for group_name, entries in groups.items():
        entries = list(entries or [])
        if not entries:
            continue
        if not lines:
            lines.append(f"- {label}:")
        for entry in entries[:2]:
            summary = entry.get("summary_text") or entry.get("title") or entry.get("record_id") or entry.get("ref") or str(entry)
            details = _format_kv(entry, ("match_reason", "batch_size", "resolved_batch_size", "peak_vram_mb", "avg_sm_utilization_pct", "confidence"))
            suffix = f" ({details})" if details else ""
            lines.append(f"  - {group_name}: {_short(summary, 180)}{suffix}")
    return lines


def _append_hardware_summary(lines: list[str], hardware: dict[str, Any]) -> None:
    if not hardware:
        return
    lines.append(f"- Hardware: {hardware.get('summary') or 'current hardware'}")
    backend = hardware.get("backend_capabilities") or {}
    if backend:
        backend_bits = _format_kv(
            backend,
            (
                "mode",
                "effective_mode",
                "backend_priority",
                "enabled_backends",
                "concurrent_groups_enabled",
                "concurrent_backend_allowlist",
            ),
        )
        if backend_bits:
            lines.append(f"- Scheduler backend config: {backend_bits}")
    limits = hardware.get("scheduler_limits") or {}
    if limits:
        limit_bits = _format_kv(
            limits,
            ("safe_vram_budget_mb", "max_packed_jobs_per_gpu", "mode", "effective_mode", "backend_priority"),
        )
        if limit_bits:
            lines.append(f"- Scheduler limits: {limit_bits}")


def _filter_diagnosis(diagnosis: dict[str, Any], include_keywords: tuple[str, ...]) -> dict[str, list[str]]:
    return {
        "profile_symptoms": _filter_string_list(
            diagnosis.get("profile_symptoms") or [],
            include_keywords=include_keywords,
            exclude_keywords=(),
            limit=8,
        ),
        "optimization_targets": _filter_string_list(
            diagnosis.get("optimization_targets") or [],
            include_keywords=include_keywords,
            exclude_keywords=(),
            limit=8,
        ),
    }


def _filter_evidence_groups(
    groups: dict[str, Any],
    *,
    include_keywords: tuple[str, ...],
    exclude_keywords: tuple[str, ...],
) -> dict[str, list[dict[str, Any]]]:
    filtered: dict[str, list[dict[str, Any]]] = {}
    for group_name, entries in groups.items():
        kept = []
        for entry in list(entries or []):
            text = _evidence_entry_text(entry)
            if _matches_keywords(text, include_keywords) and not _matches_keywords(text, exclude_keywords):
                kept.append(entry)
        if kept:
            filtered[group_name] = kept
    return filtered


def _filter_string_list(
    values: Any,
    *,
    include_keywords: tuple[str, ...],
    exclude_keywords: tuple[str, ...],
    limit: int,
) -> list[str]:
    result: list[str] = []
    for value in list(values or []):
        text = str(value)
        if not _matches_keywords(text, include_keywords):
            continue
        if _matches_keywords(text, exclude_keywords):
            continue
        short = _short(text, 240)
        if short and short not in result:
            result.append(short)
        if len(result) >= limit:
            break
    return result


def _matches_keywords(text: Any, keywords: tuple[str, ...]) -> bool:
    if not keywords:
        return False
    lowered = str(text or "").lower()
    return any(keyword in lowered for keyword in keywords)


def _evidence_entry_text(entry: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in dict(entry or {}).items():
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        else:
            parts.append(str(value))
    return " ".join(parts)


def _store_hardware_decision(node: Any, decision: dict[str, Any]) -> None:
    chain = _hardware_decision_chain(getattr(node, "hardware_decision", None))
    chain.append(dict(decision))
    _set_hardware_decision_pipeline(node, chain)


def _set_hardware_decision_pipeline(node: Any, pipeline: list[dict[str, Any]]) -> None:
    if not pipeline:
        return
    sanitized = [_latest_decision_without_pipeline(item) for item in pipeline]
    latest = dict(sanitized[-1])
    latest["pipeline"] = sanitized
    latest["latest_stage"] = latest.get("stage")
    node.hardware_decision = latest


def _hardware_decision_chain(decision: Any) -> list[dict[str, Any]]:
    if not isinstance(decision, dict):
        return []
    pipeline = decision.get("pipeline")
    if isinstance(pipeline, list):
        return [_latest_decision_without_pipeline(item) for item in pipeline if isinstance(item, dict)]
    return [_latest_decision_without_pipeline(decision)]


def _latest_decision_without_pipeline(decision: Any) -> dict[str, Any]:
    if not isinstance(decision, dict):
        return {}
    cleaned = dict(decision)
    cleaned.pop("pipeline", None)
    cleaned.pop("latest_stage", None)
    return cleaned


def _stage_decision_rationale(stage: str) -> str:
    if stage == "model_design":
        return "Stage 1 selected the model design while deferring datatype and training hyperparameters."
    if stage == "datatype_precision":
        return "Stage 2 selected the tensor datatype and precision policy before training hyperparameter tuning."
    if stage == "training_evaluation":
        return "Stage 3 selected training hyperparameters and evaluation/submission behavior after model and dtype decisions."
    return "Hardware-aware stepwise generation stage completed."


def _stage_chosen_params(stage: str, candidate: dict[str, Any]) -> dict[str, Any]:
    if stage == "model_design":
        keys = ("model_key", "model_family", "framework")
    elif stage == "datatype_precision":
        keys = ("precision_mode", "uses_amp", "framework")
    elif stage == "training_evaluation":
        keys = (
            "proposed_batch_size",
            "proposed_epochs",
            "learning_rate",
            "weight_decay",
            "gradient_accumulation_steps",
            "num_workers",
        )
    else:
        keys = ()
    return {key: candidate.get(key) for key in keys if candidate.get(key) not in (None, "", [], {})}


def _execution_resource_hints(term_out: str) -> dict[str, Any]:
    hints: dict[str, Any] = {}
    if not term_out:
        return hints
    batch = re.search(r"(?:resolved_)?batch_size[=:]\s*(\d+)", term_out, re.IGNORECASE)
    if batch:
        hints["proposed_batch_size"] = _safe_int(batch.group(1), default=None)
    peak = re.search(r"(?:peak[_ ]?vram|vram)[^\d]{0,16}(\d+(?:\.\d+)?)\s*(?:mb|mib)", term_out, re.IGNORECASE)
    if peak:
        hints["notes"] = f"parent_peak_vram_mb={peak.group(1)}"
    return {key: value for key, value in hints.items() if value is not None}


def _recommended_batch_size(compact: dict[str, Any]) -> int | None:
    for item in _all_graph_profiles(compact):
        for key in ("resolved_batch_size", "batch_size"):
            value = _safe_int(item.get(key), default=None)
            if value is not None:
                return value
    for rec in compact.get("recommendations") or []:
        match = re.search(r"batch size\s+(\d+)", str(rec), re.IGNORECASE)
        if match:
            return _safe_int(match.group(1), default=None)
    return None


def _recommended_epochs(compact: dict[str, Any]) -> int | None:
    graph = compact.get("graph_evidence") or {}
    legacy = graph.get("legacy_job_design_context") or {}
    epoch_recommendation = legacy.get("epoch_recommendation") or {}
    value = _safe_int(epoch_recommendation.get("recommended_epochs"), default=None)
    if value is not None:
        return value
    for rec in compact.get("recommendations") or []:
        match = re.search(r"(?:epoch budget|epochs?)\s+(\d+)", str(rec), re.IGNORECASE)
        if match:
            return _safe_int(match.group(1), default=None)
    return None


def _rewrite_training_params(code: str, chosen_params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    updated = code
    applied: dict[str, Any] = {}
    batch_size = _safe_int(chosen_params.get("batch_size"), default=None)
    if batch_size is not None and batch_size > 0:
        updated, changed = _replace_simple_int_assignment(
            updated,
            ("BATCH_SIZE", "batch_size", "train_batch_size"),
            batch_size,
        )
        if changed:
            applied["batch_size"] = batch_size
    epochs = _safe_int(chosen_params.get("epochs"), default=None)
    if epochs is not None and epochs > 0:
        updated, changed = _replace_simple_int_assignment(
            updated,
            ("EPOCHS", "epochs", "num_epochs", "N_EPOCHS"),
            epochs,
        )
        if changed:
            applied["epochs"] = epochs
    return updated, applied


def _replace_simple_int_assignment(code: str, names: tuple[str, ...], value: int) -> tuple[str, bool]:
    for name in names:
        pattern = re.compile(rf"(?m)^(\s*{re.escape(name)}\s*=\s*)(\d+)(\s*(?:#.*)?$)")
        new_code, count = pattern.subn(rf"\g<1>{int(value)}\g<3>", code, count=1)
        if count:
            return new_code, True
    return code, False


def _default_model_families_for_workload(workload_type: str | None) -> list[str]:
    workload = str(workload_type or "").lower()
    if "vision" in workload:
        return ["convnet", "efficientnet", "convnext", "vision_transformer", "hybrid_cnn_transformer"]
    if "transformer" in workload or "text" in workload or "nlp" in workload:
        return ["transformer", "small_transformer", "lora_transformer", "sequence_cnn"]
    if "audio" in workload:
        return ["cnn", "conformer", "spectrogram_transformer"]
    if "tabular" in workload:
        return ["lightgbm", "xgboost", "tabular_mlp", "tab_transformer"]
    return ["baseline_compatible", "cnn", "transformer", "tree_ensemble"]


def _runtime_seconds(compact: dict[str, Any]) -> float | None:
    for item in _all_graph_profiles(compact):
        for key in ("estimated_total_runtime_seconds", "runtime_seconds"):
            value = _safe_float(item.get(key))
            if value is not None:
                return value
    return None


def _peak_vram_mb(compact: dict[str, Any]) -> float | None:
    for item in _all_graph_profiles(compact):
        for key in ("peak_vram_mb", "peak_vram_mib"):
            value = _safe_float(item.get(key))
            if value is not None:
                return value
    return None


def _backend_name(compact: dict[str, Any]) -> str | None:
    for item in _all_graph_profiles(compact):
        if item.get("backend_name"):
            return str(item["backend_name"])
    hardware = compact.get("hardware_context") or {}
    backend = hardware.get("backend_capabilities") or {}
    priority = backend.get("backend_priority") or []
    if priority:
        return str(priority[0])
    return None


def _all_graph_profiles(compact: dict[str, Any]) -> list[dict[str, Any]]:
    graph = compact.get("graph_evidence") or {}
    profiles: list[dict[str, Any]] = []
    for key in ("exact_profiles", "similar_profiles", "packed_profiles"):
        profiles.extend(list(graph.get(key) or []))
    return profiles


def _pick(payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: payload.get(key) for key in keys if payload.get(key) not in (None, "", [], {})}


def _format_kv(payload: dict[str, Any], keys: tuple[str, ...]) -> str:
    parts = []
    for key in keys:
        value = payload.get(key)
        if value not in (None, "", [], {}):
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _clean_string_list(values: Any, *, limit: int) -> list[str]:
    result = []
    for value in list(values or [])[:limit]:
        text = _short(str(value), 240)
        if text and text not in result:
            result.append(text)
    return result


def _short(value: Any, limit: int) -> str:
    text = str(value or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _safe_int(value: Any, default: int | None = 0) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

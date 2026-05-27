"""Hardware/profile context helpers for MLEvolve stage prompts."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from engine.script_introspection import introspect_training_script

logger = logging.getLogger("MLEvolve")

HARDWARE_CONTEXT_HEADING = "# Hardware/Profile Optimization Context"
HARDWARE_DESIGN_HEADING = "# Hardware-Aware Model Design Brief"
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
    prompt_section = format_hardware_prompt_section(compact, max_chars=max_chars)
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
    node.hardware_decision = {
        "stage": "model_design",
        "rationale": "Hardware-aware model design brief was provided before draft generation.",
        "chosen_params": {},
        "original_params": {},
        "model_options": compact.get("model_options") or [],
        "recommendations": compact.get("recommendations") or [],
        "evidence_refs": list(compact.get("evidence_refs") or []),
        "confidence": float(compact.get("confidence") or 0.0),
        "fallback_reason": None if compact.get("model_options") else "no model-family hardware evidence found",
    }


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
            decision["previous_decision"] = previous_decision
        node.hardware_decision = decision
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
        "vector_evidence": _compact_vector_evidence(raw_context.get("vector_evidence") or {}),
        "recommendations": _clean_string_list(raw_context.get("recommendations") or [], limit=8),
        "risk_flags": _clean_string_list(raw_context.get("risk_flags") or [], limit=8),
        "evidence_refs": _clean_string_list(raw_context.get("evidence_refs") or [], limit=16),
        "confidence": round(float(raw_context.get("confidence") or 0.0), 3),
    }
    return compact


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
    compact = {
        "hardware_context": _compact_hardware_context(raw_context.get("hardware_context") or {}),
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


def _compact_vector_evidence(vector: dict[str, Any]) -> dict[str, Any]:
    return {
        "recipes": [_compact_code_knowledge(item) for item in list(vector.get("recipes") or [])[:2]],
        "docs": [_compact_code_knowledge(item) for item in list(vector.get("docs") or [])[:2]],
        "api_symbols": [_compact_code_knowledge(item) for item in list(vector.get("api_symbols") or [])[:2]],
    }


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

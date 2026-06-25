"""Pipeline-stage decision contract for prompt generation.

The decision contract stores datatype/model/optimizer/tuning choices, while
hardware-aware stepwise generation follows the SVG workflow:
model_design -> datatype_precision -> training_evaluation. Hardware/profile
evidence can inform tuning, but missing evidence must be recorded as a fallback
instead of invented claims.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from llm import generate

logger = logging.getLogger("MLEvolve")

PIPELINE_DECISION_HEADING = "# Pipeline Decision Contract"
PIPELINE_DECISION_TRACE_HEADING = "# Pipeline Decision Trace"
PIPELINE_STAGE_ORDER = ("model_design", "datatype_precision", "training_evaluation")
BATCH_SIZE_POLICIES = ("fixed", "scheduler_recommended", "adaptive")
PRECISION_POLICIES = ("fp32", "tf32", "fp16_amp", "bf16_amp", "fp8_te", "mxfp8_te", "nvfp4_te", "disabled")

PIPELINE_DECISION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "datatype": {
            "type": "object",
            "properties": {
                "modality": {
                    "type": "string",
                    "enum": ["image", "tabular", "text", "audio", "graph", "time_series", "mixed", "unknown"],
                },
                "target_type": {
                    "type": "string",
                    "enum": [
                        "classification",
                        "regression",
                        "segmentation",
                        "reconstruction",
                        "ranking",
                        "sequence",
                        "unknown",
                    ],
                },
                "shape_constraints": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
            },
            "required": ["modality", "target_type", "shape_constraints", "reason"],
            "additionalProperties": False,
        },
        "model": {
            "type": "object",
            "properties": {
                "family": {"type": "string"},
                "alternatives_considered": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
                "hardware_fit": {"type": "string"},
            },
            "required": ["family", "alternatives_considered", "reason", "hardware_fit"],
            "additionalProperties": False,
        },
        "optimizer": {
            "type": "object",
            "properties": {
                "loss": {"type": "string"},
                "optimizer": {"type": "string"},
                "scheduler": {"type": "string"},
                "reason": {"type": "string"},
                "advanced_optimizer_used": {"type": "boolean"},
            },
            "required": ["loss", "optimizer", "scheduler", "reason", "advanced_optimizer_used"],
            "additionalProperties": False,
        },
        "tuning": {
            "type": "object",
            "properties": {
                "batch_size_policy": {"type": "string", "enum": list(BATCH_SIZE_POLICIES)},
                "precision_policy": {"type": "string", "enum": list(PRECISION_POLICIES)},
                "precision_model_adaptation": {"type": "string"},
                "dataloader_policy": {"type": "string"},
                "fallbacks": {"type": "array", "items": {"type": "string"}},
                "metrics_to_log": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "batch_size_policy",
                "precision_policy",
                "precision_model_adaptation",
                "dataloader_policy",
                "fallbacks",
                "metrics_to_log",
            ],
            "additionalProperties": False,
        },
        "evidence": {
            "type": "object",
            "properties": {
                "hardware_context_used": {"type": "boolean"},
                "evidence_refs": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number"},
                "missing_evidence": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["hardware_context_used", "evidence_refs", "confidence", "missing_evidence"],
            "additionalProperties": False,
        },
    },
    "required": ["datatype", "model", "optimizer", "tuning", "evidence"],
    "additionalProperties": False,
}


def pipeline_decision_enabled(agent_instance: Any) -> bool:
    acfg = getattr(agent_instance, "acfg", None)
    return bool(getattr(acfg, "pipeline_decision_enabled", True))


def pipeline_decision_instructions(
    decision: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    """Prompt instructions shared by draft, improve, and planner stages."""
    if decision is not None and not decision:
        return {}
    return {
        "Pipeline Decision Contract": [
            "Use the Pipeline Decision Trace as the source of truth for code generation and planning.",
            "Follow the hardware-aware stepwise workflow: model_design -> datatype_precision -> training_evaluation.",
            "Do not jump directly to optimizer, precision, or batch-size choices before model family and output interface are decided.",
            "The stored datatype field describes data modality and target shape. Numeric precision such as fp32, fp16, bf16, tf32, or TE FP8/NVFP4 belongs under datatype_precision and tuning.precision_policy unless the task explicitly requires a numeric type.",
            "The datatype_precision step may make narrow precision-required model adapters such as Transformer Engine layer wrappers/replacements, padding/config hooks, autocast recipes, or higher-precision islands, but it must preserve the Stage 1 model family, loss, data features, and output interface.",
            "Hardware/profile evidence may influence model fit and tuning only when it is compatible with the task, installed packages, and available model sources.",
            "Hardware tuning must not increase epochs, folds, model size, image resolution, ensemble count, TTA, dataset size, or validation workload as a hardware-only optimization.",
            "If execution feedback contradicts an earlier decision, update only the relevant part of the decision and explain why in the plan.",
        ]
    }


def build_pipeline_decision(
    agent_instance: Any,
    *,
    stage: str,
    data_preview: str,
    hardware_contexts: list[Any] | None = None,
    parent_pipeline_decision: dict[str, Any] | None = None,
    previous_code: str = "",
    execution_output: str = "",
    stage_context: str = "",
    max_retries: int = 2,
) -> dict[str, Any]:
    """Generate and normalize the compact ordered pipeline decision."""
    if not pipeline_decision_enabled(agent_instance):
        return {}
    task_desc = getattr(agent_instance, "task_desc", "")
    evidence_state = _collect_evidence_state(hardware_contexts or [])
    fallback = _fallback_decision(
        task_desc=task_desc,
        data_preview=data_preview,
        evidence_state=evidence_state,
    )
    prompt = _build_decision_prompt(
        task_desc=task_desc,
        data_preview=data_preview,
        stage=stage,
        evidence_state=evidence_state,
        parent_pipeline_decision=parent_pipeline_decision,
        previous_code=previous_code,
        execution_output=execution_output,
        stage_context=stage_context,
    )

    for attempt in range(max(1, max_retries)):
        try:
            acfg = getattr(agent_instance, "acfg", None)
            code_cfg = getattr(acfg, "code", None)
            response = generate(
                prompt=prompt,
                temperature=getattr(code_cfg, "temp", 0.0),
                cfg=getattr(agent_instance, "cfg", None),
                json_schema=PIPELINE_DECISION_JSON_SCHEMA,
                max_retries=1,
                retry_delay=0,
            )
            decision = _parse_json_object(response)
            if decision:
                return normalize_pipeline_decision(decision, fallback=fallback, evidence_state=evidence_state)
        except Exception as exc:
            logger.debug("Pipeline decision generation failed on attempt %s: %s", attempt + 1, exc)

    fallback["evidence"]["missing_evidence"] = _unique_strings(
        list(fallback["evidence"].get("missing_evidence") or []) + ["pipeline decision LLM unavailable"]
    )
    return fallback


def format_pipeline_decision_prompt_section(decision: dict[str, Any] | None, *, max_chars: int = 3500) -> str:
    """Render the contract and current structured trace for prompts."""
    if not decision:
        return ""
    compact_decision = _compact_prompt_value(decision, string_limit=240, list_limit=8)
    payload = json.dumps(compact_decision, ensure_ascii=False, indent=2)
    text = _render_pipeline_decision_section(payload)
    if len(text) > max_chars:
        payload = json.dumps(compact_decision, ensure_ascii=False, separators=(",", ":"))
        text = _render_pipeline_decision_section(payload)
    if len(text) > max_chars:
        smaller_decision = _compact_prompt_value(decision, string_limit=100, list_limit=3)
        payload = json.dumps(smaller_decision, ensure_ascii=False, separators=(",", ":"))
        text = _render_compact_pipeline_decision_section(payload)
    return text


def _render_pipeline_decision_section(payload: str) -> str:
    return (
        f"{PIPELINE_DECISION_HEADING}\n"
        "Before writing or modifying code, follow the hardware-aware stepwise workflow: "
        "model_design -> datatype_precision -> training_evaluation.\n\n"
        "1. Model design: choose architecture/model family, loss, and output interface first.\n"
        "2. Datatype precision: choose dtype, AMP/TF32, GradScaler, TE FP8/MXFP8/NVFP4 recipes, autocast, precision-required model adapters, and precision fallback policy.\n"
        "3. Training evaluation: choose optimizer, scheduler, batch size, dataloader settings, checkpointing, validation, submission, runtime fallbacks, and logging.\n"
        "Datatype precision may include only precision-required model adapters that preserve the Stage 1 model family, loss, data features, and output interface.\n"
        "Hardware tuning must not increase epochs, folds, model size, image resolution, ensemble count, TTA, dataset size, or validation workload as a hardware-only optimization.\n"
        "If evidence is missing, use the fallback recorded in the trace instead of inventing hardware claims.\n\n"
        f"{PIPELINE_DECISION_TRACE_HEADING}\n"
        f"```json\n{payload}\n```\n"
    )


def _render_compact_pipeline_decision_section(payload: str) -> str:
    return (
        f"{PIPELINE_DECISION_HEADING}\n"
        "Required hardware-aware step order: model_design -> datatype_precision -> training_evaluation. "
        "Use the trace as the source of truth, apply hardware evidence only when compatible, "
        "and use recorded fallbacks when evidence is missing.\n\n"
        f"{PIPELINE_DECISION_TRACE_HEADING}\n"
        f"```json\n{payload}\n```\n"
    )


def apply_pipeline_decision_to_node(node: Any, decision: dict[str, Any] | None) -> None:
    if decision:
        node.pipeline_decision = decision


def normalize_pipeline_decision(
    decision: dict[str, Any] | None,
    *,
    fallback: dict[str, Any],
    evidence_state: dict[str, Any],
) -> dict[str, Any]:
    """Normalize ordering and evidence fields, stripping hallucinated refs."""
    decision = decision if isinstance(decision, dict) else {}
    normalized = {
        "datatype": _normalize_datatype(decision.get("datatype"), fallback["datatype"]),
        "model": _normalize_model(decision.get("model"), fallback["model"]),
        "optimizer": _normalize_optimizer(decision.get("optimizer"), fallback["optimizer"]),
        "tuning": _normalize_tuning(decision.get("tuning"), fallback["tuning"]),
        "evidence": _normalize_evidence(decision.get("evidence"), fallback["evidence"], evidence_state),
    }
    if not evidence_state["has_any_hardware_evidence"]:
        normalized["model"]["hardware_fit"] = "none"
        normalized["optimizer"]["advanced_optimizer_used"] = False
        normalized["tuning"]["precision_policy"] = "disabled"
        normalized["tuning"]["precision_model_adaptation"] = "none"
    if not evidence_state["has_predictor_or_graph_evidence"]:
        normalized["tuning"]["batch_size_policy"] = "fixed"
    return normalized


def _build_decision_prompt(
    *,
    task_desc: Any,
    data_preview: str,
    stage: str,
    evidence_state: dict[str, Any],
    parent_pipeline_decision: dict[str, Any] | None,
    previous_code: str,
    execution_output: str,
    stage_context: str,
) -> dict[str, str]:
    evidence_payload = {
        "available": evidence_state["has_any_hardware_evidence"],
        "has_predictor_or_graph_evidence": evidence_state["has_predictor_or_graph_evidence"],
        "evidence_refs": evidence_state["evidence_refs"],
        "confidence": evidence_state["confidence"],
        "compact_contexts": evidence_state["compact_contexts"],
        "missing_evidence": evidence_state["missing_evidence"],
    }
    system = (
        "You are the MLEvolve pipeline-decision agent. "
        "Return one compact JSON object and no prose. "
        "The required hardware-aware step order is model_design -> datatype_precision -> training_evaluation."
    )
    user = (
        f"Stage: {stage}\n\n"
        f"Task description:\n{task_desc}\n\n"
        f"Data preview:\n{data_preview}\n\n"
        "Pipeline contract:\n"
        "1. Model design: choose architecture/model family, loss, and output interface first.\n"
        "2. Datatype precision: choose dtype, AMP/TF32, GradScaler, TE FP8/MXFP8/NVFP4 recipes, autocast, precision-required model adapters, and precision fallback policy.\n"
        "3. Training evaluation: choose optimizer, scheduler, batch size, dataloader policy, checkpoint cadence, validation/submission behavior, and fallbacks.\n\n"
        "Rules:\n"
        "- Do not use hardware speed as a reason to violate task correctness, package availability, model-source availability, or submission format.\n"
        "- datatype_precision may only adapt model structure when the selected precision backend requires it, e.g. TE-compatible layer wrappers/replacements, precision shape padding/config hooks, autocast recipes, or higher-precision islands. It must not redesign model family, loss, data features, or task I/O.\n"
        "- Hardware tuning must not increase epochs, folds, model size, input resolution, ensemble count, TTA, dataset size, or validation workload as a hardware-only optimization.\n"
        "- If graph/predictor evidence is missing, record that in evidence.missing_evidence and use conservative tuning fallbacks.\n"
        "- evidence.evidence_refs must only contain refs listed in the hardware/profile evidence payload.\n\n"
        f"Hardware/profile evidence payload:\n{json.dumps(evidence_payload, ensure_ascii=False, indent=2, default=str)}\n\n"
    )
    if parent_pipeline_decision:
        user += (
            "Parent pipeline decision. Preserve unaffected decisions; update only contradicted parts:\n"
            f"{json.dumps(parent_pipeline_decision, ensure_ascii=False, indent=2, default=str)}\n\n"
        )
    if previous_code:
        user += f"Previous code excerpt:\n{_short(previous_code, 1800)}\n\n"
    if execution_output:
        user += f"Execution output excerpt:\n{_short(execution_output, 1200)}\n\n"
    if stage_context:
        user += f"Stage-specific context:\n{_short(stage_context, 2000)}\n\n"
    user += "Return JSON matching the schema exactly."
    return {"system": system, "user": user, "assistant": "I will return one complete JSON object only."}


def _fallback_decision(*, task_desc: Any, data_preview: str, evidence_state: dict[str, Any]) -> dict[str, Any]:
    text = f"{task_desc or ''}\n{data_preview or ''}"
    modality = _infer_modality(text)
    target_type = _infer_target_type(text)
    missing = list(evidence_state["missing_evidence"])
    if not evidence_state["has_any_hardware_evidence"]:
        missing.append("hardware/profile evidence not available")
    if not evidence_state["has_predictor_or_graph_evidence"]:
        missing.append("predictor/graph evidence not available")
    return {
        "datatype": {
            "modality": modality,
            "target_type": target_type,
            "shape_constraints": _infer_shape_constraints(text, modality),
            "reason": "Fallback decision inferred from the task description and data preview.",
        },
        "model": {
            "family": _fallback_model_family(modality, target_type),
            "alternatives_considered": [],
            "reason": "Use a baseline-compatible model family until stronger task or evidence signals are available.",
            "hardware_fit": "none" if not evidence_state["has_any_hardware_evidence"] else "use only compatible evidence",
        },
        "optimizer": {
            "loss": _fallback_loss(target_type),
            "optimizer": "AdamW for neural models or the library default for tree/linear baselines",
            "scheduler": "none",
            "reason": "Choose a stable default matched to the inferred target type and available model family.",
            "advanced_optimizer_used": False,
        },
        "tuning": {
            "batch_size_policy": "fixed" if not evidence_state["has_predictor_or_graph_evidence"] else "scheduler_recommended",
            "precision_policy": "disabled",
            "precision_model_adaptation": "none",
            "dataloader_policy": "safe defaults; enable workers and pinned memory only when compatible",
            "fallbacks": ["reduce physical batch size on OOM", "reduce epochs before changing model family on timeout"],
            "metrics_to_log": ["elapsed_seconds", "peak_vram_mb", "resolved_batch_size"],
        },
        "evidence": {
            "hardware_context_used": bool(evidence_state["has_any_hardware_evidence"]),
            "evidence_refs": list(evidence_state["evidence_refs"]),
            "confidence": _clamp_confidence(evidence_state["confidence"]),
            "missing_evidence": _unique_strings(missing),
        },
    }


def _collect_evidence_state(hardware_contexts: list[Any]) -> dict[str, Any]:
    compact_contexts: list[dict[str, Any]] = []
    evidence_refs: list[str] = []
    confidences: list[float] = []
    has_any = False
    has_predictor_or_graph = False

    for context in hardware_contexts:
        compact = getattr(context, "compact_context", context) or {}
        if not isinstance(compact, dict) or not compact:
            continue
        compact_contexts.append(compact)
        evidence_refs.extend(_string_list(compact.get("evidence_refs")))
        confidence = _safe_float(compact.get("confidence"))
        if confidence is not None:
            confidences.append(confidence)
        if _has_meaningful_hardware_evidence(compact):
            has_any = True
        if _has_predictor_graph_evidence(compact):
            has_predictor_or_graph = True

    missing = []
    if not has_any:
        missing.append("hardware/profile evidence not available")
    if not has_predictor_or_graph:
        missing.append("predictor/graph evidence not available")

    return {
        "compact_contexts": compact_contexts,
        "evidence_refs": _unique_strings(evidence_refs),
        "confidence": _clamp_confidence(max(confidences) if confidences else 0.0),
        "has_any_hardware_evidence": has_any,
        "has_predictor_or_graph_evidence": has_predictor_or_graph,
        "missing_evidence": _unique_strings(missing),
    }


def _has_meaningful_hardware_evidence(compact: dict[str, Any]) -> bool:
    hardware = compact.get("hardware_context") or {}
    if isinstance(hardware, dict):
        if hardware.get("found") is True or hardware.get("summary") or hardware.get("hardware"):
            return True
    if compact.get("model_options") or compact.get("selected_hardware_features"):
        return True
    if _string_list(compact.get("recommendations")) or _string_list(compact.get("risk_flags")):
        return True
    if _string_list(compact.get("evidence_refs")):
        return True
    return _has_predictor_graph_evidence(compact)


def _has_predictor_graph_evidence(compact: dict[str, Any]) -> bool:
    for key in ("predictor_evidence", "runtime_prediction", "runtime_predictions", "prediction_evidence"):
        if compact.get(key):
            return True
    graph = compact.get("graph_evidence") or {}
    if not isinstance(graph, dict):
        return False
    for values in graph.values():
        if values:
            return True
    return False


def _normalize_datatype(value: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    value = value if isinstance(value, dict) else {}
    modality = _choice(value.get("modality"), {"image", "tabular", "text", "audio", "graph", "time_series", "mixed", "unknown"}, fallback["modality"])
    target_type = _choice(
        value.get("target_type"),
        {"classification", "regression", "segmentation", "reconstruction", "ranking", "sequence", "unknown"},
        fallback["target_type"],
    )
    return {
        "modality": modality,
        "target_type": target_type,
        "shape_constraints": _string_list(value.get("shape_constraints")) or list(fallback["shape_constraints"]),
        "reason": _string(value.get("reason"), fallback["reason"]),
    }


def _normalize_model(value: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    value = value if isinstance(value, dict) else {}
    return {
        "family": _string(value.get("family"), fallback["family"]),
        "alternatives_considered": _string_list(value.get("alternatives_considered")) or list(fallback["alternatives_considered"]),
        "reason": _string(value.get("reason"), fallback["reason"]),
        "hardware_fit": _string(value.get("hardware_fit"), fallback["hardware_fit"]),
    }


def _normalize_optimizer(value: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    value = value if isinstance(value, dict) else {}
    return {
        "loss": _string(value.get("loss"), fallback["loss"]),
        "optimizer": _string(value.get("optimizer"), fallback["optimizer"]),
        "scheduler": _string(value.get("scheduler"), fallback["scheduler"]),
        "reason": _string(value.get("reason"), fallback["reason"]),
        "advanced_optimizer_used": _safe_bool(
            value.get("advanced_optimizer_used"),
            default=bool(fallback["advanced_optimizer_used"]),
        ),
    }


def _normalize_tuning(value: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    value = value if isinstance(value, dict) else {}
    return {
        "batch_size_policy": _choice(
            value.get("batch_size_policy"),
            set(BATCH_SIZE_POLICIES),
            fallback["batch_size_policy"],
        ),
        "precision_policy": _choice(
            value.get("precision_policy"),
            set(PRECISION_POLICIES),
            fallback["precision_policy"],
        ),
        "precision_model_adaptation": _string(
            value.get("precision_model_adaptation"),
            fallback.get("precision_model_adaptation", "none"),
        ),
        "dataloader_policy": _string(value.get("dataloader_policy"), fallback["dataloader_policy"]),
        "fallbacks": _string_list(value.get("fallbacks")) or list(fallback["fallbacks"]),
        "metrics_to_log": _string_list(value.get("metrics_to_log")) or list(fallback["metrics_to_log"]),
    }


def _normalize_evidence(value: Any, fallback: dict[str, Any], evidence_state: dict[str, Any]) -> dict[str, Any]:
    value = value if isinstance(value, dict) else {}
    actual_refs = list(evidence_state["evidence_refs"])
    missing = _unique_strings(
        list(fallback.get("missing_evidence") or [])
        + _string_list(value.get("missing_evidence"))
        + list(evidence_state["missing_evidence"])
    )
    if not evidence_state["has_any_hardware_evidence"]:
        missing.append("hardware/profile evidence not available")
    if not evidence_state["has_predictor_or_graph_evidence"]:
        missing.append("predictor/graph evidence not available")
    return {
        "hardware_context_used": bool(evidence_state["has_any_hardware_evidence"]),
        "evidence_refs": actual_refs,
        "confidence": _clamp_confidence(evidence_state["confidence"]),
        "missing_evidence": _unique_strings(missing),
    }


def _parse_json_object(response: Any) -> dict[str, Any] | None:
    if isinstance(response, dict):
        return response
    text = response if isinstance(response, str) else str(response)
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _infer_modality(text: str) -> str:
    lower = text.lower()
    if any(token in lower for token in ("image", "jpg", "jpeg", "png", "dicom", "mask", "segmentation", "pixel")):
        return "image"
    if any(token in lower for token in ("text", "token", "nlp", "language", "sentence", "transformer")):
        return "text"
    if any(token in lower for token in ("audio", "wav", "spectrogram", "sound")):
        return "audio"
    if any(token in lower for token in ("graph", "node", "edge")):
        return "graph"
    if any(token in lower for token in ("time series", "timeseries", "timestamp", "sequence")):
        return "time_series"
    if any(token in lower for token in ("csv", "tabular", "categorical", "numerical", "feature column")):
        return "tabular"
    return "unknown"


def _infer_target_type(text: str) -> str:
    lower = text.lower()
    if any(token in lower for token in ("segmentation", "mask")):
        return "segmentation"
    if any(token in lower for token in ("denois", "reconstruct", "autoencoder", "image output")):
        return "reconstruction"
    if any(token in lower for token in ("rank", "ranking")):
        return "ranking"
    if any(token in lower for token in ("sequence", "translate", "caption", "generated text")):
        return "sequence"
    if any(token in lower for token in ("regression", "rmse", "mae", "continuous", "value")):
        return "regression"
    if any(token in lower for token in ("class", "label", "accuracy", "auc", "logloss", "f1")):
        return "classification"
    return "unknown"


def _infer_shape_constraints(text: str, modality: str) -> list[str]:
    constraints = []
    lower = text.lower()
    if modality == "image":
        constraints.append("preserve train/validation/test image preprocessing compatibility")
        if "mask" in lower or "segmentation" in lower:
            constraints.append("model output must align spatially with mask dimensions")
    elif modality == "text":
        constraints.append("sequence length and tokenizer output must stay configurable")
    elif modality == "tabular":
        constraints.append("fit encoders/scalers on training data only")
    elif modality == "time_series":
        constraints.append("preserve temporal order in validation and feature generation")
    if not constraints:
        constraints.append("derive concrete input and target shapes from data preview before coding")
    return constraints


def _fallback_model_family(modality: str, target_type: str) -> str:
    if modality == "image" and target_type in {"segmentation", "reconstruction"}:
        return "lightweight_encoder_decoder"
    if modality == "image":
        return "cnn_or_available_pretrained_vision_model"
    if modality == "text":
        return "available_transformer_or_sequence_baseline"
    if modality == "tabular":
        return "tree_ensemble_or_tabular_baseline"
    if modality == "audio":
        return "spectrogram_cnn_or_audio_baseline"
    if modality == "graph":
        return "gnn_or_graph_feature_baseline"
    return "baseline_compatible"


def _fallback_loss(target_type: str) -> str:
    if target_type == "classification":
        return "cross entropy or metric-compatible classification loss"
    if target_type == "regression":
        return "MSE or MAE according to the evaluation metric"
    if target_type == "segmentation":
        return "dice/BCE or cross entropy according to mask target"
    if target_type == "reconstruction":
        return "L1 or MSE reconstruction loss"
    if target_type == "ranking":
        return "ranking loss or metric-compatible regression proxy"
    return "metric-compatible loss"


def _choice(value: Any, allowed: set[str], default: str) -> str:
    text = str(value or "").strip()
    return text if text in allowed else default


def _string(value: Any, default: str) -> str:
    text = str(value or "").strip()
    return text or str(default)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        return []
    return _unique_strings(str(item).strip() for item in values if str(item).strip())


def _unique_strings(values: Any) -> list[str]:
    result: list[str] = []
    for item in values or []:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return default


def _clamp_confidence(value: Any) -> float:
    parsed = _safe_float(value)
    if parsed is None:
        return 0.0
    return round(min(1.0, max(0.0, parsed)), 3)


def _compact_prompt_value(value: Any, *, string_limit: int, list_limit: int) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _compact_prompt_value(item, string_limit=string_limit, list_limit=list_limit)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [
            _compact_prompt_value(item, string_limit=string_limit, list_limit=list_limit)
            for item in list(value)[:list_limit]
        ]
    if isinstance(value, str):
        return _short(value, string_limit)
    return value


def _short(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."

"""Deterministic validation for generated training precision choices."""

from __future__ import annotations

from typing import Any

from agents.review_contracts import ReviewDecision, ReviewIssue
from engine.script_introspection import introspect_training_script
from utils.precision_policy import PrecisionPolicy, resolve_precision_policy


_DETECTED_TO_POLICY = {
    "fp32": "fp32",
    "tf32": "tf32",
    "fp16": "fp16_amp",
    "bf16": "bf16_amp",
    "mixed": "fp16_amp",
    "fp8_te": "fp8_te",
    "mxfp8_te": "mxfp8_te",
    "nvfp4_te": "nvfp4_te",
}


def precision_policy_for_context(agent: Any, context: Any | None = None) -> PrecisionPolicy:
    compact = getattr(context, "compact_context", context) or {}
    mode = getattr(
        getattr(agent, "acfg", None),
        "precision_optimization_mode",
        "normal",
    )
    if not isinstance(compact, dict):
        return resolve_precision_policy(mode=mode)
    stored = compact.get("precision_policy") or {}
    hardware_context = compact.get("hardware_context") or {}
    hardware = dict(hardware_context.get("hardware") or {})
    hardware.update(dict((compact.get("stage_hardware_features") or {}).get("hardware") or {}))
    hardware.setdefault("architecture", stored.get("architecture"))
    hardware.setdefault("compute_capability", stored.get("compute_capability"))
    datatypes = list((compact.get("stage_hardware_features") or {}).get("feature_ids") or [])
    return resolve_precision_policy(hardware, mode=mode, datatypes=datatypes)


def validate_training_precision(
    agent: Any,
    code: str,
    *,
    context: Any | None = None,
) -> tuple[ReviewIssue, ...]:
    metadata = introspect_training_script(code or "")
    detected = str(metadata.get("precision_mode") or "").strip().lower()
    if not detected:
        return ()
    policy = precision_policy_for_context(agent, context)

    if detected in {"generic_fp4", "fp6", "int8_training", "fp8_e5m2_pure"}:
        labels = {
            "generic_fp4": "generic FP4/MXFP4",
            "fp6": "FP6",
            "int8_training": "integer quantized training",
            "fp8_e5m2_pure": "pure FP8 E5M2 training",
        }
        return (
            _critical_issue(
                evidence=(
                    f"Detected {labels[detected]}, which is not an MLEvolve native training policy. "
                    "Generic FP4 is not NVFP4; FP6 has no validated Transformer Engine training path; "
                    "pure E5M2 is allowed only as the backward component of the HYBRID recipe; "
                    "integer formats are capability indicators only."
                ),
                instruction=(
                    "Replace this path with one of the allowed native training policies: "
                    f"{', '.join(policy.allowed_policies)}; keep FP32 as the fallback."
                ),
            ),
        )

    selected = _DETECTED_TO_POLICY.get(detected)
    if selected is None:
        return ()
    if policy.architecture == "unknown" and selected in {
        "fp32",
        "tf32",
        "bf16_amp",
        "fp16_amp",
    }:
        # Without hardware evidence, preserve established 16/32-bit paths.
        # Aggressive low precision still requires positive architecture proof.
        return ()
    if not policy.allows(selected):
        return (
            _critical_issue(
                evidence=(
                    f"Detected precision policy {selected!r}, but {policy.mode} mode on "
                    f"{policy.architecture} allows only {', '.join(policy.allowed_policies)}."
                ),
                instruction=(
                    "Use an allowed hardware-native precision path or disable AMP and fall back to FP32."
                ),
            ),
        )

    if selected in {"fp8_te", "mxfp8_te", "nvfp4_te"}:
        backend = str(metadata.get("precision_backend") or "")
        if backend != "transformer_engine":
            return (
                _critical_issue(
                    evidence=(
                        f"{selected} was selected without an explicit Transformer Engine forward/backward path."
                    ),
                    instruction=(
                        "Use compatible Transformer Engine modules and the documented training recipe, "
                        "or fall back to BF16/FP16."
                    ),
                ),
            )
        adaptation = str(metadata.get("precision_model_adaptation") or "")
        if adaptation != "te_module_replacement":
            return (
                _critical_issue(
                    evidence=(
                        f"{selected} uses Transformer Engine autocast/recipe code but no compatible "
                        "Transformer Engine module or explicit module-conversion path was detected."
                    ),
                    instruction=(
                        "Use Transformer Engine Linear/LayerNormLinear/TransformerLayer modules or an "
                        "explicit compatible conversion path, otherwise fall back to BF16/FP16."
                    ),
                ),
            )
    return ()


def merge_precision_review_issues(
    decision: ReviewDecision | None,
    issues: tuple[ReviewIssue, ...],
) -> ReviewDecision | None:
    if not issues:
        return decision
    existing = list(decision.issues if decision is not None else ())
    identities = {(item.source, item.category, item.evidence) for item in existing}
    for issue in issues:
        identity = (issue.source, issue.category, issue.evidence)
        if identity not in identities:
            existing.append(issue)
            identities.add(identity)
    reasoning = (
        (decision.reasoning + " ") if decision is not None else ""
    ) + "Deterministic hardware precision validation found a policy violation."
    return ReviewDecision(approved=False, reasoning=reasoning.strip(), issues=tuple(existing))


def _critical_issue(*, evidence: str, instruction: str) -> ReviewIssue:
    return ReviewIssue(
        source="precision_policy",
        severity="critical",
        category="datatype_precision",
        owner="datatype_precision",
        evidence=evidence,
        repair_instruction=instruction,
    )

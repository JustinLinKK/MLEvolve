"""Deterministic validation for generated training precision choices."""

from __future__ import annotations

import ast
from typing import Any

from agents.review_contracts import ReviewDecision, ReviewIssue
from engine.script_introspection import introspect_training_script
from utils.precision_policy import CONSERVATIVE_PRECISION_INSTRUCTION, PrecisionPolicy


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
    from agents.prompts.pipeline_decision import _resolve_pipeline_precision_policy

    return _resolve_pipeline_precision_policy(agent, [context])


def validate_training_precision(
    agent: Any,
    code: str,
    *,
    context: Any | None = None,
) -> tuple[ReviewIssue, ...]:
    policy = precision_policy_for_context(agent, context)
    if policy.mode == "conservative":
        return _validate_conservative_precision(code or "", policy)
    metadata = introspect_training_script(code or "")
    detected = str(metadata.get("precision_mode") or "").strip().lower()
    if not detected:
        return ()

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


def _validate_conservative_precision(code: str, policy: PrecisionPolicy) -> tuple[ReviewIssue, ...]:
    """Check explicit precision operations; flags/comments cannot hide dtype casts."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return (_critical_issue(evidence="Cannot validate precision in invalid Python.",
                                instruction="Repair the syntax and use FP32."),)
    aliases: dict[str, str] = {}
    constants: dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for item in node.names:
                aliases[item.asname or item.name] = item.name
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    previous = constants.get(target.id, node.value.value)
                    constants[target.id] = node.value.value if previous == node.value.value else None

    def value(node: ast.AST) -> Any:
        return node.value if isinstance(node, ast.Constant) else constants.get(node.id) if isinstance(node, ast.Name) else None

    forbidden = {"half", "double", "halftensor", "doubletensor", "bfloat16tensor", "float16", "bfloat16", "float64", "fp16", "bf16", "fp64",
                 "fp16_amp", "bf16_amp", "fp8", "fp8_te", "mxfp8_te", "nvfp4_te", "fp4", "fp6"}
    violation = ""
    for node in ast.walk(tree):
        name = node.attr if isinstance(node, ast.Attribute) else aliases.get(node.id, "") if isinstance(node, ast.Name) else ""
        if name.lower() in forbidden or name.lower().startswith(("float8_", "float4_")):
            violation = f"explicit dtype/cast {name} at line {node.lineno}"
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else aliases.get(node.func.id, node.func.id) if isinstance(node.func, ast.Name) else ""
            if name.lower() in {"autocast", "gradscaler", "fp8_autocast", "autocast_context"}:
                disabled = any(kw.arg == "enabled" and value(kw.value) is False for kw in node.keywords)
                if not disabled:
                    violation = f"enabled or unresolved {name} at line {node.lineno}"
            if name.lower() in {"quantize_dynamic", "prepare_qat", "quantizationawaretraining"}:
                violation = f"quantized model operation {name} at line {node.lineno}"
            dtype_args = list(node.args) if name in {"to", "type", "astype", "dtype", "set_default_dtype", "set_default_tensor_type"} else []
            for arg in dtype_args + [kw.value for kw in node.keywords if kw.arg in {"dtype", "torch_dtype", "precision"}]:
                selected = str(value(arg) or "").lower().split(".")[-1]
                if selected in forbidden or selected in {"f2", "f8", "16", "16-mixed", "16-true", "bf16-mixed", "bf16-true", "64", "64-true"}:
                    violation = f"non-float32 argument {selected} at line {node.lineno}"
            for kw in node.keywords:
                if kw.arg in {"load_in_8bit", "load_in_4bit"} and value(kw.value) is not False:
                    violation = f"quantized model loading at line {node.lineno}"
            if name == "set_float32_matmul_precision" and node.args and value(node.args[0]) != "highest" and not policy.allows("tf32"):
                violation = f"TF32 requested without confirmed compatible hardware at line {node.lineno}"
        if isinstance(node, ast.Assign):
            for target in node.targets:
                name = target.id if isinstance(target, ast.Name) else target.attr if isinstance(target, ast.Attribute) else ""
                selected = str(value(node.value) or "").lower()
                if name.lower() in {"precision", "precision_mode", "dtype", "amp_dtype"} and selected in forbidden:
                    violation = f"non-float32 setting {name}={selected} at line {node.lineno}"
                if name.lower() in {"allow_tf32", "use_tf32", "tf32_enabled", "enable_tf32"} and value(node.value) is not False and not policy.allows("tf32"):
                    violation = f"TF32 enabled without confirmed compatible hardware at line {node.lineno}"
                if name == "fp32_precision" and selected != "ieee" and not policy.allows("tf32"):
                    violation = f"TF32 requested without confirmed compatible hardware at line {node.lineno}"
        if violation:
            return (_critical_issue(
                evidence=f"Conservative mode rejects {violation}. Allowed policies: {', '.join(policy.allowed_policies)}.",
                instruction=CONSERVATIVE_PRECISION_INSTRUCTION,
            ),)
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

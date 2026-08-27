"""Deterministic quality-contract validation for generated training scripts."""

from __future__ import annotations

import re

from agents.review_contracts import ReviewDecision, ReviewIssue
from engine.script_introspection import introspect_training_script


def validate_training_contract(code: str) -> tuple[ReviewIssue, ...]:
    metadata = introspect_training_script(code or "")
    lowered = (code or "").lower()
    has_neural_training = (
        any(token in lowered for token in ("import torch", "from torch", "tensorflow", "keras"))
        and any(token in lowered for token in (".backward(", "optimizer.step(", ".fit("))
    )
    epochs = int(metadata.get("proposed_epochs") or 0)
    physical_batch = metadata.get("proposed_batch_size")
    issues: list[ReviewIssue] = []
    if has_neural_training and physical_batch is not None and not metadata.get(
        "quality_safe_physical_batch_sizes"
    ):
        issues.append(
            _issue(
                category="batch_quality_envelope",
                evidence=(
                    "The script chooses a physical batch but does not define "
                    "QUALITY_SAFE_PHYSICAL_BATCH_SIZES."
                ),
                instruction=(
                    "Define an explicit agent-approved list of quality-safe physical batch sizes, "
                    "including the proposed batch, plus BATCH_LR_SCALING_POLICY as fixed, linear, or sqrt."
                ),
            )
        )
    if has_neural_training and physical_batch is not None and not metadata.get(
        "learning_rate_scaling_policy"
    ):
        issues.append(
            _issue(
                category="batch_optimizer_coupling",
                evidence="The batch contract does not declare BATCH_LR_SCALING_POLICY.",
                instruction=(
                    "Declare BATCH_LR_SCALING_POLICY as fixed, linear, or sqrt so dispatch-time "
                    "batch changes resolve learning rate deterministically."
                ),
            )
        )
    if has_neural_training and epochs > 1:
        if not metadata.get("has_validation_early_stopping"):
            issues.append(
                _issue(
                    category="validation_early_stopping",
                    evidence=(
                        f"The raw neural-training script plans {epochs} epochs but has no "
                        "statically recognizable validation-based early stopping."
                    ),
                    instruction=(
                        "Add validation-metric-based patience, stop on no improvement, save the best "
                        "checkpoint, and restore it before final validation/test inference."
                    ),
                )
            )
        has_structured_epoch_marker = bool(
            "MLEVOLVE_EPOCH_METRIC" in (code or "")
            and (
                "json.dumps" in lowered
                or re.search(r"MLEVOLVE_EPOCH_METRIC\s*\{", code or "")
            )
        )
        if not has_structured_epoch_marker:
            issues.append(
                _issue(
                    category="epoch_progress_reporting",
                    evidence=(
                        "The script cannot report completed epochs and its convergence curve because "
                        "it emits no MLEVOLVE_EPOCH_METRIC marker."
                    ),
                    instruction=(
                        "After each validation epoch print MLEVOLVE_EPOCH_METRIC followed by JSON with "
                        "epoch, metric, and metric_name."
                    ),
                )
            )
    return tuple(issues)


def merge_training_contract_review_issues(
    decision: ReviewDecision | None, issues: tuple[ReviewIssue, ...]
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
    prefix = (decision.reasoning + " ") if decision is not None else ""
    return ReviewDecision(
        approved=False,
        reasoning=(
            prefix
            + "Deterministic training-contract validation found a quality or progress-reporting violation."
        ).strip(),
        issues=tuple(existing),
    )


def _issue(*, category: str, evidence: str, instruction: str) -> ReviewIssue:
    return ReviewIssue(
        source="training_contract",
        severity="critical",
        category=category,
        owner="training_evaluation",
        evidence=evidence,
        repair_instruction=instruction,
    )

"""Typed contracts shared by static review, repair, and runtime validation."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


REVIEW_OWNERS = {
    "model_design",
    "datatype_precision",
    "training_evaluation",
    "integration",
    "unclassified",
}
REVIEW_SEVERITIES = {"warning", "critical"}
REVIEW_STATUSES = {"approved", "repaired", "rejected", "unavailable_fail_open"}

_OWNER_ALIASES = {
    "model": "model_design",
    "data": "model_design",
    "structure": "model_design",
    "datatype": "datatype_precision",
    "precision": "datatype_precision",
    "optimizer": "training_evaluation",
    "scheduler": "training_evaluation",
    "batch": "training_evaluation",
    "training": "training_evaluation",
    "evaluation": "training_evaluation",
    "metric": "training_evaluation",
    "submission": "training_evaluation",
}
_TRAINING_CATEGORY_MARKERS = (
    "optimizer",
    "scheduler",
    "batch",
    "training_loop",
    "training-loop",
    "metric",
    "evaluation",
    "submission",
)
_DATATYPE_CATEGORY_MARKERS = ("datatype", "dtype", "precision", "autocast", "amp")
_MODEL_CATEGORY_MARKERS = ("data_leakage", "data-leakage", "model_structure", "model-design")


def _category_has_marker(category: str, markers: tuple[str, ...]) -> bool:
    key = re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")
    return any(
        re.search(rf"(?:^|_){re.escape(re.sub(r'[^a-z0-9]+', '_', marker))}(?:_|$)", key)
        for marker in markers
    )


class ReviewContractError(ValueError):
    """Raised when a reviewer response is structurally unusable."""


@dataclass(frozen=True)
class ReviewIssue:
    source: str
    severity: str
    category: str
    owner: str
    evidence: str
    repair_instruction: str

    def __post_init__(self) -> None:
        if self.severity not in REVIEW_SEVERITIES:
            raise ReviewContractError(f"Unsupported review severity: {self.severity}")
        if self.owner not in REVIEW_OWNERS:
            raise ReviewContractError(f"Unsupported review owner: {self.owner}")
        if not self.category.strip():
            raise ReviewContractError("Review issue category must not be empty")
        if not self.evidence.strip():
            raise ReviewContractError("Review issue evidence must not be empty")
        if not self.repair_instruction.strip():
            raise ReviewContractError("Review issue repair instruction must not be empty")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        default_source: str = "review",
        hardware_aware: bool = True,
    ) -> "ReviewIssue":
        severity = str(value.get("severity") or "").strip().lower()
        owner = str(value.get("owner") or "unclassified").strip().lower().replace("-", "_")
        owner = _OWNER_ALIASES.get(owner, owner)
        category = str(value.get("category") or "").strip()
        if _category_has_marker(category, _TRAINING_CATEGORY_MARKERS):
            owner = "training_evaluation"
        elif _category_has_marker(category, _DATATYPE_CATEGORY_MARKERS):
            owner = "datatype_precision"
        elif _category_has_marker(category, _MODEL_CATEGORY_MARKERS):
            owner = "model_design"
        if not hardware_aware and owner == "datatype_precision":
            owner = "training_evaluation"
        return cls(
            source=str(value.get("source") or default_source).strip() or default_source,
            severity=severity,
            category=category,
            owner=owner,
            evidence=str(value.get("evidence") or "").strip(),
            repair_instruction=str(value.get("repair_instruction") or "").strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReviewDecision:
    approved: bool
    reasoning: str
    issues: tuple[ReviewIssue, ...] = ()

    @property
    def critical_issues(self) -> tuple[ReviewIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "critical")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        hardware_aware: bool = True,
    ) -> "ReviewDecision":
        if not isinstance(value.get("approved"), bool):
            raise ReviewContractError("Review decision must contain boolean 'approved'")
        raw_issues = value.get("issues")
        if not isinstance(raw_issues, list):
            raise ReviewContractError("Review decision must contain an 'issues' list")
        issues = tuple(
            ReviewIssue.from_mapping(issue, hardware_aware=hardware_aware)
            for issue in raw_issues
            if isinstance(issue, Mapping)
        )
        if len(issues) != len(raw_issues):
            raise ReviewContractError("Every review issue must be an object")
        approved = bool(value["approved"])
        if approved and any(issue.severity == "critical" for issue in issues):
            raise ReviewContractError("An approved decision cannot contain critical issues")
        if not approved and not any(issue.severity == "critical" for issue in issues):
            raise ReviewContractError("A rejected decision must contain a critical issue")
        reasoning = str(value.get("reasoning") or "").strip()
        if not reasoning:
            raise ReviewContractError("Review decision reasoning must not be empty")
        return cls(approved=approved, reasoning=reasoning, issues=issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "reasoning": self.reasoning,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class StageRepairResult:
    stage: str
    patch: str = ""
    applied: bool = False
    patch_count: int = 0
    failure_reason: str | None = None
    latency_seconds: float = 0.0
    sequential_retry: bool = False

    @property
    def application_status(self) -> str:
        if self.applied:
            return "applied"
        if self.failure_reason:
            return "failed"
        return "pending"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["application_status"] = self.application_status
        return payload


@dataclass(frozen=True)
class ReviewOutcome:
    code: str
    status: str
    unresolved_issues: tuple[ReviewIssue, ...] = ()
    history: tuple[dict[str, Any], ...] = ()
    changed: bool = False

    def __post_init__(self) -> None:
        if self.status not in REVIEW_STATUSES:
            raise ReviewContractError(f"Unsupported review status: {self.status}")

    @property
    def final_code(self) -> str:
        return self.code

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "status": self.status,
            "unresolved_issues": [issue.to_dict() for issue in self.unresolved_issues],
            "history": list(self.history),
            "changed": self.changed,
        }


def normalize_review_issues(
    issues: Iterable[ReviewIssue | Mapping[str, Any]],
    *,
    default_source: str = "runtime",
    hardware_aware: bool = True,
) -> list[ReviewIssue]:
    normalized: list[ReviewIssue] = []
    for issue in issues:
        if isinstance(issue, ReviewIssue):
            normalized.append(issue)
        elif isinstance(issue, Mapping):
            normalized.append(
                ReviewIssue.from_mapping(
                    issue,
                    default_source=default_source,
                    hardware_aware=hardware_aware,
                )
            )
    return normalized


def append_review_issue(node: Any, issue: ReviewIssue) -> None:
    """Append an issue to a node without duplicating equivalent evidence."""
    current = list(getattr(node, "review_issues", None) or [])
    payload = issue.to_dict()
    identity = (payload["source"], payload["category"], payload["owner"], payload["evidence"])
    for existing in current:
        existing_identity = (
            existing.get("source"),
            existing.get("category"),
            existing.get("owner"),
            existing.get("evidence"),
        )
        if existing_identity == identity:
            return
    current.append(payload)
    node.review_issues = current

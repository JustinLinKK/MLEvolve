"""Shared types for round-level execution through the scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from engine.executor import ExecutionResult


@dataclass(slots=True)
class RoundJobSpec:
    """One runnable MLEvolve node in a scheduler submission round."""

    node_id: str
    code: str
    task_id: str | None = None
    candidate: dict[str, Any] = field(default_factory=dict)
    hardware_context: dict[str, Any] = field(default_factory=dict)
    scheduler_defaults: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RoundJobResult:
    """Execution result for one node from a scheduler round."""

    node_id: str
    result: ExecutionResult
    scheduler_job_id: str | None = None
    scheduler_profile_evidence: dict[str, Any] = field(default_factory=dict)
    probe_result: dict[str, Any] = field(default_factory=dict)
    packing_result: dict[str, Any] = field(default_factory=dict)
    failed: bool = False


@dataclass(slots=True)
class HardwareDecision:
    """Evidence-bound hardware decision made before model generation or execution."""

    stage: str
    rationale: str
    original_params: dict[str, Any] = field(default_factory=dict)
    chosen_params: dict[str, Any] = field(default_factory=dict)
    evidence_refs: list[str] = field(default_factory=list)
    confidence: float = 0.0
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "rationale": self.rationale,
            "original_params": dict(self.original_params),
            "chosen_params": dict(self.chosen_params),
            "evidence_refs": list(self.evidence_refs),
            "confidence": self.confidence,
            "fallback_reason": self.fallback_reason,
        }

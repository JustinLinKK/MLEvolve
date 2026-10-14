"""Typed records exchanged by the lesson-profile subsystem."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


MatchLevel = Literal["exact", "compatible", "similar", "none"]
Maturity = Literal["provisional", "stable", "conflicted", "stale"]
AgentRole = Literal["draft", "improve", "debug", "evolution", "fusion", "aggregation", "review"]

PROFILE_SCHEMA_VERSION = "lesson-profile-v1"


@dataclass(frozen=True, slots=True)
class ProfileIdentity:
    schema_version: str
    model_family: str
    family_confidence: float
    architecture_type: str
    hardware_key: str
    accelerator_key: str
    resource_slice_key: str
    runtime_class: str
    framework_major: str
    cuda_major: str
    backend_class: str
    workload_bucket: str
    profile_key: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LessonRecord:
    lesson_id: str
    lesson_type: str
    agent_audiences: list[str]
    content: dict[str, Any]
    confidence: float
    evidence_refs: list[str]
    change_signature: str = ""
    change_scope: str = "training_only"
    change_action: str = "other"
    layer_type: str = "other"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def empty_profile_view(*, source: str = "sqlite_fallback") -> dict[str, Any]:
    return {
        "family_hardware_profile": {
            "profile_key": "",
            "revision": 0,
            "match_level": "none",
            "maturity": "stale",
            "baseline": {},
            "relevant_lessons": [],
            "warnings": [],
            "evidence_refs": [],
            "confidence": 0.0,
            "source": source,
        }
    }

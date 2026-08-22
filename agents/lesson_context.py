"""Role-scoped prompt projection for the lesson profile database."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import yaml

from lesson_profile_database.models import empty_profile_view


@dataclass(slots=True)
class LessonPromptContext:
    role: str
    compact_context: dict[str, Any]
    prompt_section: str

    @property
    def available(self) -> bool:
        view = self.compact_context.get("family_hardware_profile") or {}
        return str(view.get("match_level") or "none") != "none"


def _anchor_for_stage(stage: str, parent_node: Any | None) -> Any | None:
    del stage
    return parent_node


def get_lesson_context_for_stage(
    agent: Any,
    stage: str,
    *,
    parent_node: Any | None = None,
    code: str = "",
    error: str = "",
) -> LessonPromptContext:
    client = getattr(agent, "lesson_profile_client", None)
    if client is None:
        raw = empty_profile_view()
    else:
        try:
            anchor = _anchor_for_stage(stage, parent_node)
            raw = client.profile_for_agent(
                agent,
                agent_role=stage,
                node=anchor,
                code=code or str(getattr(anchor, "code", "") or ""),
                error=error,
            )
        except Exception:
            raw = empty_profile_view()
    view = dict(raw.get("family_hardware_profile") or {})
    if str(view.get("match_level") or "none") == "none":
        return LessonPromptContext(role=stage, compact_context=raw, prompt_section="")
    payload = yaml.safe_dump(raw, sort_keys=False, allow_unicode=True).strip()
    section = (
        "# Family–Hardware Lesson Profile\n\n"
        "This is evidence-backed prior experience. Current task constraints and measured runtime evidence override it.\n\n"
        f"```yaml\n{payload}\n```"
    )
    return LessonPromptContext(role=stage, compact_context=raw, prompt_section=section)


def lesson_context_instructions(context: LessonPromptContext) -> dict[str, list[str]]:
    if not context.available:
        return {}
    view = context.compact_context["family_hardware_profile"]
    match = str(view.get("match_level") or "none")
    guidance = [
        "Use only lessons relevant to your current stage and proposed change.",
        "Keep the cited profile key, revision, confidence, and evidence references auditable.",
        "Current constraints and fresh measurements override remembered advice.",
    ]
    if match != "exact":
        guidance.append("This is advisory, not a verified default; explicitly revalidate every reused assumption.")
    return {"Lesson profile guidance": guidance}


def apply_lesson_context_to_pipeline_decision(
    pipeline_decision: dict[str, Any] | None,
    context: LessonPromptContext,
) -> None:
    if pipeline_decision is None or not context.available:
        return
    view = context.compact_context["family_hardware_profile"]
    evidence = pipeline_decision.setdefault("evidence", {})
    evidence.update({
        "lesson_profile_used": True,
        "lesson_profile_key": view.get("profile_key"),
        "lesson_profile_revision": view.get("revision"),
        "lesson_match_level": view.get("match_level"),
        "lesson_ids": [item.get("lesson_id") for item in view.get("relevant_lessons") or [] if item.get("lesson_id")],
        "lesson_confidence": view.get("confidence"),
        "lesson_evidence_refs": list(view.get("evidence_refs") or []),
    })


def apply_lesson_context_to_node(node: Any, context: LessonPromptContext) -> None:
    if not context.available:
        return
    view = context.compact_context["family_hardware_profile"]
    node.lesson_profile_context = copy.deepcopy(context.compact_context)
    node.lesson_profile_key = str(view.get("profile_key") or "") or None
    node.lesson_profile_revision = int(view.get("revision") or 0) or None
    node.lesson_match_level = str(view.get("match_level") or "none")
    node.lesson_profile_maturity = str(view.get("maturity") or "provisional")
    node.lesson_ids = [
        str(item.get("lesson_id"))
        for item in view.get("relevant_lessons") or []
        if item.get("lesson_id")
    ]
    node.lesson_confidence = float(view.get("confidence") or 0.0)
    node.lesson_evidence_refs = [str(item) for item in view.get("evidence_refs") or []]
    apply_lesson_context_to_pipeline_decision(getattr(node, "pipeline_decision", None), context)

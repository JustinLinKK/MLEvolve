"""Stage-aware static review and selective repair workflow."""

from __future__ import annotations

import logging
import time
from typing import Any, cast

from agents.hardware_context import get_hardware_context_for_stage, hardware_context_instructions
from agents.lesson_context import (
    apply_lesson_context_to_node,
    get_lesson_context_for_stage,
    lesson_context_instructions,
)
from agents.prompts import get_internet_clarification
from agents.prompts.validation_template_prompts import get_code_review_prompt
from agents.review_contracts import ReviewDecision, ReviewOutcome
from agents.precision_validation import merge_precision_review_issues, validate_training_precision
from agents.stage_repair import is_hardware_aware, repair_selected_stages
from engine.search_node import SearchNode
from llm import FunctionSpec, query

logger = logging.getLogger("MLEvolve")

_ISSUE_SCHEMA = {
    "type": "object",
    "properties": {
        "source": {"type": "string", "description": "Use 'static_review'."},
        "severity": {"type": "string", "enum": ["warning", "critical"]},
        "category": {"type": "string"},
        "owner": {
            "type": "string",
            "enum": ["model_design", "datatype_precision", "training_evaluation", "integration", "unclassified"],
        },
        "evidence": {"type": "string"},
        "repair_instruction": {"type": "string"},
    },
    "required": ["source", "severity", "category", "owner", "evidence", "repair_instruction"],
}

CODE_REVIEW_SPEC = FunctionSpec(
    name="submit_code_review",
    json_schema={
        "type": "object",
        "properties": {
            "approved": {"type": "boolean", "description": "True only when there are no critical issues."},
            "reasoning": {"type": "string", "description": "A concise 2-4 sentence classification summary."},
            "issues": {"type": "array", "items": _ISSUE_SCHEMA},
        },
        "required": ["approved", "reasoning", "issues"],
    },
    description="Classify stage-owned issues in a complete generated ML script.",
)


def _review_config(agent: Any) -> Any:
    return getattr(getattr(agent, "acfg", None), "review", None)


def _event(agent: Any, node: SearchNode, event_type: str, payload: dict[str, Any]) -> None:
    try:
        from utils.pipeline_logging import log_pipeline_event, record_pipeline_node_action

        log_pipeline_event(agent, event_type, node=node, payload=payload)
        record_pipeline_node_action(agent, node, event_type, payload=payload)
    except Exception:
        pass


def _build_review_prompt(agent: Any, node: SearchNode, code: str) -> tuple[dict[str, Any], Any]:
    prompt = get_code_review_prompt(task_desc=agent.task_desc, code=code)
    hardware_ctx = get_hardware_context_for_stage(
        agent, "code_review", parent_node=getattr(node, "parent", None), code=code
    )
    hardware_section = hardware_ctx.prompt_section
    if hardware_section:
        instructions = prompt.pop("Instructions")
        prompt["Hardware/Profile Optimization Context"] = hardware_section
        prompt["Instructions"] = instructions
        prompt["Instructions"] |= hardware_context_instructions(hardware_ctx)
        prompt["Instructions"]["Hardware-aware review guidance"] = [
            "Flag hardware-critical issues only when the supplied profile evidence is strong.",
            "Preserve the chosen model/backbone and classify the narrow owning stage.",
        ]
    lesson_ctx = get_lesson_context_for_stage(
        agent, "code_review", parent_node=node, code=code
    )
    if lesson_ctx.prompt_section:
        instructions = prompt.pop("Instructions")
        prompt["Family–Hardware Lesson Profile"] = lesson_ctx.prompt_section
        prompt["Instructions"] = instructions
        prompt["Instructions"] |= lesson_context_instructions(lesson_ctx)
        apply_lesson_context_to_node(node, lesson_ctx)
    internet_clarification = get_internet_clarification(getattr(agent.cfg, "pretrain_model_dir", ""))
    prompt.setdefault("Instructions", {})
    if "Implementation guideline" in prompt["Instructions"]:
        prompt["Instructions"]["Implementation guideline"].extend(internet_clarification)
    else:
        prompt["Instructions"]["⚠️ Internet Access Clarification"] = internet_clarification
    return prompt, hardware_ctx


def classify_code(agent: Any, node: SearchNode, code: str) -> tuple[ReviewDecision | None, dict[str, Any]]:
    """Return a validated decision, or None when the reviewer stays unavailable/invalid."""
    prompt, hardware_ctx = _build_review_prompt(agent, node, code)
    hardware_context_used = bool(hardware_ctx.prompt_section)
    policy_issues = validate_training_precision(agent, code, context=hardware_ctx)
    retries = max(1, int(getattr(_review_config(agent), "classifier_retries", 3)))
    started = time.monotonic()
    last_error = "reviewer unavailable"
    for attempt in range(retries):
        try:
            response = cast(
                dict[str, Any],
                query(
                    system_message=prompt,
                    user_message=None,
                    func_spec=CODE_REVIEW_SPEC,
                    model=agent.acfg.code.model,
                    temperature=agent.acfg.code.temp,
                    cfg=agent.cfg,
                ),
            )
            decision = ReviewDecision.from_mapping(response, hardware_aware=is_hardware_aware(agent))
            decision = merge_precision_review_issues(decision, policy_issues)
            return decision, {
                "attempts": attempt + 1,
                "latency_seconds": time.monotonic() - started,
                "hardware_context_used": hardware_context_used,
            }
        except Exception as exc:
            last_error = str(exc)
            if attempt + 1 < retries:
                logger.warning(
                    "Code review attempt %s/%s failed for node %s: %s",
                    attempt + 1, retries, node.id, exc,
                )
    if policy_issues:
        return merge_precision_review_issues(None, policy_issues), {
            "attempts": retries,
            "latency_seconds": time.monotonic() - started,
            "hardware_context_used": hardware_context_used,
            "error": last_error,
            "deterministic_precision_validation": True,
        }
    return None, {
        "attempts": retries,
        "latency_seconds": time.monotonic() - started,
        "hardware_context_used": hardware_context_used,
        "error": last_error,
    }


def _store_outcome(node: SearchNode, outcome: ReviewOutcome) -> None:
    node.review_status = outcome.status
    node.review_issues = [issue.to_dict() for issue in outcome.unresolved_issues]
    node.review_history = list(outcome.history)


def review_and_repair(agent: Any, node: SearchNode) -> ReviewOutcome:
    """Classify a script, selectively repair critical owners, and re-review it."""
    review_cfg = _review_config(agent)
    if review_cfg is not None and not bool(getattr(review_cfg, "enabled", True)):
        hardware_ctx = get_hardware_context_for_stage(
            agent, "code_review", parent_node=getattr(node, "parent", None), code=node.code
        )
        policy_issues = validate_training_precision(agent, node.code, context=hardware_ctx)
        if policy_issues:
            outcome = ReviewOutcome(
                code=node.code,
                status="rejected",
                unresolved_issues=policy_issues,
                history=({"event": "precision_policy_rejected_with_review_disabled"},),
            )
            _store_outcome(node, outcome)
            return outcome
        outcome = ReviewOutcome(code=node.code, status="approved", history=({"event": "review_disabled"},))
        _store_outcome(node, outcome)
        return outcome

    code = node.code
    original_code = code
    history: list[dict[str, Any]] = []
    decision, metadata = classify_code(agent, node, code)
    if decision is None:
        fail_open = bool(getattr(review_cfg, "fail_open_on_unavailable", True))
        status = "unavailable_fail_open" if fail_open else "rejected"
        history.append({"event": "review_unavailable", **metadata})
        outcome = ReviewOutcome(code=code, status=status, history=tuple(history))
        _store_outcome(node, outcome)
        _event(agent, node, "review_fail_open" if fail_open else "review_rejected", metadata)
        return outcome

    history.append({"event": "review_decision", "round": 0, **metadata, **decision.to_dict()})
    _event(
        agent, node, "review_round_completed",
        {"round": 0, **metadata, "approved": decision.approved, "issue_count": len(decision.issues)},
    )

    max_rounds = max(0, int(getattr(review_cfg, "max_repair_rounds", 2)))
    repair_round = 0
    while decision.critical_issues and repair_round < max_rounds:
        repair_round += 1
        known_critical = decision.critical_issues
        code, results, stats = repair_selected_stages(agent, node, code, known_critical)
        repair_entry = {
            "event": "repair_round",
            "round": repair_round,
            "repairs": [result.to_dict() for result in results],
            **stats,
        }
        history.append(repair_entry)
        _event(agent, node, "review_repair_round_completed", repair_entry)
        decision, metadata = classify_code(agent, node, code)
        if decision is None:
            history.append({"event": "review_unavailable_after_repair", "round": repair_round, **metadata})
            outcome = ReviewOutcome(
                code=code,
                status="rejected",
                unresolved_issues=known_critical,
                history=tuple(history),
                changed=code != original_code,
            )
            _store_outcome(node, outcome)
            _event(agent, node, "review_rejected", {"reason": "re-review unavailable", **metadata})
            return outcome
        history.append({"event": "review_decision", "round": repair_round, **metadata, **decision.to_dict()})
        _event(
            agent, node, "review_round_completed",
            {"round": repair_round, **metadata, "approved": decision.approved, "issue_count": len(decision.issues)},
        )

    unresolved = decision.critical_issues
    if unresolved and bool(getattr(review_cfg, "reject_unresolved_critical", True)):
        status = "rejected"
    else:
        status = "repaired" if code != original_code else "approved"
    outcome = ReviewOutcome(
        code=code,
        status=status,
        unresolved_issues=decision.issues,
        history=tuple(history),
        changed=code != original_code,
    )
    _store_outcome(node, outcome)
    _event(
        agent, node, "review_rejected" if status == "rejected" else "review_completed",
        {
            "status": status,
            "repair_rounds": repair_round,
            "unresolved_critical_count": len(unresolved),
            "changed": outcome.changed,
        },
    )
    return outcome


def run(agent: Any, node: SearchNode) -> str:
    """Compatibility wrapper for callers that still expect a code string."""
    return review_and_repair(agent, node).code

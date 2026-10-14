"""Stage-owned repair for candidates that fail runtime validation."""

from __future__ import annotations

from dataclasses import replace
import logging
import re

from agents.cuda_docs_context import (
    format_cuda_docs_prompt_section,
    get_cuda_docs_context,
)
from agents.hardware_context import (
    apply_hardware_context_to_node,
    get_hardware_context_for_stage,
)
from agents.lesson_context import (
    apply_lesson_context_to_node,
    apply_lesson_context_to_pipeline_decision,
    get_lesson_context_for_stage,
)
from agents.prompts import (
    apply_pipeline_decision_to_node,
    build_pipeline_decision,
)
from agents.review_contracts import ReviewIssue, normalize_review_issues
from agents.stage_repair import is_hardware_aware, repair_selected_stages
from agents.triggers import register_node
from engine.search_node import SearchNode
from utils.response import extract_plan_from_diff_response, trim_long_string

logger = logging.getLogger("MLEvolve")


_DEBUG_REPORT_HEADING_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?\s*(Bug Report|Fix Report)\s*(?:\*\*)?\s*:?\s*(.*)$"
)


def _clean_report_section(text: str) -> str:
    cleaned = str(text or "")
    stop_tokens = ["<<<<<<< SEARCH", "< SEARCH", "```"]
    stop_positions = [
        cleaned.find(token) for token in stop_tokens if cleaned.find(token) != -1
    ]
    if stop_positions:
        cleaned = cleaned[: min(stop_positions)]
    return re.sub(r"\n{3,}", "\n\n", cleaned.strip())


def _extract_debug_reports(text: str) -> tuple[str, str]:
    """Extract Bug Report and Fix Report sections from a repair response."""
    if not text:
        return "", ""

    matches = list(_DEBUG_REPORT_HEADING_RE.finditer(text))
    if not matches:
        return "", ""

    sections: dict[str, str] = {}
    for idx, match in enumerate(matches):
        key = match.group(1).lower().replace(" ", "_")
        inline_text = match.group(2).strip()
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body_text = text[match.end() : next_start].strip()
        sections[key] = _clean_report_section(
            "\n".join(part for part in (inline_text, body_text) if part)
        )
    return sections.get("bug_report", ""), sections.get("fix_report", "")


def _fallback_bug_report(parent_node: SearchNode) -> str:
    parts: list[str] = []
    if getattr(parent_node, "exc_type", None):
        parts.append(f"Exception type: {parent_node.exc_type}")
    if getattr(parent_node, "analysis", None):
        parts.append(f"Agent analysis: {parent_node.analysis}")
    if parent_node.term_out:
        parts.append(
            "Execution output: "
            + trim_long_string(parent_node.term_out, threshold=1200, k=550)
        )
    return "\n".join(parts) or (
        "The parent node failed or produced invalid output; no detailed error "
        "report was available."
    )


def _build_debug_reports(
    *,
    report_source_text: str | None,
    parent_node: SearchNode,
    plan: str | None,
) -> tuple[str, str]:
    bug_report, fix_report = _extract_debug_reports(report_source_text or "")
    if not bug_report:
        bug_report = _fallback_bug_report(parent_node)
    if not fix_report:
        fallback_plan = extract_plan_from_diff_response(report_source_text or "").strip()
        fix_report = fallback_plan or (plan or "").strip()
    if not fix_report:
        fix_report = "Applied a scoped repair for the parent node failure."
    return bug_report, fix_report


def _runtime_issue(parent_node: SearchNode) -> ReviewIssue:
    evidence = _fallback_bug_report(parent_node)
    return ReviewIssue(
        source="runtime",
        severity="critical",
        category="runtime_failure",
        owner="integration",
        evidence=evidence,
        repair_instruction=(
            "Repair the cross-stage interface or executable script defect while "
            "preserving the selected model, precision, and training intent."
        ),
    )


def _canonical_repair_issues(agent: object, parent_node: SearchNode) -> list[ReviewIssue]:
    try:
        issues = normalize_review_issues(
            getattr(parent_node, "review_issues", None) or [],
            default_source="runtime",
            hardware_aware=is_hardware_aware(agent),
        )
    except Exception as exc:
        logger.warning(
            "Runtime issue classification is unusable for node %s: %s",
            parent_node.id,
            exc,
        )
        issues = []

    critical = [issue for issue in issues if issue.severity == "critical"]
    if not critical:
        return [_runtime_issue(parent_node)]
    return [
        replace(issue, owner="integration")
        if issue.owner == "unclassified"
        else issue
        for issue in critical
    ]


def run(agent: object, parent_node: SearchNode) -> SearchNode | None:
    """Repair only the stages that own the runtime failure."""
    if not parent_node.add_expected_child_count(agent.scfg):
        logger.info(
            "Debug child limit reached for node %s, skipping generation.",
            parent_node.id,
        )
        return None

    hardware_ctx = get_hardware_context_for_stage(
        agent, "debug", parent_node=parent_node
    )
    cuda_docs_ctx = get_cuda_docs_context(
        agent,
        "debug",
        parent_node=parent_node,
        hardware_context=hardware_ctx,
    )
    cuda_docs_section = format_cuda_docs_prompt_section(
        cuda_docs_ctx,
        service=getattr(agent, "cuda_docs_service", None),
        role="debug",
    )
    lesson_ctx = get_lesson_context_for_stage(
        agent, "debug", parent_node=parent_node, code=parent_node.code
    )
    pipeline_decision = build_pipeline_decision(
        agent,
        stage="debug",
        data_preview=agent.data_preview,
        hardware_contexts=[hardware_ctx],
        parent_pipeline_decision=getattr(parent_node, "pipeline_decision", None),
        previous_code=parent_node.code,
        execution_output=parent_node.term_out,
        stage_context=str(getattr(parent_node, "analysis", "") or ""),
    )
    apply_lesson_context_to_pipeline_decision(pipeline_decision, lesson_ctx)
    critical_issues = _canonical_repair_issues(agent, parent_node)
    repaired_code, repair_results, repair_stats = repair_selected_stages(
        agent,
        parent_node,
        parent_node.code,
        critical_issues,
        cuda_docs_evidence=cuda_docs_section,
    )
    applied_results = [result for result in repair_results if result.applied]
    if repaired_code == parent_node.code or not applied_results:
        logger.warning(
            "Stage-owned runtime repair was unusable for node %s: %s",
            parent_node.id,
            [result.to_dict() for result in repair_results],
        )
        return None

    stage_names = [result.stage for result in applied_results]
    bug_report = "\n".join(
        f"{issue.owner}: {issue.evidence}" for issue in critical_issues
    )
    fix_report = (
        "Applied scoped stage repair patches for "
        + ", ".join(stage_names)
        + "; unaffected stage code and interfaces were preserved."
    )
    new_node = SearchNode(
        plan=fix_report,
        code=repaired_code,
        parent=parent_node,
        stage="debug",
        local_best_node=parent_node.local_best_node,
        from_topk=getattr(parent_node, "_topk_triggered", False),
        bug_report=bug_report,
        fix_report=fix_report,
    )
    apply_hardware_context_to_node(new_node, hardware_ctx)
    apply_pipeline_decision_to_node(new_node, pipeline_decision)
    apply_lesson_context_to_node(new_node, lesson_ctx)
    register_node(
        agent,
        new_node,
        {
            "stage_owned_runtime_repair": True,
            "issues": [issue.to_dict() for issue in critical_issues],
            "repairs": [result.to_dict() for result in repair_results],
            "stats": repair_stats,
        },
        parent_node=parent_node,
    )
    try:
        from utils.pipeline_logging import log_pipeline_event, record_pipeline_node_action

        payload = {
            "owners": sorted({issue.owner for issue in critical_issues}),
            "stages": stage_names,
            **repair_stats,
        }
        log_pipeline_event(
            agent, "runtime_stage_owned_repair", node=new_node, payload=payload
        )
        record_pipeline_node_action(
            agent, new_node, "runtime_stage_owned_repair", payload=payload
        )
    except Exception:
        pass
    logger.info(
        "[debug-stage-owned] %s -> node %s (%s)",
        parent_node.id,
        new_node.id,
        stage_names,
    )
    return new_node

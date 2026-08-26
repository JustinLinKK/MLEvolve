"""Selective stage-owned SEARCH/REPLACE repair orchestration."""

from __future__ import annotations

import ast
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, Callable, Iterable, Sequence

from agents.coder.diff_coder import SearchReplacePatcher
from agents.coder.stepwise_coder import create_default_step_agents
from agents.hardware_context import get_hardware_context_for_stage
from agents.review_contracts import ReviewIssue, StageRepairResult
from llm import generate

logger = logging.getLogger("MLEvolve")

STAGE_ORDER = ("model_design", "datatype_precision", "training_evaluation", "integration")
_PATCH_PATTERN = SearchReplacePatcher.PATCH_PATTERN


def is_hardware_aware(agent: Any) -> bool:
    experiment = getattr(getattr(agent, "cfg", None), "experiment", None)
    mode = str(getattr(experiment, "mode", "hardware_aware") or "hardware_aware").lower().replace("-", "_")
    return mode not in {"origin", "baseline"}


def _review_config(agent: Any) -> Any:
    return getattr(getattr(agent, "acfg", None), "review", None)


def group_repair_issues(agent: Any, issues: Iterable[ReviewIssue]) -> dict[str, list[ReviewIssue]]:
    grouped: dict[str, list[ReviewIssue]] = {}
    hardware_aware = is_hardware_aware(agent)
    for issue in issues:
        if issue.severity != "critical":
            continue
        owner = issue.owner
        if owner == "datatype_precision" and not hardware_aware:
            owner = "training_evaluation"
            issue = replace(issue, owner=owner)
        grouped.setdefault(owner, []).append(issue)
    return grouped


def _stage_ownership(agent: Any, stage: str) -> tuple[str, Sequence[str]]:
    for step_agent in create_default_step_agents(hardware_aware=is_hardware_aware(agent)):
        if step_agent.name == stage:
            return step_agent.description, step_agent.guidelines
    if stage == "integration":
        return (
            "Repair cross-stage interfaces, imports, control flow, and assembly only.",
            (
                "Preserve each stage's chosen model, precision policy, optimizer, and evaluation intent.",
                "Change only code necessary to restore a valid interface or executable merged script.",
            ),
        )
    return ("Repair only the issues assigned to this stage.", ())


def _build_repair_prompt(
    agent: Any,
    node: Any,
    code: str,
    stage: str,
    issues: Sequence[ReviewIssue],
    *,
    cuda_docs_evidence: str = "",
) -> str:
    description, guidelines = _stage_ownership(agent, stage)
    hardware = get_hardware_context_for_stage(
        agent,
        stage,
        parent_node=getattr(node, "parent", None),
        code=code,
    ).prompt_section
    payload = {
        "role": f"{stage} repair specialist",
        "stage_ownership": description,
        "issues": [issue.to_dict() for issue in issues],
        "task_context": getattr(agent, "task_desc", ""),
        "pipeline_decision": getattr(node, "pipeline_decision", None) or {},
        "stage_notes": getattr(node, "stage_note_board", None) or [],
        "hardware_evidence": hardware,
        # This is bounded, source-labelled evidence composed by the local
        # policy adapter. It is reference material, never an instruction or a
        # raw MCP response.
        "cuda_documentation_evidence": str(cuda_docs_evidence or ""),
        "ownership_guidelines": list(guidelines),
        "merged_script": code,
    }
    return (
        "Repair only the assigned stage-owned critical issues in the complete merged Python script below. "
        "Preserve every other stage's behavior and public variables/interfaces. Return one or more raw "
        "SEARCH/REPLACE blocks and no prose or markdown fences. Every SEARCH block must be non-empty and "
        "copied exactly from the merged script. Keep each SEARCH span as small as safely possible.\n\n"
        + json.dumps(payload, indent=2, default=str)
    )


def _default_patch_generator(agent: Any, prompt: str) -> str:
    return str(
        generate(
            prompt=prompt,
            temperature=agent.acfg.code.temp,
            cfg=agent.cfg,
        )
        or ""
    )


def generate_stage_patch(
    agent: Any,
    node: Any,
    code: str,
    stage: str,
    issues: Sequence[ReviewIssue],
    *,
    generator: Callable[[Any, str], str] | None = None,
    sequential_retry: bool = False,
    cuda_docs_evidence: str = "",
) -> StageRepairResult:
    started = time.monotonic()
    generator = generator or _default_patch_generator
    retries = max(1, int(getattr(_review_config(agent), "repair_retries", 2)))
    last_error = "repair agent did not return a patch"
    prompt = _build_repair_prompt(
        agent,
        node,
        code,
        stage,
        issues,
        cuda_docs_evidence=cuda_docs_evidence,
    )
    for _ in range(retries):
        try:
            patch = generator(agent, prompt).strip()
            blocks = list(_PATCH_PATTERN.finditer(patch))
            if not blocks:
                last_error = "malformed SEARCH/REPLACE response"
                continue
            if _PATCH_PATTERN.sub("", patch).strip():
                last_error = "response contains text outside complete SEARCH/REPLACE blocks"
                continue
            if any(not match.group(1).strip() for match in blocks):
                last_error = "empty SEARCH block"
                continue
            return StageRepairResult(
                stage=stage,
                patch=patch,
                patch_count=len(blocks),
                latency_seconds=time.monotonic() - started,
                sequential_retry=sequential_retry,
            )
        except Exception as exc:
            last_error = str(exc)
    return StageRepairResult(
        stage=stage,
        failure_reason=last_error,
        latency_seconds=time.monotonic() - started,
        sequential_retry=sequential_retry,
    )


def _resolve_patch_spans(code: str, patch: str) -> tuple[list[tuple[int, int]], str | None]:
    patcher = SearchReplacePatcher()
    spans: list[tuple[int, int]] = []
    for match in _PATCH_PATTERN.finditer(patch):
        search_text = patcher._strip_trailing_whitespace(match.group(1))
        matched_text, start = patcher._find_indented_match(search_text, code)
        if start < 0:
            return [], "SEARCH block does not match the repair base"
        length = len(matched_text)
        span = (start, start + length)
        if any(span[0] < other[1] and other[0] < span[1] for other in spans):
            return [], "patch contains overlapping SEARCH spans"
        spans.append(span)
    return spans, None


def _patches_overlap(code: str, first: StageRepairResult, second: StageRepairResult) -> tuple[bool, str | None]:
    first_spans, error = _resolve_patch_spans(code, first.patch)
    if error:
        return True, f"{first.stage}: {error}"
    second_spans, error = _resolve_patch_spans(code, second.patch)
    if error:
        return True, f"{second.stage}: {error}"
    overlap = any(a[0] < b[1] and b[0] < a[1] for a in first_spans for b in second_spans)
    return overlap, "parallel SEARCH spans overlap" if overlap else None


def _apply_result(code: str, result: StageRepairResult) -> tuple[str, StageRepairResult]:
    if result.failure_reason or not result.patch:
        return code, result
    try:
        patched, count = SearchReplacePatcher().apply_patch(result.patch, code, strict=True)
        if count != result.patch_count:
            raise ValueError(f"applied {count} of {result.patch_count} patch blocks")
        if patched == code:
            raise ValueError("patch left the script unchanged")
        ast.parse(patched)
        return patched, replace(result, applied=True, patch_count=count)
    except Exception as exc:
        return code, replace(result, applied=False, failure_reason=str(exc))


def _log_repair_event(agent: Any, node: Any, event_type: str, payload: dict[str, Any]) -> None:
    try:
        from utils.pipeline_logging import log_pipeline_event, record_pipeline_node_action

        log_pipeline_event(agent, event_type, node=node, payload=payload)
        record_pipeline_node_action(agent, node, event_type, payload=payload)
    except Exception:
        pass


def _run_parallel_pair(
    agent: Any,
    node: Any,
    code: str,
    primary_stage: str,
    training_stage: str,
    grouped: dict[str, list[ReviewIssue]],
    generator: Callable[[Any, str], str] | None,
    cuda_docs_evidence: str,
) -> tuple[str, list[StageRepairResult], bool]:
    _log_repair_event(
        agent,
        node,
        "review_parallel_repair_batch",
        {"stages": [primary_stage, training_stage]},
    )
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="stage-repair") as pool:
        primary_future = pool.submit(
            generate_stage_patch,
            agent,
            node,
            code,
            primary_stage,
            grouped[primary_stage],
            generator=generator,
            cuda_docs_evidence=cuda_docs_evidence,
        )
        training_future = pool.submit(
            generate_stage_patch,
            agent,
            node,
            code,
            training_stage,
            grouped[training_stage],
            generator=generator,
            cuda_docs_evidence=cuda_docs_evidence,
        )
        primary_result = primary_future.result()
        training_result = training_future.result()

    overlap, conflict_reason = _patches_overlap(code, primary_result, training_result)
    primary_code, primary_result = _apply_result(code, primary_result)
    results = [primary_result]
    if overlap or primary_result.failure_reason:
        _log_repair_event(
            agent,
            node,
            "review_patch_conflict",
            {"stages": [primary_stage, training_stage], "reason": conflict_reason or primary_result.failure_reason},
        )
        retry = generate_stage_patch(
            agent,
            node,
            primary_code,
            training_stage,
            grouped[training_stage],
            generator=generator,
            sequential_retry=True,
            cuda_docs_evidence=cuda_docs_evidence,
        )
        merged, retry = _apply_result(primary_code, retry)
        results.append(retry)
        return merged, results, True

    merged, training_result = _apply_result(primary_code, training_result)
    if training_result.failure_reason:
        # The patches were independent on the base but did not combine safely.
        _log_repair_event(
            agent,
            node,
            "review_patch_conflict",
            {"stages": [primary_stage, training_stage], "reason": training_result.failure_reason},
        )
        retry = generate_stage_patch(
            agent,
            node,
            primary_code,
            training_stage,
            grouped[training_stage],
            generator=generator,
            sequential_retry=True,
            cuda_docs_evidence=cuda_docs_evidence,
        )
        merged, retry = _apply_result(primary_code, retry)
        results.append(retry)
        return merged, results, True
    results.append(training_result)
    return merged, results, False


def repair_selected_stages(
    agent: Any,
    node: Any,
    code: str,
    issues: Iterable[ReviewIssue],
    *,
    generator: Callable[[Any, str], str] | None = None,
    cuda_docs_evidence: str = "",
) -> tuple[str, list[StageRepairResult], dict[str, Any]]:
    """Repair only issue owners selected by the review decision."""
    grouped = group_repair_issues(agent, issues)
    selected = [stage for stage in STAGE_ORDER if stage in grouped]
    available_pipeline_stages = 3 if is_hardware_aware(agent) else 2
    selected_pipeline_stages = sum(
        stage in grouped for stage in ("model_design", "datatype_precision", "training_evaluation")
    )
    stats: dict[str, Any] = {
        "selected_stages": selected,
        "stage_calls_skipped": max(0, available_pipeline_stages - selected_pipeline_stages),
        "stage_repair_calls": len(selected),
        "parallel_batches": 0,
        "patch_conflicts": 0,
    }
    results: list[StageRepairResult] = []
    current = code
    parallel_enabled = bool(getattr(_review_config(agent), "parallel_training_repairs", True))

    primary = "model_design" if "model_design" in grouped else "datatype_precision" if "datatype_precision" in grouped else None
    if parallel_enabled and primary and "training_evaluation" in grouped:
        current, pair_results, conflicted = _run_parallel_pair(
            agent,
            node,
            current,
            primary,
            "training_evaluation",
            grouped,
            generator,
            cuda_docs_evidence,
        )
        results.extend(pair_results)
        stats["parallel_batches"] += 1
        stats["patch_conflicts"] += int(conflicted)
        stats["stage_repair_calls"] += int(conflicted)
        handled = {primary, "training_evaluation"}
    else:
        handled = set()

    # Stage 1 then Stage 2 is an invariant. With all three selected, Stage 2 sees
    # the already merged Stage 1 + Stage 3 code.
    for stage in STAGE_ORDER:
        if stage not in grouped or stage in handled:
            continue
        result = generate_stage_patch(
            agent,
            node,
            current,
            stage,
            grouped[stage],
            generator=generator,
            cuda_docs_evidence=cuda_docs_evidence,
        )
        current, result = _apply_result(current, result)
        results.append(result)

    for result in results:
        _log_repair_event(agent, node, "review_stage_repair_completed", result.to_dict())
    return current, results, stats

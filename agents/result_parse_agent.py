import logging
import math
import re
import time
from typing import cast

from llm import FunctionSpec, query
from engine.search_node import SearchNode
from engine.executor import ExecutionResult
from utils.metric import MetricValue, WorstMetricValue
from utils.response import wrap_code
from engine.validation import call_validate, _validate_submission_with_retry, validate_submission_content_quality
from agents import data_leakage_agent
from agents.triggers import should_check_data_leakage
from agents.review_contracts import ReviewIssue, append_review_issue, normalize_review_issues

logger = logging.getLogger("MLEvolve")

_FINAL_VALIDATION_SCORE_RE = re.compile(
    r"Final\s+Validation\s+Score\s*:\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
_ADVISORY_QUALITY_CATEGORY_MARKERS = (
    "model_quality",
    "underfit",
    "overfit",
    "non_improving",
    "no_learned_signal",
    "poor_model_performance",
    "weak_metric",
    "resource_utilization",
    "auxiliary_loss_design",
)

_RUNTIME_ISSUE_SCHEMA = {
    "type": "object",
    "properties": {
        "source": {"type": "string", "description": "Use 'runtime'."},
        "severity": {
            "type": "string",
            "enum": ["warning", "critical"],
            "description": (
                "Use critical only when the execution result is unusable and is_bug=true. "
                "Underfitting, overfitting, weak/non-improving metrics, and resource "
                "utilization concerns are warnings, not execution failures."
            ),
        },
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


def _hardware_aware(agent) -> bool:
    mode = str(getattr(getattr(agent.cfg, "experiment", None), "mode", "hardware_aware") or "hardware_aware")
    return mode.strip().lower().replace("-", "_") not in {"origin", "baseline"}


def _append_runtime_issue(
    node: SearchNode,
    *,
    category: str,
    owner: str,
    evidence: str,
    repair_instruction: str,
    severity: str = "critical",
) -> None:
    append_review_issue(
        node,
        ReviewIssue(
            source="runtime_validator",
            severity=severity,
            category=category,
            owner=owner,
            evidence=evidence,
            repair_instruction=repair_instruction,
        ),
    )

metric_direction_func_spec = FunctionSpec(
    name="determine_metric_direction",
    json_schema={
        "type": "object",
        "properties": {
            "lower_is_better": {
                "type": "boolean",
                "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE, RMSE, MAE, loss, error rate), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy, F1 score, AUC, precision, recall, Jaccard score, IoU).",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of why this metric direction is chosen based on the task's evaluation metric description.",
            },
        },
        "required": [
            "lower_is_better",
            "reasoning",
        ],
    },
    description="Determine whether the evaluation metric should be minimized or maximized based on the task description.",
)


def determine_metric_direction(agent) -> None:
    logger.info("=" * 80)
    logger.info("Starting pre-determination of metric optimization direction...")
    logger.info("=" * 80)

    prompt = f"""You are analyzing a machine learning competition task. Your task is to determine whether the evaluation metric should be minimized or maximized.

    **IMPORTANT: Focus on the EVALUATION section in the task description, which specifies the metric used to score submissions.**

    Task Description:
    {agent.task_desc}

    Based on the evaluation metric mentioned in the task description, determine:
    - If the metric should be MINIMIZED (lower is better), set lower_is_better to TRUE.
    Examples: MSE, RMSE, MAE, Cross-Entropy Loss, Log Loss, Error Rate
    - If the metric should be MAXIMIZED (higher is better), set lower_is_better to FALSE.
    Examples: Accuracy, F1 Score, AUC-ROC, Precision, Recall, Jaccard Score, IoU, mAP

    **Pay special attention to:**
    1. The "Evaluation" or "Metric" section in the task description
    2. Common metric conventions (e.g., accuracy is always maximized, MSE is always minimized)
    3. Whether the metric measures error/loss (minimize) or performance/quality (maximize)

    Provide clear reasoning based on the evaluation metric specified in the task.
    """

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            if attempt == 1:
                logger.info(f"Attempt {attempt}/{max_retries} to determine metric direction...")
            else:
                logger.info(f"Retry attempt {attempt}/{max_retries} to determine metric direction...")
            response = cast(
                dict,
                query(
                    system_message=prompt,
                    user_message=None,
                    func_spec=metric_direction_func_spec,
                    model=agent.acfg.feedback.model,
                    temperature=agent.acfg.feedback.temp,
                    cfg=agent.cfg,
                    stage_name="feedback",
                    context_cache_role="result_parser",
                ),
            )

            lower_is_better = response["lower_is_better"]
            agent.metric_maximize = not lower_is_better
            reasoning = response.get("reasoning", "")
            agent.metric_maximize_reasoning = reasoning

            logger.info("=" * 80)
            logger.info("Pre-determination completed successfully:")
            logger.info(f"  - lower_is_better = {lower_is_better}")
            logger.info(f"  - maximize = {agent.metric_maximize}")
            logger.info(f"  - Reasoning: {reasoning}")
            logger.info("=" * 80)
            logger.info(f"All subsequent nodes MUST use maximize={agent.metric_maximize}, otherwise they will be marked as buggy")
            logger.info("=" * 80)
            try:
                from utils.pipeline_logging import log_pipeline_event

                log_pipeline_event(
                    agent,
                    "metric_direction_determined",
                    payload={
                        "lower_is_better": lower_is_better,
                        "metric_maximize": agent.metric_maximize,
                        "reasoning": reasoning,
                    },
                )
            except Exception:
                pass
            return

        except Exception as e:
            logger.warning(f"Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                logger.info("Retrying in a moment...")
                time.sleep(1)
            else:
                logger.error("=" * 80)
                logger.error(f"All {max_retries} attempts failed. Last error: {e}")
                logger.error("Using default value maximize=True (assuming higher is better)")
                logger.error("=" * 80)
                agent.metric_maximize = True
                agent.metric_maximize_reasoning = "Default: assuming higher is better (most common case)"
                try:
                    from utils.pipeline_logging import log_pipeline_event

                    log_pipeline_event(
                        agent,
                        "metric_direction_determined",
                        payload={
                            "lower_is_better": False,
                            "metric_maximize": True,
                            "reasoning": agent.metric_maximize_reasoning,
                            "fallback": True,
                        },
                    )
                except Exception:
                    pass


def get_review_func_spec(use_memory: bool) -> FunctionSpec:
    properties = {
        "is_bug": {
            "type": "boolean",
            "description": "true if the output log shows that the execution failed or has some bug, otherwise false. "
                           "Focus only on actual execution errors, exceptions, crashes, or unusable outputs. "
                           "A weak/non-improving metric or underfit/overfit model is not a bug.",
        },
        "summary": {
            "type": "string",
            "description": "Provide a concise summary (2-3 sentences) of the execution outcome. "
                           "If successful, describe the key empirical results. "
                           "If failed, describe the error encountered. "
                           "Focus on observations only — do not include suggestions for improvement.",
        },
        "metric": {
            "type": ["number", "null"],
            "description": "If the code ran successfully, report the value of the validation metric. Otherwise, leave it null.",
        },
        "lower_is_better": {
            "type": "boolean",
            "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy).",
        },
        "issues": {
            "type": "array",
            "items": _RUNTIME_ISSUE_SCHEMA,
            "description": (
                "Stage-owned execution issues. Successful but weak/non-improving model quality "
                "must be a warning; return [] when execution succeeded without issues."
            ),
        },
    }
    required = ["is_bug", "summary", "metric", "lower_is_better", "issues"]
    if use_memory:
        properties["code_summary"] = {
            "type": "string",
            "description": "Write a summary including the methods used in each stage of the code, such as data preprocessing, feature engineering, model architecture, etc.",
        }
        required.append("code_summary")
    return FunctionSpec(
        name="submit_review",
        json_schema={"type": "object", "properties": properties, "required": required},
        description="Submit a review evaluating the output of the training script.",
    )


def _build_introduction(agent) -> str:
    use_memory = getattr(agent.acfg, "use_global_memory", False)
    intro = (
        "You are a Kaggle grandmaster attending a competition. "
        "You have written code to solve this task and now need to evaluate the output of the code execution. "
        "You should determine if there were any bugs as well as report the empirical findings.\n\n"
        "You MUST respond with a JSON object containing ALL of the following fields:\n"
        "- \"is_bug\": (boolean) true if execution failed or has bugs, false otherwise. Must be a JSON boolean (true/false), NOT a string.\n"
        "- \"summary\": (string) A concise 2-3 sentence summary of the execution outcome.\n"
        "- \"metric\": (number or null) The validation metric value as a raw JSON number (e.g. 0.9995), NOT a string. If failed, use null.\n"
        "- \"lower_is_better\": (boolean) true if the metric should be minimized, false if maximized. Must be a JSON boolean (true/false), NOT a string.\n"
        "- \"issues\": (array) Stage-owned issues with source, severity, category, owner, evidence, and repair_instruction. Use [] on success.\n"
        "  Optimizer/scheduler/batch/training-loop/metric/submission issues belong to training_evaluation; cross-stage interface failures belong to integration.\n"
        "  A completed run with a finite metric and valid submission is NOT buggy merely because it underfits, overfits, fails to improve its parent, or uses resources poorly. Record those findings as warnings and preserve the metric.\n"
    )
    if use_memory:
        intro += (
            "- \"code_summary\": (string) A concise method summary of the code, covering key parts such as "
            "data preprocessing, feature engineering, model architecture/training, and validation strategy.\n"
        )
    intro += "\nDo NOT omit any field."
    return intro



def _check_submission_file(agent, node: SearchNode) -> bool:
    correct_path = agent.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv"

    if not correct_path.exists():
        wrong_path = agent.cfg.workspace_dir / f"submission_{node.id}.csv"
        if wrong_path.exists():
            correct_path.parent.mkdir(parents=True, exist_ok=True)
            wrong_path.rename(correct_path)
            logger.warning(f" {wrong_path} are moved to {correct_path}")

    return correct_path.exists()


def _save_code_summary(agent, node: SearchNode, response: dict):
    use_memory = getattr(agent.acfg, "use_global_memory", False)
    if not use_memory:
        node.code_summary = None
        return
    if "code_summary" in response and response["code_summary"]:
        node.code_summary = response["code_summary"]
        logger.info(f"Saved code summary for node {node.id}")
    else:
        logger.warning(f"Node {node.id} missing code_summary in response")
        node.code_summary = None


def _determine_buggy(node: SearchNode, response: dict, has_csv_submission: bool):
    """Set execution validity from hard result signals, not quality opinions."""
    failure_reasons = []
    if response["is_bug"]:
        failure_reasons.append("execution error detected")
    if any(
        issue.get("severity") == "critical"
        and not _is_advisory_quality_issue(issue)
        for issue in (node.review_issues or [])
    ):
        failure_reasons.append("critical runtime failure classified")
    if node.exc_type is not None:
        failure_reasons.append(f"exception raised: {node.exc_type}")
    try:
        has_usable_metric = response["metric"] is not None and math.isfinite(
            float(response["metric"])
        )
    except (TypeError, ValueError):
        has_usable_metric = False
    if not has_usable_metric:
        response["metric"] = None
        failure_reasons.append("no finite metric value reported")
        _append_runtime_issue(
            node,
            category="missing_metric",
            owner="training_evaluation",
            evidence="Execution output did not contain a usable validation metric.",
            repair_instruction="Compute the exact task metric and print it in the required final score format.",
        )
    if not has_csv_submission:
        failure_reasons.append("submission file not found")
        _append_runtime_issue(
            node,
            category="missing_submission",
            owner="training_evaluation",
            evidence="Execution did not create the required submission CSV.",
            repair_instruction="Generate ./submission/submission.csv from real test inference using the required columns.",
        )

    node.is_buggy = len(failure_reasons) > 0
    if node.is_buggy:
        logger.warning(f"Node {node.id} marked as buggy: {'; '.join(failure_reasons)}")


def _completed_execution_response(
    node: SearchNode,
    response: dict,
    has_csv_submission: bool,
) -> bool:
    """Return whether the parser response describes a usable completed result."""
    if response.get("is_bug") or node.exc_type is not None or not has_csv_submission:
        return False
    try:
        return math.isfinite(float(response.get("metric")))
    except (TypeError, ValueError):
        return False


def _normalize_runtime_issue_for_result(
    issue: ReviewIssue,
    *,
    completed_successfully: bool,
) -> ReviewIssue:
    """Keep advisory feedback from contradicting a successful result contract."""
    if (
        not completed_successfully
        or issue.severity != "critical"
        or not _is_advisory_quality_issue(issue)
    ):
        return issue
    logger.warning(
        "Downgrading runtime issue %s from critical to warning because execution "
        "completed with a finite metric and submission.",
        issue.category,
    )
    return ReviewIssue(
        source=issue.source,
        severity="warning",
        category=issue.category,
        owner=issue.owner,
        evidence=issue.evidence,
        repair_instruction=issue.repair_instruction,
    )


def _validate_format_with_retry(agent, node: SearchNode):
    exp_id = agent.cfg.exp_name.split("_")[2]
    submission_path = agent.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv"

    status, res = _validate_submission_with_retry(
        exp_id=exp_id,
        submission_path=submission_path,
        cfg=agent.cfg,
        max_attempts=2,
        sample_path=None,
    )

    if status:
        if not res['is_valid']:
            logger.warning(f"[validate] node {node.id}: invalid after retry attempts.")
            node.is_valid = False
            node.is_buggy = True
            node._term_out.append(f"\n{res['result']}")
            node.analysis = f"FORMAT_ERROR: Execution succeeded but submission file failed format validation.\n\nDetails:\n{res['result']}"
            _append_runtime_issue(
                node,
                category="invalid_submission",
                owner="training_evaluation",
                evidence=str(res["result"]),
                repair_instruction="Match the sample submission columns, row count, order, and output path exactly.",
            )
        else:
            _check_content_quality(agent, node, submission_path)
    else:
        logger.error(f"An unexpected error occurred: {res}, skip this stage.")
        logger.info(f"Node {node.id} format validation passed. Now checking content quality...")
        content_valid, content_error = validate_submission_content_quality(
                submission_path=submission_path,
                sample_path=None,
                constant_threshold=0.95,
            )

        if not content_valid:
            _mark_content_quality_failure(node, content_error)
        else:
            logger.info(f"[validate] node {node.id}: valid")
            node.is_valid = True


def _validate_format_simple(agent, node: SearchNode):
    exp_id = agent.cfg.exp_name.split("_")[2]
    submission_path = agent.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv"

    status, res = call_validate(exp_id=exp_id, submission_path=submission_path)
    if status:
        if not res['is_valid']:
            logger.warning(f"[validate] node {node.id}: invalid.")
            node.is_valid = False
            node.is_buggy = True
            node._term_out.append(f"\n{res['result']}")
            node.analysis = f"FORMAT_ERROR: Execution succeeded but submission file failed format validation.\n\nDetails:\n{res['result']}"
            _append_runtime_issue(
                node,
                category="invalid_submission",
                owner="training_evaluation",
                evidence=str(res["result"]),
                repair_instruction="Match the sample submission columns, row count, order, and output path exactly.",
            )
        else:
            _check_content_quality(agent, node, submission_path)
    else:
        logger.error(f"An unexpected error occurred: {res}, skip this stage.")


def _check_content_quality(agent, node: SearchNode, submission_path):
    logger.info(f"Node {node.id} format validation passed. Now checking content quality...")
    content_valid, content_error = validate_submission_content_quality(
            submission_path=submission_path,
            sample_path=None,
            constant_threshold=0.95,
        )

    if not content_valid:
        _mark_content_quality_failure(node, content_error)
    else:
        logger.info(f"✅ Node {node.id} passed both format and content quality checks.")
        node.is_valid = True


def _mark_content_quality_failure(node: SearchNode, content_error):
    logger.warning(f"Node {node.id} is marked as buggy due to content quality check failure.")
    node.is_valid = False
    node.is_buggy = True
    error_message = (
        "Submission format is correct, but content quality check FAILED:\n\n"
        f"{content_error}\n\n"
        "🚨 CRITICAL: All predictions must come from actual model inference.\n"
        "You must:\n"
        "1. Load each test sample\n"
        "2. Preprocess it with the same transformations as training\n"
        "3. Run model.predict() / model.forward() on the sample\n"
        "4. Use the model's output as the prediction\n\n"
        "Filling submissions with constants, placeholders, or dummy values is STRICTLY FORBIDDEN."
    )
    node._term_out.append(f"\n{error_message}")
    node.analysis = f"CONTENT_QUALITY_ERROR: This previous solution runs without bugs and has correct format, but failed content quality check.\n\nDetails:\n{content_error}"
    _append_runtime_issue(
        node,
        category="submission_content",
        owner="training_evaluation",
        evidence=str(content_error),
        repair_instruction="Produce non-placeholder predictions through the trained model's real test inference path.",
    )


def _validate_metric_direction(agent, node: SearchNode, response: dict):
    returned_maximize = not response["lower_is_better"]
    if agent.metric_maximize is not None and returned_maximize != agent.metric_maximize:
        logger.error("=" * 80)
        logger.error(f"METRIC DIRECTION MISMATCH for Node {node.id}!")
        logger.error(f"  - Returned lower_is_better = {response['lower_is_better']} (maximize={returned_maximize})")
        logger.error(f"  - Pre-determined maximize = {agent.metric_maximize}")
        logger.error(f"  - Marking this node as BUGGY, will NOT update top candidates")
        logger.error("=" * 80)
        node.is_buggy = True
        node.metric = WorstMetricValue()
        node.analysis = (
            f"{node.analysis}\n\n[ERROR] Metric direction mismatch detected:\n"
            f"- Returned lower_is_better={response['lower_is_better']} (maximize={returned_maximize})\n"
            f"- Expected maximize={agent.metric_maximize}\n"
            f"- Pre-determination reasoning: {agent.metric_maximize_reasoning or 'N/A'}\n"
            f"This node is marked as buggy and will not be considered for best/top candidates."
        )
        _append_runtime_issue(
            node,
            category="metric_direction",
            owner="training_evaluation",
            evidence=(
                f"Execution parser returned maximize={returned_maximize}, but the task contract expects "
                f"maximize={agent.metric_maximize}."
            ),
            repair_instruction="Compute and report the task metric with the correct optimization direction.",
        )
    else:
        logger.info(f"Node {node.id} metric direction validated: maximize={agent.metric_maximize}")
        node.metric = MetricValue(
            response["metric"], maximize=agent.metric_maximize
        )


def _check_data_leakage(agent, node: SearchNode, response: dict):
    if not (agent.acfg.check_data_leakage and should_check_data_leakage(agent, node)):
        return

    logger.warning(
        f"Node {node.id} triggers data leakage check due to extreme metric value: {node.metric.value}"
    )

    leakage_result = data_leakage_agent.run(agent, node)

    if leakage_result["has_leakage"] and leakage_result["confidence"] in ["high", "medium"]:
        logger.error(
            f"⚠️  Node {node.id} detected data leakage with {leakage_result['confidence']} confidence. "
            f"Marking as buggy and resetting metric."
        )
        node.is_buggy = True
        node.metric = WorstMetricValue()
        node.analysis = (
            f"⚠️ DATA LEAKAGE DETECTED (Confidence: {leakage_result['confidence'].upper()})\n\n"
            f"{leakage_result['reason']}\n\n"
            f"The validation metric was {response['metric']:.4f} which is unrealistically extreme due to data leakage. "
            f"To fix this issue, you need to:\n"
            f"1. Carefully review the train/validation split logic\n"
            f"2. Ensure no validation/test data is used during training\n"
            f"3. Check that feature engineering only uses training data statistics\n"
            f"4. Verify data augmentation doesn't leak validation samples\n"
            f"5. Ensure proper temporal/group separation if applicable"
        )
        _append_runtime_issue(
            node,
            category="data_leakage",
            owner="model_design",
            evidence=str(leakage_result["reason"]),
            repair_instruction="Fit preprocessing on training data only and restore dependency-aware validation splitting.",
        )
        logger.info(f"Updated node.analysis with leakage detection details for debugging")
    else:
        if leakage_result["has_leakage"]:
            logger.info(
                f"Node {node.id} has potential leakage but confidence is low. Not marking as buggy."
            )
        else:
            logger.info(
                f"Node {node.id} extreme value is justified: {leakage_result['reason']}"
            )


def _save_to_global_memory(agent, node: SearchNode):
    if getattr(agent, "global_memory", None) and not node.is_buggy and node.metric and node.metric.value is not None:
        try:
            parent_node = node.parent
            agent.global_memory.save_node(node, parent_node)
        except Exception as e:
            logger.warning(f"[AgentSearch] Failed to save node {node.id} to global memory: {e}")


def _completed_metric_from_contract(agent, node: SearchNode) -> float | None:
    """Read a finite final score only from a completed job with a submission."""
    if node.exc_type is not None:
        return None
    matches = _FINAL_VALIDATION_SCORE_RE.findall(node.term_out)
    if not matches:
        return None
    try:
        metric = float(matches[-1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(metric) or not _check_submission_file(agent, node):
        return None
    return metric


def _is_advisory_quality_issue(issue: dict | ReviewIssue) -> bool:
    if isinstance(issue, ReviewIssue):
        raw_category = issue.category
    else:
        raw_category = issue.get("category")
    category = re.sub(
        r"[^a-z0-9]+",
        "_",
        str(raw_category or "").lower(),
    ).strip("_")
    return any(marker in category for marker in _ADVISORY_QUALITY_CATEGORY_MARKERS)


def _recover_completed_quality_gate_result(agent, node: SearchNode) -> bool:
    """Restore old nodes incorrectly failed only for advisory model quality."""
    if node.review_status == "rejected":
        return False
    critical_issues = [
        issue
        for issue in (node.review_issues or [])
        if issue.get("severity") == "critical"
    ]
    if not critical_issues or any(
        not _is_advisory_quality_issue(issue) for issue in critical_issues
    ):
        return False
    metric = _completed_metric_from_contract(agent, node)
    if metric is None:
        return False

    node.review_issues = [
        {**issue, "severity": "warning"}
        if issue.get("severity") == "critical" and _is_advisory_quality_issue(issue)
        else issue
        for issue in (node.review_issues or [])
    ]
    node.is_buggy = False
    node.metric = MetricValue(metric, maximize=agent.metric_maximize)
    quality_summary = str(node.analysis or "").strip()
    node.analysis = (
        f"{quality_summary}\n\n" if quality_summary else ""
    ) + "Execution succeeded; model-quality findings are advisory and the metric is preserved."
    _validate_format_with_retry(agent, node)
    if node.is_buggy:
        return False
    logger.warning(
        "[parse] recovered completed node %s from an advisory quality-gate failure",
        node.id,
    )
    _save_to_global_memory(agent, node)
    return True


def _recover_completed_result_without_llm(agent, node: SearchNode) -> bool:
    """Recover a completed result from the mandatory final-score contract.

    The language-model review is enrichment, not an execution commit point: a
    temporary parser outage must never turn an already-completed scheduler job
    into a failed search-tree node.
    """
    metric = _completed_metric_from_contract(agent, node)
    if metric is None:
        return False

    node.is_buggy = False
    node.review_issues = [
        issue
        for issue in (node.review_issues or [])
        if issue.get("category") != "result_parser_failure"
    ]
    node.metric = MetricValue(metric, maximize=agent.metric_maximize)
    node.analysis = (
        "Execution completed and emitted the required final validation score; "
        "language-model result parsing was unavailable."
    )
    _append_runtime_issue(
        node,
        category="result_parser_unavailable",
        owner="integration",
        evidence="The execution result parser failed after retries; the final-score contract was recovered locally.",
        repair_instruction="Restore the result-parser service for richer execution summaries.",
        severity="warning",
    )
    _validate_format_with_retry(agent, node)
    if node.is_buggy:
        return False
    logger.warning(
        "[parse] recovered completed node %s from Final Validation Score without language-model parsing",
        node.id,
    )
    _save_to_global_memory(agent, node)
    return True


def run(agent, node: SearchNode, exec_result: ExecutionResult) -> SearchNode:
    max_retries = 3
    for retry_idx in range(max_retries):
        try:
            logger.info(f"Agent is parsing execution results for node {node.id}")

            node.absorb_exec_result(exec_result)

            introduction = _build_introduction(agent)
            prompt = {
                "Introduction": introduction,
                "Implementation": wrap_code(node.code),
                "Execution output": wrap_code(node.term_out, lang=""),
            }
            stable_prompt = {"Introduction": prompt["Introduction"]}
            dynamic_prompt = {
                key: value for key, value in prompt.items() if key != "Introduction"
            }

            response = cast(
                dict,
                query(
                    system_message=prompt,
                    user_message=None,
                    func_spec=get_review_func_spec(getattr(agent.acfg, "use_global_memory", False)),
                    model=agent.acfg.feedback.model,
                    temperature=agent.acfg.feedback.temp,
                    cfg=agent.cfg,
                    stage_name="feedback",
                    context_cache_role="result_parser",
                    context_cache_stable_prefix=stable_prompt,
                    context_cache_dynamic_system_message=dynamic_prompt,
                ),
            )

            # Gemini structured output may omit required fields; fill defaults
            response.setdefault("is_bug", True)
            response.setdefault("summary", "No summary returned by model.")
            response.setdefault("metric", None)
            response.setdefault("lower_is_better",
                                not agent.metric_maximize if agent.metric_maximize is not None else False)
            response.setdefault("issues", [])

            metric_val = response.get("metric")
            if not isinstance(metric_val, (int, float)):
                try:
                    response["metric"] = float(metric_val)
                except (TypeError, ValueError):
                    response["metric"] = None

            for bool_field in ("is_bug", "lower_is_better"):
                v = response.get(bool_field)
                if isinstance(v, str):
                    response[bool_field] = v.strip().lower() not in ("false", "0", "no", "")

            has_csv_submission = _check_submission_file(agent, node)

            node.analysis = response["summary"]
            _save_code_summary(agent, node, response)
            completed_successfully = _completed_execution_response(
                node,
                response,
                has_csv_submission,
            )
            try:
                for issue in normalize_review_issues(
                    response.get("issues") or [],
                    default_source="runtime",
                    hardware_aware=_hardware_aware(agent),
                ):
                    append_review_issue(
                        node,
                        _normalize_runtime_issue_for_result(
                            issue,
                            completed_successfully=completed_successfully,
                        ),
                    )
            except Exception as exc:
                logger.warning("Ignoring malformed runtime issue classification for node %s: %s", node.id, exc)
            _determine_buggy(node, response, has_csv_submission)

            if not node.is_buggy:
                _validate_format_with_retry(agent, node)

            if node.is_buggy:
                node.metric = WorstMetricValue()
                if not any(issue.get("severity") == "critical" for issue in node.review_issues):
                    evidence = (
                        f"{node.exc_type}: {node.exc_info}" if node.exc_type is not None
                        else str(node.analysis or "Execution failed without a usable stage classification.")
                    )
                    _append_runtime_issue(
                        node,
                        category="runtime_failure",
                        owner="unclassified",
                        evidence=evidence,
                        repair_instruction="Diagnose the complete script and repair the runtime failure.",
                    )
            else:
                _validate_metric_direction(agent, node, response)
                _check_data_leakage(agent, node, response)

            status = "FAIL" if node.is_buggy else "PASS"
            metric_val = node.metric.value if node.metric else None
            logger.info(f"[parse] node {node.id}: {status} | metric={metric_val}")
            try:
                from utils.pipeline_logging import log_pipeline_event, record_pipeline_node_action

                payload = {
                    "status": status,
                    "metric": metric_val,
                    "is_buggy": node.is_buggy,
                    "is_valid": node.is_valid,
                    "exec_time": node.exec_time,
                    "exc_type": node.exc_type,
                    "summary": node.analysis,
                }
                log_pipeline_event(agent, "execution_result_parsed", node=node, payload=payload)
                record_pipeline_node_action(agent, node, "execution_result_parsed", payload=payload)
            except Exception:
                pass

            _save_to_global_memory(agent, node)

            return node
        except Exception as e:
            logger.warning(f"[parse] tool call failed: {e}")
            continue

    if _recover_completed_result_without_llm(agent, node):
        return node

    logger.error(f"All {max_retries} parse attempts failed for node {node.id}, marking as buggy")
    node.is_buggy = True
    node.metric = WorstMetricValue()
    node.analysis = "Execution result parsing failed after multiple attempts."
    _append_runtime_issue(
        node,
        category="result_parser_failure",
        owner="unclassified",
        evidence=node.analysis,
        repair_instruction="Use whole-script debugging because the runtime result could not be classified.",
    )
    return node

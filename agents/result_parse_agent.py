import logging
import re
import time
from typing import Any, cast

from llm import FunctionSpec, query
from engine.search_node import SearchNode
from engine.executor import ExecutionResult
from localml_scheduler.runtime_environment import validate_generated_training_code
from utils.metric import MetricValue, WorstMetricValue
from utils.response import trim_long_string, wrap_code
from engine.validation import call_validate, _validate_submission_with_retry, validate_submission_content_quality
from agents import data_leakage_agent
from agents.triggers import should_check_data_leakage

logger = logging.getLogger("MLEvolve")

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
                    cfg=agent.cfg
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
                           "Focus only on actual execution errors, exceptions, or crashes.",
        },
        "summary": {
            "type": "string",
            "description": "Provide a concise summary (2-3 sentences) of the execution outcome. "
                           "If successful, describe the key empirical results. "
                           "If failed, describe the error encountered. "
                           "Focus on observations only — do not include suggestions for improvement.",
        },
        "metric": {
            "type": "number",
            "description": "If the code ran successfully, report the value of the validation metric. Otherwise, leave it null.",
        },
        "lower_is_better": {
            "type": "boolean",
            "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy).",
        },
    }
    required = ["is_bug", "summary", "metric", "lower_is_better"]
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


def _determine_buggy(node: SearchNode, response: dict, has_csv_submission: bool) -> list[str]:
    failure_reasons = []
    if response["is_bug"]:
        failure_reasons.append("execution error detected")
    if node.exc_type is not None:
        failure_reasons.append(f"exception raised: {node.exc_type}")
    if response["metric"] is None:
        failure_reasons.append("no metric value reported")
    if not has_csv_submission:
        failure_reasons.append("submission file not found")

    node.is_buggy = len(failure_reasons) > 0
    if node.is_buggy:
        logger.warning(f"Node {node.id} marked as buggy: {'; '.join(failure_reasons)}")
    return failure_reasons


def _post_validation_failure_reasons(node: SearchNode) -> list[str]:
    analysis = str(getattr(node, "analysis", "") or "")
    lowered = analysis.lower()
    if "format_error" in lowered or "format validation" in lowered:
        return ["submission format validation failed"]
    if "content_quality_error" in lowered or "content quality check failed" in lowered:
        return ["submission content quality validation failed"]
    if "data leakage" in lowered:
        return ["data leakage validation failed"]
    if getattr(node, "is_buggy", False):
        return ["post-execution validation failed"]
    return []


def _build_structured_bug_report(
    node: SearchNode,
    response: dict[str, Any],
    failure_reasons: list[str],
    *,
    has_csv_submission: bool,
) -> tuple[str, str]:
    compatibility = validate_generated_training_code(node.code or "", stage="result_parse")
    critical_issues = [
        issue for issue in compatibility.get("issues", [])
        if issue.get("severity") == "critical"
    ]
    category, root_cause, repair_hint = _diagnose_failure(
        node=node,
        response=response,
        critical_issues=critical_issues,
        has_csv_submission=has_csv_submission,
    )
    evidence = _failure_evidence(node, response, critical_issues)

    lines = [
        f"failure_category: {category}",
        f"root_cause: {root_cause}",
        f"missing_submission: {not has_csv_submission}",
    ]
    if not has_csv_submission and category != "missing_submission":
        lines.append("missing_submission_role: consequence of the earlier runtime failure")
    if node.exc_type:
        lines.append(f"exception_type: {node.exc_type}")
    if node.exc_info:
        lines.append(f"exception_info: {trim_long_string(str(node.exc_info), threshold=500, k=240)}")
    if failure_reasons:
        lines.append("failure_reasons: " + "; ".join(failure_reasons))
    if evidence:
        lines.extend(["evidence:", evidence])
    return "\n".join(lines).strip(), repair_hint


def _diagnose_failure(
    *,
    node: SearchNode,
    response: dict[str, Any],
    critical_issues: list[dict[str, Any]],
    has_csv_submission: bool,
) -> tuple[str, str, str]:
    combined = "\n".join(
        str(part or "")
        for part in (
            response.get("summary"),
            node.analysis,
            node.term_out,
            node.exc_type,
            node.exc_info,
        )
    )
    lowered = combined.lower()

    for issue in critical_issues:
        category = str(issue.get("category") or "")
        if category == "bf16_numpy_conversion":
            return (
                "bf16_numpy_conversion",
                "Validation or metric code converted BF16 tensors directly to NumPy, which this PyTorch/NumPy stack does not support.",
                str(issue.get("repair_hint") or "Cast validation predictions to float32 before CPU/NumPy conversion."),
            )
        if category == "low_precision_numpy_export":
            return (
                "low_precision_numpy_export",
                "Validation, metric, or submission code converted low-precision predictions/logits directly to NumPy without a float32 export boundary.",
                str(issue.get("repair_hint") or "Cast prediction/logit/probability tensors to float32 before CPU/NumPy conversion."),
            )
        if category == "invalid_torch_scheduler_argument":
            return (
                "invalid_torch_scheduler_argument",
                str(issue.get("message") or "A PyTorch scheduler was called with an unsupported keyword argument."),
                str(issue.get("repair_hint") or "Use the installed PyTorch scheduler signature."),
            )
        if category == "syntax_error":
            return (
                "syntax_error",
                str(issue.get("message") or "Generated code failed Python syntax validation."),
                str(issue.get("repair_hint") or "Fix Python syntax before execution."),
            )

    if "unsupported scalartype bfloat16" in lowered or "got unsupported scalartype bfloat16" in lowered:
        return (
            "bf16_numpy_conversion",
            "Validation failed because BF16 predictions/logits were converted to CPU/NumPy without first casting to float32.",
            "Cast predictions/logits/probabilities to float32 before `.cpu().numpy()` and run metric/submission export outside autocast.",
        )
    low_precision_error_markers = (
        "unsupported scalartype",
        "unsupported dtype",
        "unsupported scalar type",
        "float8",
        "fp8",
        "nvfp4",
        "mxfp4",
        "mxfp8",
        "low precision",
        "dtype mismatch",
    )
    if any(marker in lowered for marker in low_precision_error_markers) and (
        ".cpu().numpy" in lowered or "numpy" in lowered or "log_loss" in lowered or "sklearn" in lowered
    ):
        return (
            "low_precision_numpy_export",
            "Validation failed because low-precision predictions/logits were passed to NumPy/sklearn/submission export without first casting to float32.",
            "Use `tensor.detach().to(torch.float32).cpu().numpy()` for prediction/logit/probability export and keep low precision limited to forward/loss computation.",
        )
    if "cosineannealinglr" in lowered and ("unexpected keyword argument" in lowered or "t_eta" in lowered):
        return (
            "invalid_torch_scheduler_argument",
            "The training script called torch.optim.lr_scheduler.CosineAnnealingLR with a keyword unsupported by the installed PyTorch signature.",
            "Use `eta_min=...` and `T_max=...`; remove invalid keywords such as `T_eta_min` or `T_eta`.",
        )
    if "format_error" in lowered or "format validation" in lowered:
        return (
            "submission_format_validation",
            "Execution completed, but the generated submission failed the post-run format validator.",
            "Write the submission with exactly the expected sample-submission columns, row count, ID order, and parseable prediction values.",
        )
    if "content_quality_error" in lowered or "content quality check failed" in lowered:
        return (
            "submission_content_quality",
            "Execution produced a correctly shaped submission, but the post-run content validator rejected it as low-quality or non-inference output.",
            "Generate predictions from actual model inference on each test sample instead of constants, placeholders, shuffled rows, or dummy values.",
        )
    if "data leakage detected" in lowered:
        return (
            "data_leakage",
            "Post-run validation marked the node buggy because the metric indicates likely validation/test leakage.",
            "Fix the train/validation split and feature engineering so validation/test rows and statistics are not used during training.",
        )
    if node.exc_type:
        return (
            "runtime_exception",
            f"Execution raised {node.exc_type} before a valid metric/submission was produced.",
            "Use the traceback evidence to fix the first runtime exception before changing model design.",
        )
    if not has_csv_submission:
        return (
            "missing_submission",
            "The script finished parsing without creating the required submission file.",
            "Ensure the script writes `./submission/submission.csv` or the node-specific submission path with the required columns after test inference.",
        )
    return (
        "execution_failed",
        str(response.get("summary") or "Execution failed before a valid metric was produced."),
        "Fix the concrete execution failure shown in the evidence before attempting score improvements.",
    )


def _failure_evidence(
    node: SearchNode,
    response: dict[str, Any],
    critical_issues: list[dict[str, Any]],
) -> str:
    parts: list[str] = []
    for issue in critical_issues[:3]:
        if issue.get("evidence"):
            parts.append(f"- compatibility: {issue['evidence']}")
    traceback_excerpt = _traceback_excerpt(node.term_out)
    if traceback_excerpt:
        parts.append("- traceback/output:\n" + traceback_excerpt)
    elif getattr(node, "analysis", None):
        parts.append("- parser_analysis: " + str(node.analysis))
    elif response.get("summary"):
        parts.append("- parser_summary: " + str(response.get("summary")))
    return trim_long_string("\n".join(parts), threshold=1800, k=850)


def _traceback_excerpt(text: str) -> str:
    text = str(text or "")
    if not text.strip():
        return ""
    match = re.search(r"Traceback \(most recent call last\):.*?(?=\n\n|\Z)", text, re.DOTALL)
    if match:
        return trim_long_string(match.group(0), threshold=1300, k=620)
    lines = [
        line for line in text.splitlines()
        if any(token in line for token in ("Error", "Exception", "Traceback", "TypeError", "RuntimeError"))
    ]
    return trim_long_string("\n".join(lines[-12:]), threshold=1300, k=620)


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
    if agent.global_memory and not node.is_buggy and node.metric and node.metric.value is not None:
        try:
            parent_node = node.parent
            agent.global_memory.save_node(node, parent_node)
        except Exception as e:
            logger.warning(f"[AgentSearch] Failed to save node {node.id} to global memory: {e}")


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

            response = cast(
                dict,
                query(
                    system_message=prompt,
                    user_message=None,
                    func_spec=get_review_func_spec(getattr(agent.acfg, "use_global_memory", False)),
                    model=agent.acfg.feedback.model,
                    temperature=agent.acfg.feedback.temp,
                    cfg=agent.cfg
                ),
            )

            # Gemini structured output may omit required fields; fill defaults
            response.setdefault("is_bug", True)
            response.setdefault("summary", "No summary returned by model.")
            response.setdefault("metric", None)
            response.setdefault("lower_is_better",
                                not agent.metric_maximize if agent.metric_maximize is not None else False)

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
            failure_reasons = _determine_buggy(node, response, has_csv_submission)

            if not node.is_buggy:
                _validate_format_with_retry(agent, node)

            if node.is_buggy:
                if not failure_reasons:
                    failure_reasons.extend(_post_validation_failure_reasons(node))
                node.bug_report, node.fix_report = _build_structured_bug_report(
                    node,
                    response,
                    failure_reasons,
                    has_csv_submission=has_csv_submission,
                )
                node.metric = WorstMetricValue()
            else:
                _validate_metric_direction(agent, node, response)
                _check_data_leakage(agent, node, response)
                if node.is_buggy:
                    if not failure_reasons:
                        failure_reasons.extend(_post_validation_failure_reasons(node))
                    node.bug_report, node.fix_report = _build_structured_bug_report(
                        node,
                        response,
                        failure_reasons,
                        has_csv_submission=has_csv_submission,
                    )

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
                    "bug_report": getattr(node, "bug_report", None),
                    "fix_report": getattr(node, "fix_report", None),
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

    logger.error(f"All {max_retries} parse attempts failed for node {node.id}, marking as buggy")
    node.is_buggy = True
    node.metric = WorstMetricValue()
    node.analysis = "Execution result parsing failed after multiple attempts."
    node.bug_report = (
        "failure_category: result_parse_failed\n"
        "root_cause: Execution logs could not be parsed into structured feedback after multiple attempts.\n"
        f"exception_type: {node.exc_type or ''}\n"
        "evidence:\n"
        f"{trim_long_string(node.term_out, threshold=1800, k=850)}"
    ).strip()
    node.fix_report = "Inspect the raw execution output and fix the first runtime error before changing the solution design."
    return node

"""Post-execution validation: validate_executed_node (csv existence, metric=0.0, register success)."""

import logging

from engine.search_node import SearchNode
from utils.metric import WorstMetricValue

logger = logging.getLogger("MLEvolve")

_ZERO_METRIC_ANALYSIS = (
    "Performance is 0.0 (complete failure). This indicates fundamental issues that need debugging:\n"
    "1. Model architecture may be incorrect or not learning\n"
    "2. Data preprocessing might be broken (wrong format, normalization issues)\n"
    "3. Loss function or evaluation metric calculation may be faulty\n"
    "4. Training loop might not be updating weights properly\n"
    "5. Input data might not be loaded correctly\n\n"
    "Please review the code carefully to identify the root cause."
)


def validate_executed_node(agent, node: SearchNode):
    """Check submission.csv exists, metric=0.0 anomaly; register successful node to branch."""
    if node.is_buggy:
        _log_validation(agent, node, "skipped_buggy")
        return

    submission_path = agent.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv"
    if not submission_path.exists():
        node.is_buggy = True
        node.metric = WorstMetricValue()
        if not getattr(node, "bug_report", None):
            node.bug_report = (
                "failure_category: missing_submission\n"
                "root_cause: The script did not create the required node-specific submission file after execution.\n"
                f"missing_submission_path: {submission_path}"
            )
        if not getattr(node, "fix_report", None):
            node.fix_report = (
                "Ensure the script performs test inference and writes the required submission CSV "
                "with the competition columns before exiting."
            )
        logger.info(f"Node {node.id} did not produce a submission.csv")
        _log_validation(agent, node, "missing_submission")
        return

    if node.metric.maximize and node.metric.value == 0.0:
        original_metric = node.metric.value
        node.is_buggy = True
        node.metric = WorstMetricValue()
        node.analysis = _ZERO_METRIC_ANALYSIS
        node.bug_report = (
            "failure_category: zero_metric\n"
            "root_cause: The run produced a maximize-style metric of exactly 0.0, which is treated as a complete failure.\n"
            f"metric: {original_metric}"
        )
        node.fix_report = "Debug data loading, target encoding, loss/metric computation, and weight updates before optimizing score."
        logger.warning(
            f"Node {node.id} has metric=0.0 (maximize=True), marking as buggy for debugging."
        )
        _log_validation(agent, node, "zero_metric")
        return

    if hasattr(node, 'branch_id') and node.branch_id:
        if node.branch_id not in agent.branch_successful_nodes:
            agent.branch_successful_nodes[node.branch_id] = []
        agent.branch_successful_nodes[node.branch_id].append(node)
    _log_validation(agent, node, "valid")


def _log_validation(agent, node: SearchNode, outcome: str) -> None:
    try:
        from utils.pipeline_logging import log_pipeline_event, record_pipeline_node_action

        payload = {
            "outcome": outcome,
            "is_buggy": node.is_buggy,
            "is_valid": node.is_valid,
            "metric": node.metric.value if node.metric else None,
        }
        log_pipeline_event(agent, "validation_completed", node=node, payload=payload)
        record_pipeline_node_action(agent, node, "validation_completed", payload=payload)
    except Exception:
        pass

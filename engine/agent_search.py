"""AgentSearch: tree search coordinator; delegates to node_selection, evaluation, execution, solution_manager."""

import logging
import random
import time
from typing import Callable, List, Dict, Optional

from engine.executor import ExecutionResult
from engine.search_node import SearchNode, Journal
import utils.data_preview as data_preview
from config import Config
from utils.metric import WorstMetricValue
import threading
import json

from agents import (
    draft_agent, improve_agent, debug_agent,
    evolution_agent, fusion_agent, aggregation_agent,
    code_review_agent,
    result_parse_agent,
)
from engine import node_selection, evaluation, execution, solution_manager
from engine.conditions import is_branch_stagnant
from utils.data_preview import clean_task_desc

logger = logging.getLogger("MLEvolve")


ExecCallbackType = Callable[[str, bool], ExecutionResult]
ExecManyCallbackType = Callable[[list[tuple[str, str]]], dict[str, ExecutionResult]]

class AgentSearch:
    def __init__(
            self,
            task_desc: str,
            cfg: Config,
            journal: Journal,
            pipeline_logger=None,
    ):
        self.cfg = cfg
        self.pipeline_logger = pipeline_logger
        self.acfg = cfg.agent
        self.scfg = cfg.agent.search
        self.task_desc = clean_task_desc(task_desc, cfg)
        self.journal = journal
        self.data_preview: str | None = None
        self.current_step = 0
        self.current_node: SearchNode | None = None
        self.all_root = True
        self.virtual_root = SearchNode(parent=None, plan="(root)", code="", metric=WorstMetricValue(),
                                     stage="root")
        self.current_node_list = []
        self.journal.append(self.virtual_root)
        self.best_metric: float = None
        self.best_node: SearchNode = None
        self.search_start_time = None
        self.journal_lock = threading.Lock()
        self.save_node_lock = threading.Lock()
        self.start_time = time.time()
        self.use_stepwise_generation = True

        self.next_branch_id = 1
        self.branch_all_nodes: Dict[int, List[SearchNode]] = {}
        self.branch_successful_nodes: Dict[int, List[SearchNode]] = {}
        self.branch_node_count: Dict[int, int] = {}
        self.use_coldstart = cfg.coldstart.use_coldstart
        self.coldstart_description = cfg.coldstart.description
        self.scheduler_client = None
        self.hardware_cache_status: dict | None = None

        # Top-N candidates
        self.top_k = self.scfg.top_candidates_size
        self.top_candidates: List[SearchNode] = []

        # Performance stagnation detection
        self.best_metric_history = []
        self.stagnation_threshold = self.scfg.stagnation_window
        self.post_process_triggered = False
        self.post_process_attempts = 0
        self.max_post_process_attempts = 4
        self.improve_attempts_count = 0
        self.last_successful_improve_step = 0

        self.fusion_draft_count = 0
        self.max_fusion_drafts = cfg.agent.max_fusion_drafts

        self.metric_maximize: bool | None = None
        self.metric_maximize_reasoning: str | None = None
        result_parse_agent.determine_metric_direction(self)
        from utils.pipeline_logging import log_pipeline_event

        log_pipeline_event(
            self,
            "agent_initialized",
            payload={
                "mode": getattr(getattr(cfg, "experiment", None), "mode", None),
                "metric_maximize": self.metric_maximize,
                "metric_maximize_reasoning": self.metric_maximize_reasoning,
            },
        )

        # Global memory
        self.global_memory = None
        if self.acfg.use_global_memory:
            try:
                from agents.memory.global_memory import GlobalMemoryLayer
                memory_dir = str(self.cfg.workspace_dir / "global_memory")
                self.global_memory = GlobalMemoryLayer(
                    memory_dir=memory_dir,
                    embedding_model_path=self.acfg.memory_embedding_model_path,
                    embedding_device=self.acfg.memory_embedding_device,
                    similarity_threshold=self.acfg.memory_similarity_threshold,
                )
                logger.info(f"[AgentSearch] Global memory enabled and initialized at {memory_dir}")
            except Exception as e:
                import traceback
                logger.warning(f"[AgentSearch] Failed to initialize global memory: {e}")
                logger.debug(f"[AgentSearch] Global memory initialization traceback: {traceback.format_exc()}")
                self.global_memory = None
        else:
            logger.info("[AgentSearch] Global memory is disabled by config")

    def attach_scheduler(self, scheduler_client) -> None:
        """Expose the in-process scheduler client to stage agents for read-only context."""
        self.scheduler_client = scheduler_client
        self.hardware_cache_status = self._prewarm_current_hardware_context(scheduler_client)

    def _hardware_context_enabled(self) -> bool:
        experiment = getattr(self.cfg, "experiment", None)
        mode = str(getattr(experiment, "mode", "") or "").strip().lower().replace("-", "_")
        if mode in {"origin", "baseline"}:
            return False
        return bool(getattr(self.acfg, "hardware_context_enabled", True))

    def _prewarm_current_hardware_context(self, scheduler_client) -> dict:
        if not self._hardware_context_enabled():
            return {"ok": False, "reason": "hardware context disabled"}
        prewarm = getattr(scheduler_client, "prewarm_current_hardware_neighborhood", None)
        if not callable(prewarm):
            return {"ok": False, "reason": "scheduler client has no hardware prewarm hook"}
        try:
            configured_limit = int(getattr(self.acfg, "hardware_context_limit", 8) or 8)
        except (TypeError, ValueError):
            configured_limit = 8
        limit = max(256, configured_limit * 32)
        try:
            status = dict(prewarm(hardware_id="current", limit=limit) or {})
            status.setdefault("ok", bool(status.get("hardware_name") or status.get("hardware_id")))
            status.setdefault("requested_hardware_id", "current")
            status.setdefault("limit", limit)
            if status.get("ok"):
                logger.info(
                    "Prewarmed hardware graph context for %s with %s feature(s) via %s.",
                    status.get("hardware_name") or status.get("hardware_id") or "current hardware",
                    status.get("feature_count", 0),
                    status.get("source") or "unknown source",
                )
            else:
                logger.info("Hardware graph prewarm completed without a matching node: %s", status.get("reason"))
            return status
        except Exception as exc:
            logger.warning("Hardware graph prewarm failed; continuing with direct lookup fallback: %s", exc)
            return {"ok": False, "requested_hardware_id": "current", "limit": limit, "reason": str(exc)}

    def refresh_hardware_context(self, node: SearchNode) -> None:
        """Refresh compact hardware/profile evidence for a generated node."""
        try:
            from agents.hardware_context import refresh_node_hardware_context

            refresh_node_hardware_context(self, node)
        except Exception as exc:
            logger.debug("Skipping hardware/profile context refresh for node %s: %s", node.id, exc)

    def has_selectable_work(self) -> bool:
        return node_selection.has_selectable_work(self)

    def _discard_unfinished_node(self, node: SearchNode) -> None:
        """Remove a generated node that failed before execution/journal append."""
        parent = node.parent
        if parent is not None:
            parent.children.discard(node)
        if node.branch_id is not None:
            branch_nodes = self.branch_all_nodes.get(node.branch_id)
            if branch_nodes and node in branch_nodes:
                branch_nodes.remove(node)
                if not branch_nodes:
                    self.branch_all_nodes.pop(node.branch_id, None)
            successful_nodes = self.branch_successful_nodes.get(node.branch_id)
            if successful_nodes and node in successful_nodes:
                successful_nodes.remove(node)
                if not successful_nodes:
                    self.branch_successful_nodes.pop(node.branch_id, None)
        node.lock = False

    def _serialize_prompt(self, prompt_complete) -> str | None:
        """Serialize prompt (str or dict) to string for saving in node."""
        if prompt_complete is None:
            return None
        if isinstance(prompt_complete, str):
            return prompt_complete
        elif isinstance(prompt_complete, dict):
            return json.dumps(prompt_complete, ensure_ascii=False, indent=2)
        else:
            return str(prompt_complete)

    def update_data_preview(self):
        base_preview = data_preview.generate(self.cfg.workspace_dir)
        submission_format_warning = """

        ⚠️  CRITICAL SUBMISSION FORMAT NOTE:
        - If you see sample_submission.csv or similar files, those contain the CORRECT submission format
        - The column names in these files are the FINAL AUTHORITY for submission format
        - Always use the column names from the actual sample submission files
        """
        self.data_preview = base_preview + submission_format_warning

    def is_root(self, node: SearchNode):
        return node.id is self.virtual_root.id

    def _run_single_step(
        self,
        parent_node: SearchNode,
        exec_callback: ExecCallbackType,
        execute_immediately: bool = True,
        init_solution_path: Optional[str] = None,
    ):
        """Run one search step: select action (draft/debug/improve), execute, parse, validate."""
        result_node = None
        _root = False

        if not parent_node.is_terminal:
            try:
                if self.is_root(parent_node):
                    if parent_node.reached_child_limit(scfg=self.scfg):
                        logger.info("🎯 Regular draft limit reached, triggering multi-branch aggregation (conditions already checked in select())")
                        result_node = aggregation_agent.run(self, mode="node", parent_node=parent_node)
                        if result_node:
                            result_node.lock = True
                            logger.info(f"[_run_single_step] Aggregation branch node {result_node.id} is locked.")
                        else:
                            logger.info("Aggregation failed or limit reached, skipping. Will continue normal search.")
                            result_node = None
                    else:
                        result_node = draft_agent.run(self, init_solution_path=init_solution_path)
                        if result_node:
                            result_node.lock = True
                            logger.info(f"[_run_single_step] Draft node {result_node.id} is locked.")
                        else:
                            logger.info("Draft generation skipped because no child slot was available.")
                elif parent_node.is_buggy or parent_node.is_valid is False:
                    result_node = debug_agent.run(self, parent_node)

                elif parent_node.is_buggy is False:
                    can_use_fusion = False
                    if self.search_start_time:
                        elapsed_time = time.time() - self.search_start_time
                        if elapsed_time >= self.acfg.time_limit / 2:
                            can_use_fusion = True
                    is_from_topk = getattr(parent_node, '_topk_triggered', False)
                    stagnation_threshold = self.scfg.topk_stagnation_threshold if is_from_topk else self.scfg.branch_stagnation_threshold
                    if is_from_topk:
                        logger.info(f"🎯 Exploitation mode: using relaxed stagnation threshold ({stagnation_threshold} attempts)")

                    if is_branch_stagnant(self, parent_node.branch_id, threshold=stagnation_threshold):
                        if can_use_fusion:
                            if random.random() < self.acfg.fusion_vs_evolution_prob:
                                logger.info(f"🎯 Triggering fusion for stagnant node {parent_node.id} (after 6h)")
                                result_node = fusion_agent.run(self, parent_node)
                            else:
                                logger.info(f"🎯 Triggering intra-branch evolution for stagnant node {parent_node.id} (after 6h)")
                                result_node = evolution_agent.run(self, parent_node)
                        else:
                            logger.info(f"🔄 Using evolution for stagnant node {parent_node.id} (before 6h)")
                            result_node = evolution_agent.run(self, parent_node)
                    else:
                        logger.info(f"🔄 Using normal improve for node {parent_node.id}")
                        result_node = improve_agent.run(self, parent_node)

                else:
                    logger.warning(f"[_run_single_step] node {parent_node.id} is_buggy is None.")

                if result_node:
                    self.refresh_hardware_context(result_node)
                    if init_solution_path:
                        logger.info(f"Node {result_node.id} from init_solution, skipping code review")
                    else:
                        reviewed_code = code_review_agent.run(self, result_node)
                        if reviewed_code.strip() != result_node.code.strip():
                            logger.info(f"Node {result_node.id} code has been reviewed and modified")
                            result_node.code = reviewed_code
                            self.refresh_hardware_context(result_node)
                        else:
                            logger.info(f"Node {result_node.id} passed code review without changes")

                    if not execute_immediately:
                        logger.info(f"Node {result_node.id} code generated and reviewed, execution deferred")
                        result_node.pending_execution = True
                        from utils.pipeline_logging import record_pipeline_node_action

                        record_pipeline_node_action(self, result_node, "execution_deferred")
                        return _root, result_node
                    from utils.pipeline_logging import record_pipeline_node_action

                    record_pipeline_node_action(self, result_node, "execution_started")
                    exe_res = exec_callback(result_node.code, result_node.id, True)
                    result_node = result_parse_agent.run(self,
                        node=result_node,
                        exec_result=exe_res
                    )
                    if self.pipeline_logger is not None and result_node.metric is not None:
                        self.pipeline_logger.update_job_packet_for_node(
                            str(result_node.id),
                            metric=result_node.metric.value,
                            duration_seconds=result_node.exec_time,
                            status="parsed_buggy" if result_node.is_buggy else "parsed_valid",
                        )
                    execution.validate_executed_node(self, result_node)
                    logger.info(f"The metric value of node {result_node.id} is {result_node.metric.value}.")
                    result_node.finish_time = time.strftime("%Y-%m-%dT%H:%M:%S")
                    record_pipeline_node_action(self, result_node, "execution_finished")

                    if parent_node.is_buggy and result_node.is_buggy is False:
                        parent_node.is_debug_success = True

                    _root = evaluation.check_improvement(self, result_node, parent_node)
                    if result_node.stage in ["draft", "fusion_draft"]:
                        result_node.lock = False
                    with self.journal_lock:
                        if self.best_node and result_node.metric.maximize and self.best_node.metric.maximize != result_node.metric.maximize:
                            logger.warning(
                                "New node's metric is inconsistent with metrics in the journal. Returning to the parent node to regenerate.")
                            raise ValueError(
                                "New node's metric is inconsistent with metrics in the journal. Returning to the parent node to regenerate.")
                        else:
                            self.journal.append(result_node)

            except Exception as e:
                logger.warning(f"Step failed for parent {parent_node.id}, rolling back expected child count and propagating zero reward.")
                evaluation.backpropagate(node=parent_node, value=0, add_to_tree=False)
                if result_node is not None and result_node not in self.journal.nodes:
                    self._discard_unfinished_node(result_node)
                parent_node.sub_expected_child_count()
                raise e

        else:
            evaluation.backpropagate(node=parent_node, value=0)
            _root = True
        return _root, result_node

    def step(
        self,
        node: SearchNode | None = None,
        exec_callback: ExecCallbackType | None = None,
        execute_immediately: bool = True,
        init_solution_path: Optional[str] = None,
    ) -> SearchNode | None:
        if exec_callback is None:
            raise ValueError("exec_callback is required")

        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()
            self.search_start_time = time.time()

        if not node or node.stage == "root":
            node = node_selection.select_with_soft_switch(self)
            if node is None:
                logger.info("[step] no selectable work available")
                self.current_step = len(self.journal)
                return None

        _root, result_node = self._run_single_step(
            node,
            exec_callback=exec_callback,
            execute_immediately=execute_immediately,
            init_solution_path=init_solution_path,
        )

        if result_node:
            metric_value = result_node.metric.value if result_node.metric else None
            best_metric = self.best_node.metric.value if (self.best_node and self.best_node.metric) else None
            logger.info(f"[step] {node.id} → {result_node.id}: metric={metric_value}, best={best_metric}")

        if result_node and result_node.metric and result_node.metric.value is not None:
            solution_manager.update_best_solution(self, result_node)

        self.current_step = len(self.journal)

        # Cumulative stats
        total_nodes = len(self.journal)
        n_branches = len(self.branch_all_nodes)
        best_val = self.best_node.metric.value if (self.best_node and self.best_node.metric) else None
        logger.info(f"[stats] step={self.current_step}, nodes={total_nodes}, branches={n_branches}, best={best_val}")

        if result_node is None:
            return None
        if _root:
            return self.virtual_root
        return result_node

    def execute_deferred_node(self, node: SearchNode, exec_callback: ExecCallbackType) -> SearchNode:
        """Execute a node that was generated and reviewed but not yet run (pending_execution=True)."""
        if not hasattr(node, 'pending_execution') or not node.pending_execution:
            logger.warning(f"Node {node.id} is not marked for deferred execution")
            return node

        logger.info(f"Executing deferred node {node.id}")
        parent_node = node.parent

        try:
            from utils.pipeline_logging import record_pipeline_node_action

            record_pipeline_node_action(self, node, "execution_started")
            exe_res = exec_callback(node.code, node.id, True)
            node = result_parse_agent.run(self,
                node=node,
                exec_result=exe_res
            )
            if self.pipeline_logger is not None and node.metric is not None:
                self.pipeline_logger.update_job_packet_for_node(
                    str(node.id),
                    metric=node.metric.value,
                    duration_seconds=node.exec_time,
                    status="parsed_buggy" if node.is_buggy else "parsed_valid",
                )

            execution.validate_executed_node(self, node)

            logger.info(f"Node {node.id} execution completed: metric={node.metric.value}, is_buggy={node.is_buggy}")

            node.finish_time = time.strftime("%Y-%m-%dT%H:%M:%S")
            record_pipeline_node_action(self, node, "execution_finished")

            if parent_node and parent_node.is_buggy and node.is_buggy is False:
                parent_node.is_debug_success = True

            _root = evaluation.check_improvement(self, node, parent_node)
            if node.stage in ["draft", "fusion_draft"]:
                node.lock = False

            with self.journal_lock:
                if self.best_node and node.metric.maximize and self.best_node.metric.maximize != node.metric.maximize:
                    logger.warning("New node's metric is inconsistent with metrics in the journal")
                    raise ValueError("New node's metric is inconsistent with metrics in the journal")
                else:
                    self.journal.append(node)
                    logger.info(f"Node {node.id} added to journal")

            node.pending_execution = False
            solution_manager.update_best_solution(self, node)

            return node

        except Exception as e:
            logger.exception(f"Exception during deferred node execution: {e}")
            evaluation.backpropagate(node=parent_node, value=0, add_to_tree=False)
            if node not in self.journal.nodes:
                self._discard_unfinished_node(node)
            parent_node.sub_expected_child_count()
            raise e

    def execute_deferred_nodes(
        self,
        nodes: list[SearchNode],
        exec_many_callback: ExecManyCallbackType,
    ) -> list[SearchNode]:
        """Execute a scheduler round of deferred nodes and append successful parses to the journal."""
        runnable_nodes = [node for node in nodes if getattr(node, "pending_execution", False)]
        if not runnable_nodes:
            return []

        try:
            from agents.hardware_context import optimize_training_parameters_for_round

            decisions = optimize_training_parameters_for_round(self, runnable_nodes)
            if decisions:
                from utils.pipeline_logging import log_pipeline_event

                log_pipeline_event(
                    self,
                    "hardware_round_review",
                    payload={
                        "node_ids": [str(node.id) for node in runnable_nodes],
                        "decisions": decisions,
                    },
                )
        except Exception as exc:
            logger.debug("Skipping hardware-aware pre-submit round review: %s", exc)

        from utils.pipeline_logging import record_pipeline_node_action

        for node in runnable_nodes:
            record_pipeline_node_action(self, node, "execution_started")

        results = exec_many_callback([(node.code, str(node.id)) for node in runnable_nodes])
        executed_nodes: list[SearchNode] = []

        for node in runnable_nodes:
            parent_node = node.parent
            try:
                exe_res = results.get(str(node.id))
                if exe_res is None:
                    exe_res = ExecutionResult(
                        term_out=["Scheduler round did not return a result for this node."],
                        exec_time=0.0,
                        exc_type="RuntimeError",
                        exc_info={"message": "missing scheduler round result"},
                        exc_stack=[],
                    )
                node = result_parse_agent.run(self, node=node, exec_result=exe_res)
                if self.pipeline_logger is not None and node.metric is not None:
                    self.pipeline_logger.update_job_packet_for_node(
                        str(node.id),
                        metric=node.metric.value,
                        duration_seconds=node.exec_time,
                        status="parsed_buggy" if node.is_buggy else "parsed_valid",
                    )

                execution.validate_executed_node(self, node)
                logger.info("Node %s round execution completed: metric=%s, is_buggy=%s", node.id, node.metric.value, node.is_buggy)

                node.finish_time = time.strftime("%Y-%m-%dT%H:%M:%S")
                record_pipeline_node_action(self, node, "execution_finished")

                if parent_node and parent_node.is_buggy and node.is_buggy is False:
                    parent_node.is_debug_success = True

                evaluation.check_improvement(self, node, parent_node)
                if node.stage in ["draft", "fusion_draft"]:
                    node.lock = False

                with self.journal_lock:
                    if self.best_node and node.metric.maximize and self.best_node.metric.maximize != node.metric.maximize:
                        logger.warning("New node's metric is inconsistent with metrics in the journal")
                        raise ValueError("New node's metric is inconsistent with metrics in the journal")
                    self.journal.append(node)
                    logger.info("Node %s added to journal", node.id)

                node.pending_execution = False
                solution_manager.update_best_solution(self, node)
                executed_nodes.append(node)
            except Exception as exc:
                logger.exception("Exception during scheduler round node execution for %s: %s", node.id, exc)
                if parent_node is not None:
                    evaluation.backpropagate(node=parent_node, value=0, add_to_tree=False)
                    parent_node.sub_expected_child_count()
                if node not in self.journal.nodes:
                    self._discard_unfinished_node(node)
        self.current_step = len(self.journal)
        return executed_nodes

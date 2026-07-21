from __future__ import annotations

from types import SimpleNamespace
import threading
import time

import engine.agent_search as agent_search_module
from engine.agent_search import AgentSearch
from engine.executor import ExecutionResult, SchedulerJobHandle
from engine.search_node import Journal, SearchNode
from run import _live_scheduler_empty_generation_action
from utils.metric import MetricValue, WorstMetricValue


def _make_agent(tmp_path) -> AgentSearch:
    search_cfg = SimpleNamespace(
        num_drafts=3,
        num_improves=2,
        num_bugs=1,
        topk_max_improves=3,
        explore_switch_start=0.5,
        explore_switch_end=0.8,
        min_exploration_weight=0.2,
        metric_improvement_threshold=0.001,
        max_improve_failure=2,
        force_backprop_late_threshold=0.9,
        force_backprop_late_prob=0.0,
        force_backprop_mid_threshold=0.5,
        force_backprop_mid_modulo=10,
        recent_best_window=5,
        topk_early_k=1,
        topk_early_max_per_branch=1,
        topk_late_k=1,
        topk_late_max_per_branch=1,
        branch_stagnation_threshold=3,
        topk_stagnation_threshold=3,
    )
    decay_cfg = SimpleNamespace(
        exploration_constant=1.414,
        phase_ratios=[0.5, 0.8],
        alpha=0.01,
        lower_bound=0.7,
    )
    agent_cfg = SimpleNamespace(
        search=search_cfg,
        decay=decay_cfg,
        steps=10,
        time_limit=100,
        branch_fusion_trigger_prob=0.0,
        fusion_vs_evolution_prob=0.0,
    )
    cfg = SimpleNamespace(
        agent=agent_cfg,
        experiment=SimpleNamespace(mode="hardware_aware"),
        exp_id="unit-task",
        workspace_dir=tmp_path,
    )
    (tmp_path / "submission").mkdir(parents=True, exist_ok=True)
    (tmp_path / "best_solution").mkdir(parents=True, exist_ok=True)

    journal = Journal()
    root = SearchNode(parent=None, plan="root", code="", metric=WorstMetricValue(), stage="root")
    journal.append(root)

    agent = AgentSearch.__new__(AgentSearch)
    agent.cfg = cfg
    agent.pipeline_logger = None
    agent.acfg = agent_cfg
    agent.scfg = search_cfg
    agent.task_desc = "unit task"
    agent.journal = journal
    agent.data_preview = "preview"
    agent.current_step = 0
    agent.current_node_list = []
    agent.virtual_root = root
    agent.best_node = None
    agent.best_metric = None
    agent.metric_maximize = True
    agent.search_start_time = time.time()
    agent.journal_lock = threading.Lock()
    agent.save_node_lock = threading.Lock()
    agent.branch_all_nodes = {}
    agent.branch_successful_nodes = {}
    agent.branch_node_count = {}
    agent.top_k = 1
    agent.top_candidates = []
    agent.pending_scheduler_nodes = {}
    agent.refresh_hardware_context = lambda node: None
    return agent


def test_empty_live_scheduler_generation_waits_while_jobs_are_outstanding() -> None:
    assert _live_scheduler_empty_generation_action(1, 2) == "wait"
    assert _live_scheduler_empty_generation_action(5, 1) == "wait"
    assert _live_scheduler_empty_generation_action(1, 0) == "retry"
    assert _live_scheduler_empty_generation_action(3, 0) == "stop"


def test_step_with_scheduler_handle_defers_journal_and_best_until_collection(monkeypatch, tmp_path) -> None:
    agent = _make_agent(tmp_path)

    def fake_draft(run_agent, init_solution_path=None):
        assert run_agent.virtual_root.add_expected_child_count(run_agent.scfg)
        return SearchNode(
            plan="plan",
            code="print('train')",
            parent=run_agent.virtual_root,
            stage="draft",
            local_best_node=run_agent.virtual_root,
        )

    def fake_parse(run_agent, node, exec_result):
        node.absorb_exec_result(exec_result)
        node.metric = MetricValue(0.8, maximize=True)
        node.is_buggy = False
        node.is_valid = True
        return node

    monkeypatch.setattr(agent_search_module.draft_agent, "run", fake_draft)
    monkeypatch.setattr(agent_search_module.code_review_agent, "run", lambda run_agent, node: node.code)
    monkeypatch.setattr(agent_search_module.result_parse_agent, "run", fake_parse)
    monkeypatch.setattr(agent_search_module.execution, "validate_executed_node", lambda run_agent, node: None)
    monkeypatch.setattr(agent_search_module.solution_manager, "update_best_solution", lambda run_agent, node: setattr(run_agent, "best_node", node))

    def submit_callback(code, node_id, *args, **kwargs):
        return SchedulerJobHandle(node_id=str(node_id), job_id="job-stream", submission_label="stream")

    submitted_node = agent.step(exec_callback=submit_callback)

    assert submitted_node is not None
    assert submitted_node.scheduler_submitted is True
    assert submitted_node.scheduler_job_id == "job-stream"
    assert submitted_node.id in agent.pending_scheduler_nodes
    assert submitted_node not in agent.journal.nodes
    assert len(agent.journal) == 1
    assert agent.best_node is None

    finalized = agent.finalize_scheduler_results(
        {submitted_node.id: ExecutionResult(term_out=["ok"], exec_time=1.0, exc_type=None, exc_info={}, exc_stack=[])}
    )

    assert finalized == [submitted_node]
    assert len(agent.journal) == 2
    assert agent.journal.nodes[-1] is submitted_node
    assert agent.best_node is submitted_node
    assert submitted_node.scheduler_submitted is False
    assert submitted_node.pending_execution is False
    assert submitted_node.lock is False
    assert agent.pending_scheduler_nodes == {}

    assert agent.finalize_scheduler_results(
        {submitted_node.id: ExecutionResult(term_out=["ok"], exec_time=1.0, exc_type=None, exc_info={}, exc_stack=[])}
    ) == []
    assert len(agent.journal) == 2

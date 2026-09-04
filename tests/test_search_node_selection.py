from __future__ import annotations

from types import SimpleNamespace

from engine import node_selection
from engine.agent_search import AgentSearch
from engine.search_node import Journal, SearchNode


def test_search_node_term_out_is_safe_for_unexecuted_and_serialized_output() -> None:
    node = SearchNode(code="", plan="draft", stage="draft")
    assert node.term_out == ""

    node._term_out = ["hello", " ", "world"]
    assert node.term_out == "hello world"

    node._term_out = "<OMITTED>"
    assert node.term_out == "<OMITTED>"


def test_root_at_draft_limit_with_locked_children_has_no_selectable_work() -> None:
    root = SearchNode(code="", plan="root", stage="root")
    journal = Journal()
    journal.append(root)

    for _ in range(5):
        child = SearchNode(code="", plan="draft", parent=root, stage="draft")
        child.lock = True

    root.expected_child_count = len(root.children)

    agent = SimpleNamespace(
        virtual_root=root,
        journal=journal,
        scfg=SimpleNamespace(num_drafts=5),
        fusion_draft_count=0,
        max_fusion_drafts=0,
        search_start_time=None,
        is_root=lambda node: node is root,
    )

    assert node_selection.select(agent, root) is None
    assert node_selection.has_selectable_work(agent) is False


def test_restore_search_state_releases_stale_locks_and_rebuilds_branch_indexes() -> None:
    root = SearchNode(code="", plan="root", stage="root")
    child = SearchNode(code="", plan="draft", parent=root, stage="draft")
    child.branch_id = 3
    child.lock = True
    child.is_buggy = False
    child.is_valid = True

    search = AgentSearch.__new__(AgentSearch)
    search.scfg = SimpleNamespace(top_candidates_size=3)
    search.metric_maximize = False
    search.best_metric_history = []

    AgentSearch.restore_search_state(search, Journal(nodes=[root, child]))

    assert search.virtual_root is root
    assert search.current_step == 1
    assert child.lock is False
    assert child.expected_child_count == 0
    assert search.branch_all_nodes == {3: [child]}
    assert search.next_branch_id == 4


def test_resumed_search_without_a_persisted_metric_contract_redetermines_direction(
    monkeypatch,
) -> None:
    """A journal containing only failed nodes must not fall back to maximize=True."""
    search = AgentSearch.__new__(AgentSearch)
    search.metric_maximize = None
    search.metric_maximize_reasoning = None

    calls: list[object] = []

    def determine_direction(agent) -> None:
        calls.append(agent)
        agent.metric_maximize = False
        agent.metric_maximize_reasoning = "RMSE is minimized"

    monkeypatch.setattr(
        "engine.agent_search.result_parse_agent.determine_metric_direction",
        determine_direction,
    )

    AgentSearch._ensure_metric_direction(search)

    assert calls == [search]
    assert search.metric_maximize is False
    assert search.metric_maximize_reasoning == "RMSE is minimized"

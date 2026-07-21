from __future__ import annotations

from types import SimpleNamespace

from engine import node_selection
from engine.search_node import Journal, SearchNode
from utils.metric import MetricValue


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


def test_fetch_child_memory_handles_metricless_successful_nodes() -> None:
    root = SearchNode(code="", plan="root", stage="root")
    metric_child = SearchNode(code="", plan="metric", parent=root, stage="draft")
    metric_child.is_buggy = False
    metric_child.metric = MetricValue(0.73, maximize=True)
    metricless_child = SearchNode(code="", plan="metricless", parent=root, stage="draft")
    metricless_child.is_buggy = False
    metricless_child.metric = None
    metricless_child.outcome = "validation_unavailable"

    summary = root.fetch_child_memory()

    assert "2 successful (best: 0.7300, 1 without metric)" in summary

from types import SimpleNamespace

from engine.evaluation import check_improvement
from engine.search_node import SearchNode


def test_buggy_metricless_node_backpropagates_without_local_best() -> None:
    root = SearchNode(code="", plan="root", stage="root")
    node = SearchNode(code="broken", plan="draft", parent=root, stage="draft")
    node.is_buggy = True
    node.metric = None
    node.local_best_node = None
    agent = SimpleNamespace(
        search_start_time=None,
        metric_maximize=False,
        best_metric=None,
        best_node=None,
        current_node_list=[],
    )

    assert check_improvement(agent, node, root) is True
    assert root.visits == 1

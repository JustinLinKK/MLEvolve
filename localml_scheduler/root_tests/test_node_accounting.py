from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from engine.node_accounting import (
    count_budget_nodes,
    count_budget_nodes_from_json,
    node_counts_toward_budget,
)


def _node(**values: object) -> SimpleNamespace:
    defaults: dict[str, object] = {
        "stage": "improve",
        "is_buggy": False,
        "exec_time": 120.0,
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def test_nodes_returned_within_thirty_seconds_do_not_consume_the_search_budget() -> None:
    nodes = [
        _node(stage="root", is_buggy=False, exec_time=None),
        _node(is_buggy=True, exec_time=0.0),
        _node(is_buggy=True, exec_time=29.999),
        _node(is_buggy=True, exec_time=30.0),
        _node(is_buggy=False, exec_time=2.0),
    ]

    assert count_budget_nodes(nodes) == 1
    assert node_counts_toward_budget(nodes[3]) is True
    assert node_counts_toward_budget(nodes[4]) is False


def test_failed_node_without_execution_time_does_not_consume_budget() -> None:
    node = _node(is_buggy=True, exec_time=None)

    assert node_counts_toward_budget(node) is False


def test_serialized_journal_uses_the_same_budget_rule(tmp_path: Path) -> None:
    journal = tmp_path / "journal.json"
    journal.write_text(
        json.dumps(
            {
                "nodes": [
                    {"stage": "root", "is_buggy": False, "exec_time": None},
                    {"stage": "draft", "is_buggy": True, "exec_time": 29.0},
                    {"stage": "debug", "is_buggy": False, "exec_time": 12.0},
                    {"stage": "improve", "is_buggy": True, "exec_time": 30.0},
                ]
            }
        )
    )

    assert count_budget_nodes_from_json(journal) == 1

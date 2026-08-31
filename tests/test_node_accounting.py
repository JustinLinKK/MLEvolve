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


def test_quick_failed_nodes_do_not_consume_the_search_budget() -> None:
    nodes = [
        _node(stage="root", is_buggy=False, exec_time=None),
        _node(is_buggy=True, exec_time=0.0),
        _node(is_buggy=True, exec_time=25.0),
        _node(is_buggy=True, exec_time=59.999),
        _node(is_buggy=True, exec_time=60.0),
        _node(is_buggy=False, exec_time=2.0),
    ]

    assert count_budget_nodes(nodes) == 2
    assert node_counts_toward_budget(nodes[4]) is True
    assert node_counts_toward_budget(nodes[5]) is True


def test_serialized_journal_uses_the_same_budget_rule(tmp_path: Path) -> None:
    journal = tmp_path / "journal.json"
    journal.write_text(
        json.dumps(
            {
                "nodes": [
                    {"stage": "root", "is_buggy": False, "exec_time": None},
                    {"stage": "draft", "is_buggy": True, "exec_time": 42.0},
                    {"stage": "debug", "is_buggy": False, "exec_time": 12.0},
                    {"stage": "improve", "is_buggy": True, "exec_time": 75.0},
                ]
            }
        )
    )

    assert count_budget_nodes_from_json(journal) == 2


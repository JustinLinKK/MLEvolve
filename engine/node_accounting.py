"""Search-budget accounting rules shared by runs and experiment verifiers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


# A candidate rejected by the CPU preflight/container in under thirty seconds did
# not perform a meaningful experiment, so it must not consume one of the user
# requested nodes.  At ten seconds or longer, retain the failed attempt as a
# real runtime observation for the search budget and post-run accounting.
MIN_COUNTED_FAILED_EXECUTION_SECONDS = 30.0


def _field(node: object, name: str, default: Any = None) -> Any:
    if isinstance(node, Mapping):
        return node.get(name, default)
    return getattr(node, name, default)


def node_counts_toward_budget(node: object) -> bool:
    """Return whether a retained journal node consumes one search step."""
    if _field(node, "stage") == "root":
        return False
    if _field(node, "is_buggy") is not True:
        return True
    try:
        execution_seconds = float(_field(node, "exec_time"))
    except (TypeError, ValueError):
        return False
    return execution_seconds >= MIN_COUNTED_FAILED_EXECUTION_SECONDS


def count_budget_nodes(nodes: Iterable[object]) -> int:
    """Count nodes after excluding static or quickly detected failures."""
    return sum(node_counts_toward_budget(node) for node in nodes)


def count_budget_nodes_from_json(path: str | Path) -> int:
    """Count budget nodes in a serialized MLEvolve journal."""
    payload = json.loads(Path(path).read_text())
    return count_budget_nodes(payload.get("nodes", []))

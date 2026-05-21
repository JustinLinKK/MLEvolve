"""Verify the seeded graph instantiates every node label and every
relationship type defined in schema/graph_schema.yaml.

Exit 0 if all rows pass; non-zero otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from mini_graph_demo._driver import connect, session

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_SCHEMA = REPO_ROOT / "schema" / "graph_schema.yaml"


def _schema_labels_and_rels() -> tuple[list[str], list[str]]:
    spec = yaml.safe_load(GRAPH_SCHEMA.read_text(encoding="utf-8"))
    nodes = spec.get("node_types", {})
    # Skip abstract labels (e.g. Job is abstract — only used as additional label)
    labels = [name for name, body in nodes.items() if not body.get("abstract")]
    rels = list(spec.get("relationship_types", {}).keys())
    return labels, rels


def _count_nodes(s, label: str) -> int:
    rec = s.run(
        f"MATCH (n:{label} {{demo_run: true}}) RETURN count(n) AS c"
    ).single()
    return rec["c"]


def _count_rels(s, rel_type: str) -> int:
    rec = s.run(
        f"MATCH ()-[r:{rel_type} {{demo_run: true}}]->() RETURN count(r) AS c"
    ).single()
    return rec["c"]


def main() -> int:
    labels, rels = _schema_labels_and_rels()
    rows: list[tuple[str, str, int, int, bool]] = []
    driver = connect()
    try:
        with session(driver) as s:
            for lbl in labels:
                obs = _count_nodes(s, lbl)
                rows.append(("node", lbl, 1, obs, obs >= 1))
            for rel in rels:
                obs = _count_rels(s, rel)
                rows.append(("rel", rel, 1, obs, obs >= 1))
    finally:
        driver.close()

    width_type = max(len(r[1]) for r in rows)
    print(f"{'kind':<5} | {'type':<{width_type}} | {'min':<3} | {'obs':<3} | pass")
    for kind, t, mn, obs, ok in rows:
        print(f"{kind:<5} | {t:<{width_type}} | {mn:<3} | {obs:<3} | "
              f"{'yes' if ok else 'NO'}")

    failed = [r for r in rows if not r[4]]
    if failed:
        sys.stderr.write(f"verify_coverage: {len(failed)} type(s) missing coverage\n")
        return 1
    print("verify_coverage: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

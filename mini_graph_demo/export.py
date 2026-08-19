"""Dump the demo subgraph (demo_run=true) to mini_graph_demo/graph.json
in Cytoscape.js element format expected by mini_graph_demo/visualize.html.

Output schema:
  {
    "nodes": [{"data": {"id", "type", "label", ...props}}, ...],
    "edges": [{"data": {"id", "source", "target", "type", ...props}}]
  }
"""
from __future__ import annotations

import json
from pathlib import Path

from mini_graph_demo._driver import connect, session

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = Path(__file__).resolve().parent / "graph.json"

# Pick the most specific label (e.g. SingleJob over Job) for visualizer typing.
LABEL_PRIORITY = [
    "PackedJobMember", "SingleJob", "PackedJob", "Job",
    "Hardware", "Model", "Technology", "TrainingConfig",
]

PRIMARY_KEY_FIELDS = (
    "job_id", "hardware_key", "model_key", "config_key",
    "technology_key", "member_id",
)


def _node_type(labels: list[str]) -> str:
    for lbl in LABEL_PRIORITY:
        if lbl in labels:
            return lbl
    return labels[0] if labels else "Unknown"


def _node_label(props: dict, node_type: str) -> str:
    # Prefer a readable name if present, else the primary key with mgd: stripped.
    for f in ("model_name", "product_name", "name"):
        if props.get(f):
            return str(props[f])
    for f in PRIMARY_KEY_FIELDS:
        if props.get(f):
            return str(props[f]).removeprefix("mgd:")
    return node_type


def _node_id(props: dict, neo4j_id: int) -> str:
    for f in PRIMARY_KEY_FIELDS:
        if props.get(f):
            return str(props[f])
    return f"_n{neo4j_id}"


def main() -> int:
    driver = connect()
    try:
        with session(driver) as s:
            node_rows = list(s.run(
                "MATCH (n {demo_run: true}) "
                "RETURN id(n) AS nid, labels(n) AS labels, properties(n) AS props"
            ))
            edge_rows = list(s.run(
                "MATCH (a {demo_run: true})-[r {demo_run: true}]->(b {demo_run: true}) "
                "RETURN id(a) AS src, id(b) AS dst, "
                "       labels(a) AS src_labels, labels(b) AS dst_labels, "
                "       properties(a) AS src_props, properties(b) AS dst_props, "
                "       type(r) AS rtype, properties(r) AS props, id(r) AS rid"
            ))

        nodes = []
        for rec in node_rows:
            t = _node_type(rec["labels"])
            nid = _node_id(rec["props"], rec["nid"])
            data = {
                "id": nid,
                "type": t,
                "label": _node_label(rec["props"], t),
                **rec["props"],
            }
            nodes.append({"data": data})

        edges = []
        for rec in edge_rows:
            sid = _node_id(rec["src_props"], rec["src"])
            tid = _node_id(rec["dst_props"], rec["dst"])
            data = {
                "id": f"e{rec['rid']}",
                "source": sid,
                "target": tid,
                "type": rec["rtype"],
                **rec["props"],
            }
            edges.append({"data": data})

        OUTPUT_PATH.write_text(
            json.dumps({"nodes": nodes, "edges": edges}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"exported {len(nodes)} nodes, {len(edges)} edges -> {OUTPUT_PATH}")
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

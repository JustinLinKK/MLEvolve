"""Build the hardware-feature GraphDB instance data from the vector schemas.

This is the data migration that turns vector schemas A/B/D + feature_docs into
the pure-graph model defined in hardware_knowledge_graph_schema.yaml:

  (:Hardware) -[:HAS_FEATURE]-> (:Feature) -[:REQUIRES_FEATURE]-> (:Feature)

API symbols and one canonical example_code are MERGED onto each Feature node.

Outputs (these become the GraphDB; the vector schemas can then be removed):
  schema/hardware_knowledge_graph.json    nodes + edges instance data
  schema/hardware_knowledge_graph.cypher  Neo4j load script (constraints + MERGE)

Run once: it reads the vector schemas. After the vector schemas are deleted it
is kept only as the migration record (re-run requires the vector sources).
"""
from __future__ import annotations
import glob
import json
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
A_DIR = HERE / "hardware_feature_records"
B_DIR = HERE / "code_doc_chunks"
D_DIR = HERE / "api_symbol_chunks"
FEATURE_DOCS = HERE / "feature_docs.yaml"
OUT_JSON = HERE / "hardware_knowledge_graph.json"
OUT_CYPHER = HERE / "hardware_knowledge_graph.cypher"

# intrinsic minimum compute capability per feature (gating), for HAS_FEATURE edge
MIN_CC = {
    "bf16": "8.0", "tf32": "8.0", "tensor_cores_3gen": "8.0",
    "sm_80": "8.0", "sm_86": "8.6",
    "fp8": "8.9", "fp8_e4m3": "8.9", "fp8_e5m2": "8.9",
    "sm_89": "8.9", "tensor_cores_4gen": "8.9",
    "sm_90": "9.0",
    "fp4": "10.0", "sm_100": "10.0", "tensor_cores_5gen": "10.0",
    "sm_120": "12.0",
}

REQUIRES = [
    ("fp8_e4m3", "fp8"), ("fp8_e5m2", "fp8"),
    ("tensor_cores_3gen", "tensor_cores"),
    ("tensor_cores_4gen", "tensor_cores"),
    ("tensor_cores_5gen", "tensor_cores"),
]


def feature_category(fid: str) -> str:
    if fid.startswith("sm_"):
        return "compute_capability"
    if fid.startswith("tensor_cores"):
        return "tensor_core"
    if fid == "cuda_graphs":
        return "kernel"
    if fid == "pcie5_x16":
        return "interconnect"
    if fid in ("bf16", "fp16", "fp4", "fp64", "fp8", "fp8_e4m3",
               "fp8_e5m2", "int8", "tf32", "amp"):
        return "precision"
    return "other"


def build():
    # merge api symbols + pick one best example per feature, from B (+ D apis)
    feat_apis: dict[str, set] = defaultdict(set)
    feat_cands: dict[str, list] = defaultdict(list)
    feat_desc: dict[str, str] = {}
    for p in glob.glob(str(B_DIR / "*.yaml")):
        d = yaml.safe_load(Path(p).read_text())
        code = (d.get("example_code") or "").strip()
        hwfk = d.get("hardware_feature_keys") or []
        for f in hwfk:
            for api in d.get("api_symbols") or []:
                feat_apis[f].add(api)
            if code:
                feat_cands[f].append({"chunk": d.get("chunk_id", ""),
                                      "code": code, "n_keys": len(hwfk)})
            if len(hwfk) == 1 and d.get("text"):
                feat_desc.setdefault(f, d["text"].strip())
    for p in glob.glob(str(D_DIR / "*.yaml")):
        d = yaml.safe_load(Path(p).read_text())
        api = d.get("api_symbol")
        for f in d.get("hardware_feature_keys") or []:
            if api:
                feat_apis[f].add(api)

    def best_example(fid):
        c = feat_cands.get(fid)
        if not c:
            return None
        return min(c, key=lambda x: (0 if fid in x["chunk"] else 1, x["n_keys"]))

    fd = (yaml.safe_load(FEATURE_DOCS.read_text()) or {}).get("feature_docs") or {}
    a_records = [yaml.safe_load(Path(p).read_text())
                 for p in sorted(glob.glob(str(A_DIR / "*.yaml")))]
    all_feats = set(fd)
    for a in a_records:
        all_feats.update(a.get("features") or [])

    nodes = []
    for fid in sorted(all_feats):
        meta = fd.get(fid, {})
        ex = best_example(fid)
        nodes.append({
            "label": "Feature", "id": f"feat:{fid}",
            "properties": {
                "feature_id": fid,
                "name": fid,
                "category": feature_category(fid),
                "description": feat_desc.get(fid, meta.get("title", "")),
                "tensor_core_generation": (
                    fid if fid.startswith("tensor_cores_") else ""),
                "example_code": ex["code"] if ex else "",
                "example_code_source": ex["chunk"] if ex else "",
                "api_symbols": sorted(feat_apis.get(fid, [])),
                "source_url": meta.get("url", ""),
            },
        })
    for a in a_records:
        nodes.append({
            "label": "Hardware", "id": f"hw:{a.get('record_id')}",
            "properties": {
                "hardware_id": a.get("record_id"),
                "name": a.get("title"),
                "vendor": a.get("vendor"),
                "aliases": a.get("accelerator_names") or [],
                "architectures": a.get("architectures") or [],
                "compute_capabilities": a.get("compute_capabilities") or [],
                "vram_MB": a.get("vram_MB"),
                "vram_type": a.get("vram_type"),
                "sm_count": a.get("sm_count"),
                "workload_types": a.get("workload_types") or [],
                "recommended_patterns": a.get("recommended_patterns") or [],
                "avoid_patterns": a.get("avoid_patterns") or [],
            },
        })

    present = {n["id"] for n in nodes}
    edges = []
    for a in a_records:
        hid = f"hw:{a.get('record_id')}"
        cc = (a.get("compute_capabilities") or [None])[0]
        for f in a.get("features") or []:
            fid = f"feat:{f}"
            if fid in present:
                edges.append({"type": "HAS_FEATURE", "from": hid, "to": fid,
                              "properties": {
                                  "support_level": "supported",
                                  "min_compute_capability": MIN_CC.get(f, ""),
                                  "device_compute_capability": cc,
                                  "verified": False,
                              }})
    for c, p in REQUIRES:
        if f"feat:{c}" in present and f"feat:{p}" in present:
            edges.append({"type": "REQUIRES_FEATURE", "from": f"feat:{c}",
                          "to": f"feat:{p}", "properties": {"relation": "specializes"}})
    return nodes, edges


def emit_cypher(nodes, edges) -> str:
    lines = [
        "// hardware-feature knowledge GraphDB. Generated by build_graph_db.py.",
        "// Independent of graph_schema.yaml (the job-evidence graph).",
        "CREATE CONSTRAINT hardware_id IF NOT EXISTS FOR (h:Hardware) REQUIRE h.hardware_id IS UNIQUE;",
        "CREATE CONSTRAINT feature_id IF NOT EXISTS FOR (f:Feature) REQUIRE f.feature_id IS UNIQUE;",
        "",
    ]
    for n in nodes:
        key = "hardware_id" if n["label"] == "Hardware" else "feature_id"
        props = json.dumps(n["properties"], ensure_ascii=False)
        lines.append(
            f"MERGE (n:{n['label']} {{{key}: {json.dumps(n['properties'][key])}}}) "
            f"SET n += apoc.convert.fromJsonMap('{props.replace(chr(39), chr(92) + chr(39))}');"
        )
    lines.append("")
    for e in edges:
        fkey = "hardware_id" if e["from"].startswith("hw:") else "feature_id"
        fval = e["from"].split(":", 1)[1]
        tval = e["to"].split(":", 1)[1]
        flabel = "Hardware" if e["from"].startswith("hw:") else "Feature"
        props = json.dumps(e["properties"], ensure_ascii=False)
        lines.append(
            f"MATCH (a:{flabel} {{{fkey}: {json.dumps(fval)}}}), "
            f"(b:Feature {{feature_id: {json.dumps(tval)}}}) "
            f"MERGE (a)-[r:{e['type']}]->(b) "
            f"SET r += apoc.convert.fromJsonMap('{props.replace(chr(39), chr(92) + chr(39))}');"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    nodes, edges = build()
    OUT_JSON.write_text(json.dumps({"nodes": nodes, "edges": edges},
                                   ensure_ascii=False, indent=2))
    OUT_CYPHER.write_text(emit_cypher(nodes, edges))
    n_hw = sum(1 for n in nodes if n["label"] == "Hardware")
    n_ft = sum(1 for n in nodes if n["label"] == "Feature")
    n_hf = sum(1 for e in edges if e["type"] == "HAS_FEATURE")
    n_rq = sum(1 for e in edges if e["type"] == "REQUIRES_FEATURE")
    print(f"built GraphDB: {n_hw} Hardware + {n_ft} Feature nodes, "
          f"{n_hf} HAS_FEATURE + {n_rq} REQUIRES_FEATURE edges")
    print(f"  -> {OUT_JSON.name}")
    print(f"  -> {OUT_CYPHER.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

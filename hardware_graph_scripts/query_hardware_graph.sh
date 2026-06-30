#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hardware_graph_env.sh
source "$SCRIPT_DIR/hardware_graph_env.sh"

usage() {
  cat <<'EOF'
Usage:
  hardware_graph_scripts/query_hardware_graph.sh MODE [HARDWARE_NAME] [STAGE_OR_QUERY] [LIMIT]

Query tool wrapper for hardware graph pre-integration checks.

Modes:
  node             Query one Hardware node from schema/hardware_knowledge_graph.json.
  features         Query Feature nodes for one hardware and pipeline stage.
  stage-context    Return both node and features for a pipeline stage.
  db-search        Search loaded Neo4j hardware graph records.
  db-neighborhood  Return loaded Neo4j features directly linked to hardware.

Stages:
  model_structure, datatype, training_parameters, all

Legacy filter names model, optimizer, and tuning are intentionally rejected.

Examples:
  ./hardware_graph_scripts/query_hardware_graph.sh node "GeForce RTX 5090" model_structure
  ./hardware_graph_scripts/query_hardware_graph.sh features "GeForce RTX 5090" training_parameters 24
  ./hardware_graph_scripts/query_hardware_graph.sh stage-context "GeForce RTX 5090" training_parameters 24
  ./hardware_graph_scripts/query_hardware_graph.sh db-neighborhood "GeForce RTX 5090" all 32
EOF
}

mode="${1:-}"
hardware_name="${2:-${HARDWARE_NAME:-GeForce RTX 5090}}"
stage_or_query="${3:-all}"
limit="${4:-8}"

if [[ -z "$mode" || "$mode" == "-h" || "$mode" == "--help" ]]; then
  usage
  exit 0
fi

cd "$HARDWARE_GRAPH_REPO_ROOT"

case "$mode" in
  node|hardware-node|features|stage-context)
    "$HARDWARE_GRAPH_PYTHON" - "$mode" "$hardware_name" "$stage_or_query" "$limit" <<'PY'
import json
import sys

from localml_scheduler.hardware_knowledge.feature_filter import (
    query_hardware_features,
    query_hardware_node,
)

mode, hardware_name, stage, limit_raw = sys.argv[1:5]
limit = max(1, int(limit_raw))
stage_clean = stage.strip().lower()
stage_value = None if stage_clean in {"", "all", "none", "null"} else stage_clean
valid_stages = {"model_structure", "datatype", "training_parameters"}
if stage_value is not None and stage_value not in valid_stages:
    allowed = ", ".join(sorted(valid_stages))
    raise SystemExit(
        f"Unsupported static hardware stage {stage!r}; use one of: {allowed}, or all."
    )

def limit_features(payload: dict) -> dict:
    if "features" not in payload:
        return payload
    features = list(payload.get("features") or [])
    payload = dict(payload)
    payload["total_feature_count"] = int(payload.get("feature_count") or len(features))
    payload["features"] = features[:limit]
    payload["returned_feature_count"] = len(payload["features"])
    return payload

if mode in {"node", "hardware-node"}:
    payload = query_hardware_node(hardware_name, stage_value)
elif mode == "features":
    payload = limit_features(query_hardware_features(hardware_name, stage_value))
else:
    feature_result = limit_features(query_hardware_features(hardware_name, stage_value))
    payload = {
        "hardware_name": hardware_name,
        "stage_filter": stage_value,
        "hardware_node": query_hardware_node(hardware_name, stage_value),
        "feature_result": feature_result,
    }

print(json.dumps(payload, indent=2, sort_keys=True))
PY
    ;;
  db-search|db-neighborhood)
    wait_for_hardware_neo4j
    runtime_config="$(mktemp /tmp/mlevolve-hardware-query.XXXXXX.yaml)"
    cleanup() {
      rm -f "$runtime_config"
    }
    trap cleanup EXIT
    make_hardware_graph_runtime_config "$runtime_config"
    "$HARDWARE_GRAPH_PYTHON" - "$runtime_config" "$mode" "$hardware_name" "$stage_or_query" "$limit" <<'PY'
import json
import sys
from pathlib import Path

import yaml

from localml_scheduler.config import SchedulerConfig
from localml_scheduler.hardware_knowledge.store import HardwareKnowledgeGraphStore

config_path, mode, hardware_name, query, limit_raw = sys.argv[1:6]
limit = max(1, int(limit_raw))

def hardware_terms(value: str) -> list[str]:
    raw = [
        value,
        value.replace("NVIDIA ", "").replace("nvidia ", ""),
        value.replace("GeForce ", "").replace("geforce ", ""),
        value.replace("NVIDIA GeForce ", "").replace("nvidia geforce ", ""),
        value.replace(" ", "_"),
    ]
    seen: set[str] = set()
    terms: list[str] = []
    for item in raw:
        cleaned = " ".join(str(item or "").strip().lower().split())
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            terms.append(cleaned)
    return terms

payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
settings_payload = payload.get("scheduler", {}).get("settings", payload)
settings = SchedulerConfig.from_dict(settings_payload)
store = HardwareKnowledgeGraphStore(settings)
try:
    if mode == "db-search":
        normalized_hardware = hardware_name.replace("NVIDIA ", "").replace("nvidia ", "")
        results = store.search(
            query=None if query.lower() == "all" else query,
            hardware=normalized_hardware,
            limit=limit,
        )
        payload = {
            "mode": mode,
            "hardware_name": hardware_name,
            "query": query,
            "result_count": len(results),
            "results": results,
        }
    else:
        neighborhood = store.get_feature_neighborhood(
            hardware_terms=hardware_terms(hardware_name),
            limit=limit,
        )
        payload = {
            "mode": mode,
            "hardware_name": hardware_name,
            "query": query,
            **neighborhood,
        }
finally:
    driver = getattr(store, "_driver", None)
    if driver is not None:
        driver.close()

print(json.dumps(payload, indent=2, sort_keys=True))
PY
    ;;
  *)
    echo "Unknown mode: $mode" >&2
    usage >&2
    exit 2
    ;;
esac

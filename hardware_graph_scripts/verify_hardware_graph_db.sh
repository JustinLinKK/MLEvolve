#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hardware_graph_env.sh
source "$SCRIPT_DIR/hardware_graph_env.sh"

usage() {
  cat <<'EOF'
Usage:
  hardware_graph_scripts/verify_hardware_graph_db.sh [HARDWARE_NAME]

Verify the loaded Neo4j hardware graph can return the expected pre-integration
feature records for the default three-stage agent simulation.

Default hardware: GeForce RTX 5090
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

hardware_name="${1:-${HARDWARE_NAME:-GeForce RTX 5090}}"

cd "$HARDWARE_GRAPH_REPO_ROOT"
wait_for_hardware_neo4j

runtime_config="$(mktemp /tmp/mlevolve-hardware-verify.XXXXXX.yaml)"
cleanup() {
  rm -f "$runtime_config"
}
trap cleanup EXIT
make_hardware_graph_runtime_config "$runtime_config"

"$HARDWARE_GRAPH_PYTHON" - "$runtime_config" "$hardware_name" <<'PY'
import json
import sys
from pathlib import Path

import yaml

from localml_scheduler.config import SchedulerConfig
from localml_scheduler.hardware_knowledge.store import HardwareKnowledgeGraphStore

config_path, hardware_name = sys.argv[1:3]

expected_by_stage = {
    "model_design": ["dataset_decomposition", "nvimagecodec_gpu_decode", "tensor_cores", "sm_120", "tensor_cores_5gen"],
    "datatype_precision": ["bf16", "fp8_rowwise_scaling"],
    "training_evaluation": ["muon_optimizer", "gram_newton_schulz_symmetric_gemm", "async_tensor_parallel"],
}

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
    rows = store._query_neighborhood_rows(
        hardware_terms=hardware_terms(hardware_name),
        row_limit=2048,
    )
    feature_ids = {
        str((row.get("feature") or {}).get("feature_id"))
        for row in rows
        if (row.get("feature") or {}).get("feature_id")
    }
    hardware_records = {
        str((row.get("hardware") or {}).get("hardware_id"))
        for row in rows
        if (row.get("hardware") or {}).get("hardware_id")
    }

    failures: list[str] = []
    if not rows:
        failures.append(f"no HAS_FEATURE rows returned for {hardware_name!r}")

    missing_by_stage = {}
    for stage, expected_ids in expected_by_stage.items():
        missing = sorted(set(expected_ids) - feature_ids)
        missing_by_stage[stage] = missing
        if missing:
            failures.append(f"{stage} missing feature ids: {', '.join(missing)}")

    search_results = store.search(
        query="tensor core optimizer precision dataloader",
        hardware=hardware_name.replace("NVIDIA ", "").replace("nvidia ", ""),
        limit=16,
    )
    if not search_results:
        failures.append("store.search returned no ranked records")

    summary = {
        "ok": not failures,
        "hardware_name": hardware_name,
        "matched_hardware_ids": sorted(hardware_records),
        "feature_count": len(feature_ids),
        "relationship_rows": len(rows),
        "expected_by_stage": expected_by_stage,
        "missing_by_stage": missing_by_stage,
        "ranked_search_count": len(search_results),
        "ranked_search_feature_ids": [
            item.get("feature_id") for item in search_results[:8]
        ],
    }
finally:
    driver = getattr(store, "_driver", None)
    if driver is not None:
        driver.close()

print(json.dumps(summary, indent=2, sort_keys=True))
if failures:
    print("\nFAIL:")
    for failure in failures:
        print(f"- {failure}")
    sys.exit(1)
PY

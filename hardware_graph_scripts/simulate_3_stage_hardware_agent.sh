#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hardware_graph_env.sh
source "$SCRIPT_DIR/hardware_graph_env.sh"

usage() {
  cat <<'EOF'
Usage:
  hardware_graph_scripts/simulate_3_stage_hardware_agent.sh [HARDWARE_NAME] [options]

Simulate a three-stage hardware optimization agent. The simulated agent calls
query_hardware_graph.sh for Hardware node and Feature node evidence, then
asserts that the returned payloads contain the expected stage-specific facts.

Three simulated stages:
  1. candidate_construction (hardware_context_lookup + data_processing + model_design)
  2. datatype_precision
  3. training_evaluation

Options:
  --db-check        Also verify the loaded Neo4j graph before checking payloads.
  --generic         Only require non-empty stage outputs; skip RTX 5090 exact IDs.
  --limit N         Feature limit passed to the query tool. Default: 24
  --out-dir PATH    Directory for captured query payloads. Default: temp dir.
  -h, --help        Show this help.

Default hardware: GeForce RTX 5090
EOF
}

hardware_name="${HARDWARE_NAME:-GeForce RTX 5090}"
if (($#)) && [[ "${1:-}" != -* ]]; then
  hardware_name="$1"
  shift
fi

db_check=0
strict_expected=1
limit=24
out_dir=""

while (($#)); do
  case "$1" in
    --db-check)
      db_check=1
      ;;
    --generic)
      strict_expected=0
      ;;
    --limit)
      if (($# < 2)); then
        echo "--limit requires a number." >&2
        exit 2
      fi
      limit="$2"
      shift
      ;;
    --out-dir)
      if (($# < 2)); then
        echo "--out-dir requires a path." >&2
        exit 2
      fi
      out_dir="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

cd "$HARDWARE_GRAPH_REPO_ROOT"

if (( strict_expected && limit < 24 )); then
  echo "Strict RTX 5090 checks require --limit 24 or higher so merged training_parameters expected features are present." >&2
  echo "Use --generic to run non-empty checks with a smaller limit." >&2
  exit 2
fi

if [[ -z "$out_dir" ]]; then
  out_dir="$(mktemp -d /tmp/mlevolve-hardware-agent.XXXXXX)"
else
  mkdir -p "$out_dir"
fi

query_tool="$SCRIPT_DIR/query_hardware_graph.sh"

if (( db_check )); then
  "$SCRIPT_DIR/verify_hardware_graph_db.sh" "$hardware_name" > "$out_dir/db_verification.json"
fi

calls=(
  "stage1_candidate_construction node datatype"
  "stage1_candidate_construction features datatype"
  "stage1_candidate_construction node model_structure"
  "stage1_candidate_construction features model_structure"
  "stage2_datatype_precision node datatype"
  "stage2_datatype_precision features datatype"
  "stage2_datatype_precision node training_parameters"
  "stage2_datatype_precision features training_parameters"
  "stage3_training_evaluation node training_parameters"
  "stage3_training_evaluation features training_parameters"
)

for call in "${calls[@]}"; do
  read -r phase tool stage <<<"$call"
  target="$out_dir/${phase}_${tool}_${stage}.json"
  echo "agent:${phase} call:${tool} stage:${stage} -> $target"
  "$query_tool" "$tool" "$hardware_name" "$stage" "$limit" > "$target"
done

HARDWARE_AGENT_STRICT_EXPECTED="$strict_expected" "$HARDWARE_GRAPH_PYTHON" - "$out_dir" "$hardware_name" <<'PY'
import json
import os
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
hardware_name = sys.argv[2]
strict = os.environ.get("HARDWARE_AGENT_STRICT_EXPECTED", "1") == "1"

feature_expectations = {
    "stage1_candidate_construction_features_datatype.json": [
        "dataset_decomposition",
        "nvimagecodec_gpu_decode",
    ],
    "stage1_candidate_construction_features_model_structure.json": [
        "tensor_cores",
        "sm_120",
        "tensor_cores_5gen",
    ],
    "stage2_datatype_precision_features_datatype.json": [
        "dataset_decomposition",
        "nvimagecodec_gpu_decode",
    ],
    "stage2_datatype_precision_features_training_parameters.json": [
        "bf16",
        "fp8_rowwise_scaling",
    ],
    "stage3_training_evaluation_features_training_parameters.json": [
        "muon_optimizer",
        "gram_newton_schulz_symmetric_gemm",
        "bf16",
        "fp8_rowwise_scaling",
        "async_tensor_parallel",
    ],
}

node_expectations = {
    "stage1_candidate_construction_node_datatype.json": {
        "stage_feature_keys": ["dataset_decomposition", "nvimagecodec_gpu_decode"],
    },
    "stage1_candidate_construction_node_model_structure.json": {
        "stage_feature_keys": ["tensor_cores", "sm_120", "tensor_cores_5gen"],
    },
    "stage2_datatype_precision_node_datatype.json": {
        "stage_feature_keys": ["dataset_decomposition", "nvimagecodec_gpu_decode"],
    },
    "stage2_datatype_precision_node_training_parameters.json": {
        "stage_feature_keys": ["bf16", "fp8_rowwise_scaling"],
    },
    "stage3_training_evaluation_node_training_parameters.json": {
        "stage_feature_keys": ["muon_optimizer", "gram_newton_schulz_symmetric_gemm"],
        "not_recommended_feature_keys": ["soap_optimizer", "ademamix_optimizer"],
    },
}

failures: list[str] = []
summary: dict[str, object] = {
    "hardware_name": hardware_name,
    "strict_expected_ids": strict,
    "payload_dir": str(out_dir),
    "stages": {},
}

def load_json(name: str) -> dict:
    path = out_dir / name
    if not path.exists():
        failures.append(f"missing payload {name}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        failures.append(f"could not parse {name}: {exc}")
        return {}

def contains_url(value) -> bool:
    if isinstance(value, str):
        return "http://" in value or "https://" in value
    if isinstance(value, list):
        return any(contains_url(item) for item in value)
    if isinstance(value, dict):
        return any(contains_url(item) for item in value.values())
    return False

def feature_key(value) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    return str(value)

def invalid_feature_key_pair(value) -> bool:
    return (
        not isinstance(value, list)
        or len(value) != 2
        or not isinstance(value[0], str)
        or not isinstance(value[1], str)
        or not value[0]
        or not value[1]
    )

for name, required_feature_ids in feature_expectations.items():
    payload = load_json(name)
    features = list(payload.get("features") or [])
    feature_ids = {
        str(feature.get("feature_id"))
        for feature in features
        if feature.get("feature_id")
    }
    stage = str(payload.get("stage_filter") or "all")
    summary["stages"][name] = {
        "stage_filter": stage,
        "found": bool(payload.get("found")),
        "feature_count": len(features),
        "feature_ids": sorted(feature_ids),
    }
    if not payload.get("found"):
        failures.append(f"{name} did not find hardware/features")
    if not features:
        failures.append(f"{name} returned no features")
    if strict:
        missing = sorted(set(required_feature_ids) - feature_ids)
        if missing:
            failures.append(f"{name} missing expected feature ids: {', '.join(missing)}")
    if contains_url(payload):
        failures.append(f"{name} contains an external URL")
    if "node_id" in payload or "source_urls" in payload:
        failures.append(f"{name} contains removed response fields")

for name, required_by_field in node_expectations.items():
    payload = load_json(name)
    summary["stages"][name] = {
        "stage_filter": payload.get("stage_filter"),
        "found": bool(payload.get("found")),
        "gpu_name": payload.get("gpu_name"),
    }
    if not payload.get("found"):
        failures.append(f"{name} did not find hardware node")
    for removed_key in ("node_id", "source_urls"):
        if removed_key in payload:
            failures.append(f"{name} contains removed field {removed_key}")
    if contains_url(payload):
        failures.append(f"{name} contains an external URL")
    if strict:
        for field, required_values in required_by_field.items():
            field_values = list(payload.get(field) or [])
            bad_pairs = [item for item in field_values if invalid_feature_key_pair(item)]
            if bad_pairs:
                failures.append(f"{name} field {field} contains non-2D feature-key rows")
            values = {feature_key(item) for item in field_values}
            missing = sorted(set(required_values) - values)
            if missing:
                failures.append(f"{name} field {field} missing: {', '.join(missing)}")

optimizer_payload = load_json("stage3_training_evaluation_features_training_parameters.json")
optimizer_by_id = {
    str(feature.get("feature_id")): feature
    for feature in optimizer_payload.get("features") or []
}
if strict and "soap_optimizer" in optimizer_by_id:
    soap = optimizer_by_id["soap_optimizer"]
    if soap.get("recommended") is not False:
        failures.append("soap_optimizer should remain an unconfirmed/non-recommended candidate")
    if soap.get("support_level") != "experimental":
        failures.append("soap_optimizer should have experimental support_level")

summary["ok"] = not failures
print(json.dumps(summary, indent=2, sort_keys=True))
if failures:
    print("\nFAIL:")
    for failure in failures:
        print(f"- {failure}")
    sys.exit(1)
print(f"\nPASS: three-stage hardware agent query simulation matched expected outputs in {out_dir}")
PY

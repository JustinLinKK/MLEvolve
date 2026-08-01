#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hardware_graph_env.sh
source "$SCRIPT_DIR/hardware_graph_env.sh"

usage() {
  cat <<'EOF'
Usage:
  hardware_graph_scripts/run_hardware_graph_preintegration.sh [HARDWARE_NAME] [options]

Run the full hardware graph pre-integration workflow:
  1. load schema/hardware_knowledge_graph.json into the hardware Neo4j DB
  2. verify the loaded DB can return expected hardware-feature edges
  3. simulate the three-stage agent query flow

Options:
  --recreate         Wipe existing hardware graph nodes before ingesting.
  --start-databases  Run ./docker_host_databases.sh up first.
  --generic          Skip RTX 5090 exact feature-id assertions.
  --limit N          Feature limit passed to the agent query simulation.
  --out-dir PATH     Directory for captured simulation payloads.
  -h, --help         Show this help.

Default hardware: GeForce RTX 5090
EOF
}

hardware_name="${HARDWARE_NAME:-GeForce RTX 5090}"
if (($#)) && [[ "${1:-}" != -* ]]; then
  hardware_name="$1"
  shift
fi

recreate=0
start_databases=0
generic=0
limit=16
out_dir=""

while (($#)); do
  case "$1" in
    --recreate)
      recreate=1
      ;;
    --start-databases)
      start_databases=1
      ;;
    --generic)
      generic=1
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

setup_args=(--execute)
if (( recreate )); then
  setup_args+=(--recreate)
else
  setup_args+=(--no-recreate)
fi
if (( start_databases )); then
  setup_args+=(--start-databases)
fi

"$SCRIPT_DIR/setup_hardware_graph_db.sh" "${setup_args[@]}"
"$SCRIPT_DIR/verify_hardware_graph_db.sh" "$hardware_name"

simulation_args=("$hardware_name" --db-check --limit "$limit")
if (( generic )); then
  simulation_args+=(--generic)
fi
if [[ -n "$out_dir" ]]; then
  simulation_args+=(--out-dir "$out_dir")
fi

"$SCRIPT_DIR/simulate_3_stage_hardware_agent.sh" "${simulation_args[@]}"

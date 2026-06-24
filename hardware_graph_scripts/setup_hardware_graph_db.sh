#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hardware_graph_env.sh
source "$SCRIPT_DIR/hardware_graph_env.sh"

usage() {
  cat <<'EOF'
Usage:
  hardware_graph_scripts/setup_hardware_graph_db.sh [options]

Set up and load the static hardware knowledge graph into Neo4j.

Options:
  --recreate             Wipe existing hardware graph nodes before ingesting.
  --no-recreate          Merge records without wiping first. Default.
  --dry-run              Validate and summarize schema records without writing.
  --execute              Write records to Neo4j. Default.
  --start-databases      Start/reuse the local hardware Neo4j Docker service.
  --no-wait              Do not wait for Neo4j before ingesting.
  --schema-root PATH     Schema root containing hardware_knowledge_graph.json.
  -h, --help             Show this help.

Environment:
  HARDWARE_GRAPH_DB_URI                         Default: bolt://127.0.0.1:7688
  LOCALML_SCHEDULER_HARDWARE_NEO4J_PASSWORD    Default: test12345
  HARDWARE_NEO4J_USERNAME                      Default: neo4j
  HARDWARE_NEO4J_DATABASE                      Default: neo4j
  MLEVOLVE_CONFIG                              Optional source config.
EOF
}

recreate=0
dry_run=0
start_databases=0
wait_db=1
schema_root="$HARDWARE_GRAPH_SCHEMA_ROOT"

start_hardware_neo4j() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is not available; cannot use --start-databases." >&2
    exit 127
  fi

  local compose_file="${COMPOSE_FILE:-$HARDWARE_GRAPH_REPO_ROOT/docker-compose.local.yml}"
  local compose_project="${COMPOSE_PROJECT_NAME:-mlevolve}"
  if [[ -f "$compose_file" ]] && docker compose version >/dev/null 2>&1; then
    NEO4J_HARDWARE_BOLT_PORT="${NEO4J_HARDWARE_BOLT_PORT:-7688}" \
    NEO4J_HARDWARE_HTTP_PORT="${NEO4J_HARDWARE_HTTP_PORT:-7475}" \
    LOCALML_SCHEDULER_HARDWARE_NEO4J_PASSWORD="$LOCALML_SCHEDULER_HARDWARE_NEO4J_PASSWORD" \
      docker compose -f "$compose_file" -p "$compose_project" up -d neo4j-hardware
    return 0
  fi

  local container="${NEO4J_HARDWARE_CONTAINER:-mlevolve-neo4j-hardware}"
  if docker container inspect "$container" >/dev/null 2>&1; then
    docker start "$container" >/dev/null
    return 0
  fi

  docker run -d \
    --name "$container" \
    --restart unless-stopped \
    -p "${NEO4J_HARDWARE_HTTP_PORT:-7475}:7474" \
    -p "${NEO4J_HARDWARE_BOLT_PORT:-7688}:7687" \
    -e "NEO4J_AUTH=neo4j/${LOCALML_SCHEDULER_HARDWARE_NEO4J_PASSWORD}" \
    -v "${NEO4J_HARDWARE_DATA_VOLUME:-mlevolve_neo4j_hardware_data}:/data" \
    -v "${NEO4J_HARDWARE_LOGS_VOLUME:-mlevolve_neo4j_hardware_logs}:/logs" \
    "${NEO4J_IMAGE:-neo4j:5.26}" >/dev/null
}

while (($#)); do
  case "$1" in
    --recreate)
      recreate=1
      ;;
    --no-recreate)
      recreate=0
      ;;
    --dry-run)
      dry_run=1
      ;;
    --execute|--no-dry-run)
      dry_run=0
      ;;
    --start-databases)
      start_databases=1
      ;;
    --no-wait)
      wait_db=0
      ;;
    --schema-root)
      if (($# < 2)); then
        echo "--schema-root requires a path." >&2
        exit 2
      fi
      schema_root="$2"
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
require_hardware_graph_file "$schema_root/hardware_knowledge_graph.json" "hardware graph JSON"

if (( start_databases )); then
  start_hardware_neo4j
fi

if (( ! dry_run && wait_db )); then
  wait_for_hardware_neo4j
fi

runtime_config="$(mktemp /tmp/mlevolve-hardware-graph.XXXXXX.yaml)"
cleanup() {
  rm -f "$runtime_config"
}
trap cleanup EXIT

make_hardware_graph_runtime_config "$runtime_config"

ingest_args=(
  "$HARDWARE_GRAPH_PYTHON" -m localml_scheduler.cli hardware-knowledge ingest
  --config "$runtime_config"
  --schema-root "$schema_root"
)
if (( recreate )); then
  ingest_args+=(--recreate)
else
  ingest_args+=(--no-recreate)
fi
if (( dry_run )); then
  ingest_args+=(--dry-run)
else
  ingest_args+=(--no-dry-run)
fi

"${ingest_args[@]}"

if (( ! dry_run )); then
"$HARDWARE_GRAPH_PYTHON" - "$runtime_config" <<'PY'
import json
import sys
from pathlib import Path

import yaml

from localml_scheduler.config import SchedulerConfig
from localml_scheduler.hardware_knowledge.store import HardwareKnowledgeGraphStore

payload = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
settings_payload = payload.get("scheduler", {}).get("settings", payload)
settings = SchedulerConfig.from_dict(settings_payload)
store = HardwareKnowledgeGraphStore(settings)
rows = store._run(
    """
    MATCH (h:HardwareSpec)
    WITH count(h) AS hardware_count
    MATCH (f:Feature)
    WITH hardware_count, count(f) AS feature_count
    MATCH (linked:HardwareSpec)-[r:HAS_FEATURE]->(:Feature)
    RETURN hardware_count,
           feature_count,
           count(DISTINCT linked) AS linked_hardware_count,
           count(r) AS relationship_count
    """
)
print(json.dumps({"loaded_counts": rows[0] if rows else {}}, indent=2, sort_keys=True))
driver = getattr(store, "_driver", None)
if driver is not None:
    driver.close()
PY
fi

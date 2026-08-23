#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CONFIG_PATH="${MLEVOLVE_CONFIG:-$ROOT/config.yaml}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  CONFIG_PATH="$ROOT/config.example.yaml"
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "No config file found. Expected $ROOT/config.yaml or $ROOT/config.example.yaml." >&2
  exit 2
fi

export HARDWARE_KNOWLEDGE_NEO4J_PASSWORD="${HARDWARE_KNOWLEDGE_NEO4J_PASSWORD:-test12345}"

read_config_value() {
  local dotted_key="$1"
  local default_value="$2"
  python - "$CONFIG_PATH" "$dotted_key" "$default_value" <<'PY'
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
dotted_key = sys.argv[2]
default = sys.argv[3]

try:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    current = data
    for part in dotted_key.split("."):
        current = current[part]
except Exception:
    current = default

print(default if current is None else current)
PY
}

CONFIG_HARDWARE_GRAPH_URI="$(read_config_value hardware_knowledge.settings.graph.uri bolt://127.0.0.1:7688)"
CONFIG_HARDWARE_NEO4J_USERNAME="$(read_config_value hardware_knowledge.settings.graph.username neo4j)"
CONFIG_HARDWARE_NEO4J_DATABASE="$(read_config_value hardware_knowledge.settings.graph.database neo4j)"

export HARDWARE_GRAPH_URI="${HARDWARE_GRAPH_URI:-${HARDWARE_GRAPH_DB_URI:-$CONFIG_HARDWARE_GRAPH_URI}}"
export HARDWARE_NEO4J_USERNAME="${HARDWARE_NEO4J_USERNAME:-$CONFIG_HARDWARE_NEO4J_USERNAME}"
export HARDWARE_NEO4J_DATABASE="${HARDWARE_NEO4J_DATABASE:-$CONFIG_HARDWARE_NEO4J_DATABASE}"

TEMP_CONFIG=""
cleanup() {
  if [[ -n "$TEMP_CONFIG" && -f "$TEMP_CONFIG" ]]; then
    rm -f "$TEMP_CONFIG"
  fi
}
trap cleanup EXIT

write_runtime_config() {
  TEMP_CONFIG="$(mktemp /tmp/mlevolve-config.XXXXXX.yaml)"
  python - "$CONFIG_PATH" "$TEMP_CONFIG" <<'PY'
import os
import sys
from pathlib import Path

import yaml

source = Path(sys.argv[1])
target = Path(sys.argv[2])
data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}

hardware_settings = data.setdefault("hardware_knowledge", {}).setdefault("settings", {})
graph = hardware_settings.setdefault("graph", {})
graph["enabled"] = True
graph["provider"] = "neo4j"
graph["uri"] = os.environ["HARDWARE_GRAPH_URI"]
graph["username"] = os.environ["HARDWARE_NEO4J_USERNAME"]
graph["database"] = os.environ["HARDWARE_NEO4J_DATABASE"]
graph["password_env"] = "HARDWARE_KNOWLEDGE_NEO4J_PASSWORD"

target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
PY
  export MLEVOLVE_CONFIG="$TEMP_CONFIG"
}

start_local_databases() {
  if ! command -v docker >/dev/null 2>&1; then
    cat >&2 <<'EOF'
Docker is not available in this shell.

Point HARDWARE_GRAPH_URI and HARDWARE_KNOWLEDGE_NEO4J_PASSWORD at an
existing Neo4j instance, or run this script from a host/devcontainer with Docker
access and MLEVOLVE_START_LOCAL_DATABASES=1.
EOF
    exit 127
  fi

  docker start mlevolve-neo4j-hardware >/dev/null 2>&1 || \
  docker run -d \
    --name mlevolve-neo4j-hardware \
    --restart unless-stopped \
    -p 7475:7474 -p 7688:7687 \
    -e NEO4J_AUTH="neo4j/${HARDWARE_KNOWLEDGE_NEO4J_PASSWORD:-test12345}" \
    -v mlevolve_neo4j_hardware_data:/data \
    -v mlevolve_neo4j_hardware_logs:/logs \
    neo4j:5.26
}

wait_for_hardware_neo4j() {
  local waited=0
  local max_wait="${NEO4J_WAIT_SECONDS:-60}"

  until python - <<'PY' >/dev/null 2>&1
import os

from neo4j import GraphDatabase

uri = os.environ["HARDWARE_GRAPH_URI"]
username = os.environ.get("HARDWARE_NEO4J_USERNAME", "")
password = os.environ.get("HARDWARE_KNOWLEDGE_NEO4J_PASSWORD", "")
database = os.environ.get("HARDWARE_NEO4J_DATABASE") or None
auth = (username, password) if username else None

driver = GraphDatabase.driver(uri, auth=auth)
try:
    driver.verify_connectivity()
    with driver.session(database=database) as session:
        session.run("RETURN 1").consume()
finally:
    driver.close()
PY
  do
    if (( waited >= max_wait )); then
      cat >&2 <<EOF
Hardware Neo4j did not become reachable within ${max_wait}s.

Tried: $HARDWARE_GRAPH_URI
User:  $HARDWARE_NEO4J_USERNAME

If it is running on your host from inside a devcontainer, try:
  HARDWARE_GRAPH_URI=bolt://host.docker.internal:7688 ./bootstrap.sh
EOF
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
}

ingest_hardware_knowledge() {
  if [[ ! -f schema/hardware_knowledge_graph.json ]]; then
    echo "Expected schema/hardware_knowledge_graph.json." >&2
    return 1
  fi
  local -a ingest_args=(
    python -m hardware_knowledge_graph.cli ingest
    --config "$MLEVOLVE_CONFIG"
    --schema-root schema
  )
  if [[ "${MLEVOLVE_RECREATE_KNOWLEDGE:-0}" == "1" ]]; then
    ingest_args+=(--recreate)
  else
    ingest_args+=(--no-recreate)
  fi
  "${ingest_args[@]}"
}

if [[ "${MLEVOLVE_START_LOCAL_DATABASES:-0}" == "1" ]]; then
  start_local_databases
fi

echo "Using hardware knowledge database endpoint:"
echo "  Neo4j hardware: $HARDWARE_GRAPH_URI"

wait_for_hardware_neo4j

if [[ "$HARDWARE_GRAPH_URI" != "$CONFIG_HARDWARE_GRAPH_URI" ||
      "$HARDWARE_NEO4J_USERNAME" != "$CONFIG_HARDWARE_NEO4J_USERNAME" ||
      "$HARDWARE_NEO4J_DATABASE" != "$CONFIG_HARDWARE_NEO4J_DATABASE" ]]; then
  write_runtime_config
else
  export MLEVOLVE_CONFIG="$CONFIG_PATH"
fi

if [[ "${MLEVOLVE_INGEST_KNOWLEDGE:-0}" == "1" ]]; then
  ingest_hardware_knowledge
fi

echo "Hardware knowledge setup complete. Run the trace benchmark with:"
echo "  bash scheduler_benchmark_test/run_histopath_scheduler_replay.sh"

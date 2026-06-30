#!/usr/bin/env bash
# Shared environment helpers for hardware graph setup and smoke tests.

set -Eeuo pipefail

HARDWARE_GRAPH_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARDWARE_GRAPH_REPO_ROOT="$(cd "$HARDWARE_GRAPH_SCRIPT_DIR/.." && pwd)"

export PYTHONPATH="$HARDWARE_GRAPH_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LOCALML_SCHEDULER_HARDWARE_NEO4J_PASSWORD="${LOCALML_SCHEDULER_HARDWARE_NEO4J_PASSWORD:-test12345}"
if [[ -z "${HARDWARE_GRAPH_DB_URI:-}" ]]; then
  if [[ -f /.dockerenv ]]; then
    HARDWARE_GRAPH_ACCESS_HOST="${HARDWARE_GRAPH_ACCESS_HOST:-host.docker.internal}"
  else
    HARDWARE_GRAPH_ACCESS_HOST="${HARDWARE_GRAPH_ACCESS_HOST:-127.0.0.1}"
  fi
  export HARDWARE_GRAPH_DB_URI="bolt://${HARDWARE_GRAPH_ACCESS_HOST}:${NEO4J_HARDWARE_BOLT_PORT:-7688}"
fi
export HARDWARE_NEO4J_USERNAME="${HARDWARE_NEO4J_USERNAME:-neo4j}"
export HARDWARE_NEO4J_DATABASE="${HARDWARE_NEO4J_DATABASE:-neo4j}"
export HARDWARE_GRAPH_SCHEMA_ROOT="${HARDWARE_GRAPH_SCHEMA_ROOT:-$HARDWARE_GRAPH_REPO_ROOT/schema}"

if [[ -n "${MLEVOLVE_CONFIG:-}" ]]; then
  HARDWARE_GRAPH_CONFIG_SOURCE="$MLEVOLVE_CONFIG"
elif [[ -f "$HARDWARE_GRAPH_REPO_ROOT/config.yaml" ]]; then
  HARDWARE_GRAPH_CONFIG_SOURCE="$HARDWARE_GRAPH_REPO_ROOT/config.yaml"
else
  HARDWARE_GRAPH_CONFIG_SOURCE="$HARDWARE_GRAPH_REPO_ROOT/config.example.yaml"
fi
export HARDWARE_GRAPH_CONFIG_SOURCE

hardware_graph_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    printf '%s\n' "$PYTHON"
    return 0
  fi
  if [[ -x "$HARDWARE_GRAPH_REPO_ROOT/.venv/bin/python" ]]; then
    printf '%s\n' "$HARDWARE_GRAPH_REPO_ROOT/.venv/bin/python"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  command -v python
}

HARDWARE_GRAPH_PYTHON="$(hardware_graph_python)"
export HARDWARE_GRAPH_PYTHON

require_hardware_graph_file() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "Missing ${label}: $path" >&2
    exit 2
  fi
}

make_hardware_graph_runtime_config() {
  local target="$1"
  require_hardware_graph_file "$HARDWARE_GRAPH_CONFIG_SOURCE" "config source"
  "$HARDWARE_GRAPH_PYTHON" - "$HARDWARE_GRAPH_CONFIG_SOURCE" "$target" <<'PY'
import os
import sys
from pathlib import Path

import yaml

source = Path(sys.argv[1])
target = Path(sys.argv[2])
data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}

settings = data.setdefault("scheduler", {}).setdefault("settings", {})

# Keep these scripts focused on the static hardware knowledge graph. The
# profile/evidence graph can be unavailable during pre-integration checks.
graph_db = settings.setdefault("graph_db", {})
graph_db["enabled"] = False
graph_db["mode"] = "off"

redis_cache = settings.setdefault("redis_cache", {})
redis_cache["enabled"] = False

hardware_graph = settings.setdefault("hardware_knowledge_graph", {})
hardware_graph["enabled"] = True
hardware_graph["provider"] = "neo4j"
hardware_graph["uri"] = os.environ["HARDWARE_GRAPH_DB_URI"]
hardware_graph["username"] = os.environ["HARDWARE_NEO4J_USERNAME"]
hardware_graph["database"] = os.environ["HARDWARE_NEO4J_DATABASE"]
hardware_graph["password_env"] = "LOCALML_SCHEDULER_HARDWARE_NEO4J_PASSWORD"

target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
PY
}

wait_for_hardware_neo4j() {
  local waited=0
  local max_wait="${NEO4J_WAIT_SECONDS:-60}"

  until "$HARDWARE_GRAPH_PYTHON" - <<'PY' >/dev/null 2>&1
import os

from neo4j import GraphDatabase

uri = os.environ["HARDWARE_GRAPH_DB_URI"]
username = os.environ.get("HARDWARE_NEO4J_USERNAME", "")
password = os.environ.get("LOCALML_SCHEDULER_HARDWARE_NEO4J_PASSWORD", "")
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

Tried: $HARDWARE_GRAPH_DB_URI
User:  $HARDWARE_NEO4J_USERNAME

Start local databases with:
  ./docker_host_databases.sh up

Or point HARDWARE_GRAPH_DB_URI and LOCALML_SCHEDULER_HARDWARE_NEO4J_PASSWORD at
an existing Neo4j instance.
EOF
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
}

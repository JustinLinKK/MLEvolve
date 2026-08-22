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
CONFIG_LESSON_PROFILES_ENABLED="$(read_config_value lesson_profiles.enabled true)"
CONFIG_LESSON_QDRANT_URL="$(read_config_value lesson_profiles.qdrant.url http://127.0.0.1:6333)"
CONFIG_LESSON_REDIS_URL="$(read_config_value lesson_profiles.redis_cache.url redis://127.0.0.1:6379/1)"

export HARDWARE_GRAPH_URI="${HARDWARE_GRAPH_URI:-${HARDWARE_GRAPH_DB_URI:-$CONFIG_HARDWARE_GRAPH_URI}}"
export HARDWARE_NEO4J_USERNAME="${HARDWARE_NEO4J_USERNAME:-$CONFIG_HARDWARE_NEO4J_USERNAME}"
export HARDWARE_NEO4J_DATABASE="${HARDWARE_NEO4J_DATABASE:-$CONFIG_HARDWARE_NEO4J_DATABASE}"
export LESSON_QDRANT_URL="${LESSON_QDRANT_URL:-$CONFIG_LESSON_QDRANT_URL}"
export MLEVOLVE_LESSON_REDIS_URL="${MLEVOLVE_LESSON_REDIS_URL:-$CONFIG_LESSON_REDIS_URL}"

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

lesson_profiles = data.setdefault("lesson_profiles", {})
lesson_profiles.setdefault("qdrant", {})["url"] = os.environ["LESSON_QDRANT_URL"]
lesson_profiles.setdefault("redis_cache", {})["url"] = os.environ["MLEVOLVE_LESSON_REDIS_URL"]

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

  "$ROOT/docker_host_databases.sh" up
}

wait_for_lesson_services() {
  local waited=0
  local max_wait="${LESSON_SERVICE_WAIT_SECONDS:-60}"
  until python - <<'PY' >/dev/null 2>&1
import os
import urllib.request

import redis

with urllib.request.urlopen(os.environ["LESSON_QDRANT_URL"].rstrip("/") + "/collections", timeout=2) as response:
    if response.status >= 400:
        raise RuntimeError(response.status)
redis.Redis.from_url(os.environ["MLEVOLVE_LESSON_REDIS_URL"], socket_timeout=2).ping()
PY
  do
    if (( waited >= max_wait )); then
      echo "Lesson Qdrant/Redis services did not become reachable within ${max_wait}s." >&2
      echo "Qdrant: $LESSON_QDRANT_URL" >&2
      echo "Redis:   $MLEVOLVE_LESSON_REDIS_URL" >&2
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
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
if [[ "${CONFIG_LESSON_PROFILES_ENABLED,,}" == "true" ]]; then
  echo "  Lesson Qdrant:  $LESSON_QDRANT_URL"
  echo "  Lesson Redis:   $MLEVOLVE_LESSON_REDIS_URL"
fi

wait_for_hardware_neo4j
if [[ "${CONFIG_LESSON_PROFILES_ENABLED,,}" == "true" ]]; then
  wait_for_lesson_services
fi

if [[ "$HARDWARE_GRAPH_URI" != "$CONFIG_HARDWARE_GRAPH_URI" ||
      "$HARDWARE_NEO4J_USERNAME" != "$CONFIG_HARDWARE_NEO4J_USERNAME" ||
      "$HARDWARE_NEO4J_DATABASE" != "$CONFIG_HARDWARE_NEO4J_DATABASE" ||
      "$LESSON_QDRANT_URL" != "$CONFIG_LESSON_QDRANT_URL" ||
      "$MLEVOLVE_LESSON_REDIS_URL" != "$CONFIG_LESSON_REDIS_URL" ]]; then
  write_runtime_config
else
  export MLEVOLVE_CONFIG="$CONFIG_PATH"
fi

if [[ "${MLEVOLVE_INGEST_KNOWLEDGE:-0}" == "1" ]]; then
  ingest_hardware_knowledge
fi

if [[ "${CONFIG_LESSON_PROFILES_ENABLED,,}" == "true" && "${MLEVOLVE_INIT_LESSON_PROFILES:-1}" == "1" ]]; then
  python -m lesson_profile_database.cli --config "$MLEVOLVE_CONFIG" init
fi

echo "Hardware knowledge and lesson profile database setup complete. Run the trace benchmark with:"
echo "  bash scheduler_benchmark_test/run_histopath_scheduler_replay.sh"

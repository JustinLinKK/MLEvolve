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

export LOCALML_SCHEDULER_NEO4J_PASSWORD="${LOCALML_SCHEDULER_NEO4J_PASSWORD:-test12345}"
export LOCALML_SCHEDULER_HARDWARE_NEO4J_PASSWORD="${LOCALML_SCHEDULER_HARDWARE_NEO4J_PASSWORD:-test12345}"

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

CONFIG_QDRANT_URL="$(read_config_value scheduler.settings.hardware_feature_db.url http://127.0.0.1:6333)"
CONFIG_QDRANT_API_KEY_ENV="$(read_config_value scheduler.settings.hardware_feature_db.api_key_env LOCALML_SCHEDULER_QDRANT_API_KEY)"
CONFIG_HARDWARE_COLLECTION="$(read_config_value scheduler.settings.hardware_feature_db.collection_name hardware_feature_knowledge)"
CONFIG_CODE_DOC_COLLECTION="$(read_config_value scheduler.settings.hardware_feature_db.code_doc_collection_name code_doc_chunks)"
CONFIG_RECIPE_COLLECTION="$(read_config_value scheduler.settings.hardware_feature_db.optimization_recipe_collection_name optimization_recipe_chunks)"
CONFIG_API_SYMBOL_COLLECTION="$(read_config_value scheduler.settings.hardware_feature_db.api_symbol_collection_name api_symbol_chunks)"
CONFIG_GRAPH_DB_URI="$(read_config_value scheduler.settings.graph_db.uri bolt://127.0.0.1:7687)"
CONFIG_NEO4J_USERNAME="$(read_config_value scheduler.settings.graph_db.username neo4j)"
CONFIG_NEO4J_DATABASE="$(read_config_value scheduler.settings.graph_db.database neo4j)"
CONFIG_HARDWARE_GRAPH_DB_URI="$(read_config_value scheduler.settings.hardware_knowledge_graph.uri bolt://127.0.0.1:7688)"
CONFIG_HARDWARE_NEO4J_USERNAME="$(read_config_value scheduler.settings.hardware_knowledge_graph.username neo4j)"
CONFIG_HARDWARE_NEO4J_DATABASE="$(read_config_value scheduler.settings.hardware_knowledge_graph.database neo4j)"

export QDRANT_URL="${QDRANT_URL:-$CONFIG_QDRANT_URL}"
export QDRANT_API_KEY="${QDRANT_API_KEY:-}"
if [[ -z "$QDRANT_API_KEY" && -n "$CONFIG_QDRANT_API_KEY_ENV" ]]; then
  export QDRANT_API_KEY="${!CONFIG_QDRANT_API_KEY_ENV:-}"
fi
export HARDWARE_COLLECTION="${HARDWARE_COLLECTION:-$CONFIG_HARDWARE_COLLECTION}"
export CODE_DOC_COLLECTION="${CODE_DOC_COLLECTION:-$CONFIG_CODE_DOC_COLLECTION}"
export RECIPE_COLLECTION="${RECIPE_COLLECTION:-$CONFIG_RECIPE_COLLECTION}"
export API_SYMBOL_COLLECTION="${API_SYMBOL_COLLECTION:-$CONFIG_API_SYMBOL_COLLECTION}"
export GRAPH_DB_URI="${GRAPH_DB_URI:-$CONFIG_GRAPH_DB_URI}"
export NEO4J_USERNAME="${NEO4J_USERNAME:-$CONFIG_NEO4J_USERNAME}"
export NEO4J_DATABASE="${NEO4J_DATABASE:-$CONFIG_NEO4J_DATABASE}"
export HARDWARE_GRAPH_DB_URI="${HARDWARE_GRAPH_DB_URI:-$CONFIG_HARDWARE_GRAPH_DB_URI}"
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

settings = data.setdefault("scheduler", {}).setdefault("settings", {})
graph_db = settings.setdefault("graph_db", {})
hardware_knowledge_graph = settings.setdefault("hardware_knowledge_graph", {})
hardware_feature_db = settings.setdefault("hardware_feature_db", {})

graph_db["uri"] = os.environ["GRAPH_DB_URI"]
graph_db["username"] = os.environ["NEO4J_USERNAME"]
graph_db["database"] = os.environ["NEO4J_DATABASE"]
hardware_knowledge_graph["uri"] = os.environ["HARDWARE_GRAPH_DB_URI"]
hardware_knowledge_graph["username"] = os.environ["HARDWARE_NEO4J_USERNAME"]
hardware_knowledge_graph["database"] = os.environ["HARDWARE_NEO4J_DATABASE"]
hardware_feature_db["url"] = os.environ["QDRANT_URL"]
hardware_feature_db["collection_name"] = os.environ["HARDWARE_COLLECTION"]
hardware_feature_db["code_doc_collection_name"] = os.environ["CODE_DOC_COLLECTION"]
hardware_feature_db["optimization_recipe_collection_name"] = os.environ["RECIPE_COLLECTION"]
hardware_feature_db["api_symbol_collection_name"] = os.environ["API_SYMBOL_COLLECTION"]

target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
PY
  export MLEVOLVE_CONFIG="$TEMP_CONFIG"
}

start_local_databases() {
  if ! command -v docker >/dev/null 2>&1; then
    cat >&2 <<'EOF'
Docker is not available in this shell.

This bootstrap now uses existing Qdrant/Neo4j services by default. To start local
containers with this script, run it from a host/devcontainer with Docker access.
EOF
    exit 127
  fi

  docker start mlevolve-qdrant >/dev/null 2>&1 || \
  docker run -d \
    --name mlevolve-qdrant \
    --restart unless-stopped \
    -p 6333:6333 -p 6334:6334 \
    -v mlevolve_qdrant:/qdrant/storage \
    qdrant/qdrant

  docker start mlevolve-neo4j-profile >/dev/null 2>&1 || \
  docker run -d \
    --name mlevolve-neo4j-profile \
    --restart unless-stopped \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH="neo4j/${LOCALML_SCHEDULER_NEO4J_PASSWORD}" \
    -v mlevolve_neo4j_profile_data:/data \
    -v mlevolve_neo4j_profile_logs:/logs \
    neo4j:5.26

  docker start mlevolve-neo4j-hardware >/dev/null 2>&1 || \
  docker run -d \
    --name mlevolve-neo4j-hardware \
    --restart unless-stopped \
    -p 7475:7474 -p 7688:7687 \
    -e NEO4J_AUTH="neo4j/${LOCALML_SCHEDULER_HARDWARE_NEO4J_PASSWORD:-test12345}" \
    -v mlevolve_neo4j_hardware_data:/data \
    -v mlevolve_neo4j_hardware_logs:/logs \
    neo4j:5.26
}

qdrant_curl() {
  local url="$1"
  if [[ -n "${QDRANT_API_KEY:-}" ]]; then
    curl -fsS -H "api-key: ${QDRANT_API_KEY}" "$url"
  else
    curl -fsS "$url"
  fi
}

wait_for_qdrant() {
  local waited=0
  local max_wait="${QDRANT_WAIT_SECONDS:-30}"
  local collections_url="${QDRANT_URL%/}/collections"

  until qdrant_curl "$collections_url" >/dev/null; do
    if (( waited >= max_wait )); then
      cat >&2 <<EOF
Qdrant did not become reachable within ${max_wait}s.

Tried: $collections_url

If the existing Qdrant is running on your host, try:
  QDRANT_URL=http://host.docker.internal:6333 ./bootstrap.sh

If it is in Kubernetes, port-forward it first, then point QDRANT_URL at the
forwarded address.
EOF
      return 1
    fi
    sleep 1
    waited=$((waited + 1))
  done
}

vector_db_is_empty() {
  python - <<'PY'
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

base_url = os.environ["QDRANT_URL"].rstrip("/")
api_key = os.environ.get("QDRANT_API_KEY", "")
collections = [
    os.environ["HARDWARE_COLLECTION"],
    os.environ["CODE_DOC_COLLECTION"],
    os.environ["RECIPE_COLLECTION"],
    os.environ["API_SYMBOL_COLLECTION"],
]

headers = {"Content-Type": "application/json"}
if api_key:
    headers["api-key"] = api_key

counts: dict[str, int] = {}
missing: list[str] = []

try:
    for collection_name in collections:
        quoted_name = urllib.parse.quote(collection_name, safe="")
        url = f"{base_url}/collections/{quoted_name}/points/count"
        request = urllib.request.Request(
            url,
            data=b'{"exact": true}',
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                missing.append(collection_name)
                counts[collection_name] = 0
                continue
            raise

        result = payload.get("result", {})
        counts[collection_name] = int(result.get("count") or 0)
except Exception as exc:
    print(f"Could not inspect Qdrant collections: {type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(2)

populated = {name: count for name, count in counts.items() if count > 0}
if populated:
    summary = ", ".join(f"{name}={count}" for name, count in populated.items())
    print(f"Qdrant already has knowledge points: {summary}")
    sys.exit(1)

summary = ", ".join(f"{name}=0" for name in collections)
if missing:
    print(f"Qdrant knowledge collections are empty or missing: {summary}; missing={', '.join(missing)}")
else:
    print(f"Qdrant knowledge collections are empty: {summary}")
sys.exit(0)
PY
}

wait_for_neo4j() {
  local uri="$1"
  local username="$2"
  local password="$3"
  local database="$4"
  local label="$5"
  local waited=0
  local max_wait="${NEO4J_WAIT_SECONDS:-60}"

  until GRAPH_DB_URI="$uri" NEO4J_USERNAME="$username" LOCALML_SCHEDULER_NEO4J_PASSWORD="$password" NEO4J_DATABASE="$database" python - <<'PY' >/dev/null 2>&1
import os

from neo4j import GraphDatabase

uri = os.environ["GRAPH_DB_URI"]
username = os.environ.get("NEO4J_USERNAME", "")
password = os.environ.get("LOCALML_SCHEDULER_NEO4J_PASSWORD", "")
database = os.environ.get("NEO4J_DATABASE") or None
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
Neo4j (${label}) did not become reachable within ${max_wait}s.

Tried: $uri
User:  $username

If the existing Neo4j is running on your host, try:
  GRAPH_DB_URI=bolt://host.docker.internal:7687 HARDWARE_GRAPH_DB_URI=bolt://host.docker.internal:7688 ./bootstrap.sh

If it is in Kubernetes, port-forward it first, then point GRAPH_DB_URI at the
forwarded address. Also make sure LOCALML_SCHEDULER_NEO4J_PASSWORD is set to the
password for that existing database.
EOF
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
}

ensure_schema_knowledge_dirs() {
  if [[ -f schema/hardware_knowledge_graph.json ]]; then
    return 0
  fi
  if [[ -d schema/api_symbol_chunks || -d schema/code_doc_chunks || -d schema/hardware_feature_records ]]; then
    return 0
  fi
  echo "Expected schema/hardware_knowledge_graph.json or legacy knowledge directories under schema/." >&2
  return 1
}

ingest_schema_knowledge() {
  ensure_schema_knowledge_dirs
  local -a ingest_args=(
    python -m localml_scheduler.cli knowledge ingest-schema
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

echo "Using existing database endpoints:"
echo "  Qdrant: $QDRANT_URL"
echo "  Neo4j profile:  $GRAPH_DB_URI"
echo "  Neo4j hardware: $HARDWARE_GRAPH_DB_URI"

wait_for_qdrant
wait_for_neo4j "$GRAPH_DB_URI" "$NEO4J_USERNAME" "${LOCALML_SCHEDULER_NEO4J_PASSWORD:-}" "$NEO4J_DATABASE" "profile"
wait_for_neo4j "$HARDWARE_GRAPH_DB_URI" "$HARDWARE_NEO4J_USERNAME" "${LOCALML_SCHEDULER_HARDWARE_NEO4J_PASSWORD:-test12345}" "$HARDWARE_NEO4J_DATABASE" "hardware"

if [[ "$QDRANT_URL" != "$CONFIG_QDRANT_URL" ||
      "$HARDWARE_COLLECTION" != "$CONFIG_HARDWARE_COLLECTION" ||
      "$CODE_DOC_COLLECTION" != "$CONFIG_CODE_DOC_COLLECTION" ||
      "$RECIPE_COLLECTION" != "$CONFIG_RECIPE_COLLECTION" ||
      "$API_SYMBOL_COLLECTION" != "$CONFIG_API_SYMBOL_COLLECTION" ||
      "$GRAPH_DB_URI" != "$CONFIG_GRAPH_DB_URI" ||
      "$NEO4J_USERNAME" != "$CONFIG_NEO4J_USERNAME" ||
      "$NEO4J_DATABASE" != "$CONFIG_NEO4J_DATABASE" ||
      "$HARDWARE_GRAPH_DB_URI" != "$CONFIG_HARDWARE_GRAPH_DB_URI" ||
      "$HARDWARE_NEO4J_USERNAME" != "$CONFIG_HARDWARE_NEO4J_USERNAME" ||
      "$HARDWARE_NEO4J_DATABASE" != "$CONFIG_HARDWARE_NEO4J_DATABASE" ]]; then
  write_runtime_config
else
  export MLEVOLVE_CONFIG="$CONFIG_PATH"
fi

if [[ "${MLEVOLVE_INGEST_KNOWLEDGE:-0}" == "1" ]]; then
  ingest_schema_knowledge
else
  set +e
  vector_db_is_empty
  vector_status=$?
  set -e
  case "$vector_status" in
    0)
      if [[ "${MLEVOLVE_AUTO_INGEST_IF_EMPTY:-1}" == "1" ]]; then
        echo "Qdrant is empty; ingesting schema knowledge without recreating collections."
        MLEVOLVE_RECREATE_KNOWLEDGE=0 ingest_schema_knowledge
      else
        echo "Qdrant is empty; auto-ingest disabled by MLEVOLVE_AUTO_INGEST_IF_EMPTY=0."
      fi
      ;;
    1)
      echo "Skipping knowledge ingestion; existing Qdrant data is left untouched."
      ;;
    *)
      echo "Could not determine whether Qdrant is empty." >&2
      exit "$vector_status"
      ;;
  esac
fi

if [[ "${MLEVOLVE_RUN_COMPARE:-1}" != "1" ]]; then
  echo "Database checks passed. Set MLEVOLVE_RUN_COMPARE=1 to run the benchmark."
  exit 0
fi

COMPETITION_ID="${MLEVOLVE_COMPETITION_ID:-histopathologic-cancer-detection}"
DATASET_ROOT="${MLEVOLVE_DATASET_ROOT:-$ROOT/data/mle-bench}"
STEPS="${MLEVOLVE_STEPS:-5}"
INITIAL_DRAFTS="${MLEVOLVE_INITIAL_DRAFTS:-3}"
TIMEOUT_SECONDS="${MLEVOLVE_TIMEOUT_SECONDS:-0}"
MEMORY_INDEX="${MLEVOLVE_MEMORY_INDEX:-0}"

bash compare_hardware_awareness.sh "$COMPETITION_ID" \
  --config "$MLEVOLVE_CONFIG" \
  --dataset-root "$DATASET_ROOT" \
  --steps "$STEPS" \
  --initial-drafts "$INITIAL_DRAFTS" \
  --timeout-seconds "$TIMEOUT_SECONDS" \
  --memory-index "$MEMORY_INDEX"

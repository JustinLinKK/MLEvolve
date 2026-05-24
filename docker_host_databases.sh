#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-$ROOT/docker-compose.local.yml}"
COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-mlevolve}"
QDRANT_CONTAINER="${QDRANT_CONTAINER:-mlevolve-qdrant}"
NEO4J_CONTAINER="${NEO4J_CONTAINER:-mlevolve-neo4j}"
QDRANT_IMAGE="${QDRANT_IMAGE:-qdrant/qdrant}"
NEO4J_IMAGE="${NEO4J_IMAGE:-neo4j:5.26}"
QDRANT_VOLUME="${QDRANT_VOLUME:-mlevolve_qdrant}"
NEO4J_DATA_VOLUME="${NEO4J_DATA_VOLUME:-mlevolve_neo4j_data}"
NEO4J_LOGS_VOLUME="${NEO4J_LOGS_VOLUME:-mlevolve_neo4j_logs}"
QDRANT_HTTP_PORT="${QDRANT_HTTP_PORT:-6333}"
QDRANT_GRPC_PORT="${QDRANT_GRPC_PORT:-6334}"
NEO4J_HTTP_PORT="${NEO4J_HTTP_PORT:-7474}"
NEO4J_BOLT_PORT="${NEO4J_BOLT_PORT:-7687}"
LOCALML_SCHEDULER_NEO4J_PASSWORD="${LOCALML_SCHEDULER_NEO4J_PASSWORD:-test12345}"
if [[ -f /.dockerenv ]]; then
  DOCKER_ACCESS_HOST="${DOCKER_ACCESS_HOST:-host.docker.internal}"
else
  DOCKER_ACCESS_HOST="${DOCKER_ACCESS_HOST:-127.0.0.1}"
fi

usage() {
  cat <<'EOF'
Usage:
  ./docker_host_databases.sh [up|status|logs|stop|restart]

Run this on the Docker host, or inside the MLEvolve devcontainer after the
docker-outside-of-docker feature is installed and the container has been rebuilt.
It starts/reuses local Qdrant and Neo4j containers with persistent Docker
volumes and published ports.

Environment overrides:
  LOCALML_SCHEDULER_NEO4J_PASSWORD  Neo4j password. Default: test12345
  QDRANT_HTTP_PORT                   Host Qdrant HTTP port. Default: 6333
  NEO4J_BOLT_PORT                    Host Neo4j Bolt port. Default: 7687
  COMPOSE_PROJECT_NAME               Compose project name. Default: mlevolve
  COMPOSE_FILE                       Compose file. Default: ./docker-compose.local.yml
  DOCKER_ACCESS_HOST                 Hostname reachable from this shell.
  QDRANT_CONTAINER                   Container name. Default: mlevolve-qdrant
  NEO4J_CONTAINER                    Container name. Default: mlevolve-neo4j
EOF
}

require_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    cat >&2 <<'EOF'
Docker is not available in this shell.

Rebuild the devcontainer after enabling docker-outside-of-docker, or open a
terminal on your Docker host and run this script there. After it starts the
services, use the printed QDRANT_URL and GRAPH_DB_URI values with ./bootstrap.sh.
EOF
    exit 127
  fi
}

use_compose() {
  [[ -f "$COMPOSE_FILE" ]] && docker compose version >/dev/null 2>&1
}

compose_cmd() {
  docker compose -f "$COMPOSE_FILE" -p "$COMPOSE_PROJECT_NAME" "$@"
}

container_exists() {
  docker container inspect "$1" >/dev/null 2>&1
}

container_running() {
  [[ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null || true)" == "true" ]]
}

start_qdrant() {
  if container_exists "$QDRANT_CONTAINER"; then
    docker start "$QDRANT_CONTAINER" >/dev/null
    return 0
  fi

  docker run -d \
    --name "$QDRANT_CONTAINER" \
    --restart unless-stopped \
    -p "${QDRANT_HTTP_PORT}:6333" \
    -p "${QDRANT_GRPC_PORT}:6334" \
    -v "${QDRANT_VOLUME}:/qdrant/storage" \
    "$QDRANT_IMAGE" >/dev/null
}

start_neo4j() {
  if container_exists "$NEO4J_CONTAINER"; then
    docker start "$NEO4J_CONTAINER" >/dev/null
    return 0
  fi

  docker run -d \
    --name "$NEO4J_CONTAINER" \
    --restart unless-stopped \
    -p "${NEO4J_HTTP_PORT}:7474" \
    -p "${NEO4J_BOLT_PORT}:7687" \
    -e "NEO4J_AUTH=neo4j/${LOCALML_SCHEDULER_NEO4J_PASSWORD}" \
    -v "${NEO4J_DATA_VOLUME}:/data" \
    -v "${NEO4J_LOGS_VOLUME}:/logs" \
    "$NEO4J_IMAGE" >/dev/null
}

start_compose_stack() {
  compose_cmd up -d qdrant neo4j
}

wait_for_qdrant() {
  local waited=0
  local max_wait="${QDRANT_WAIT_SECONDS:-60}"
  local url="http://${DOCKER_ACCESS_HOST}:${QDRANT_HTTP_PORT}/collections"

  until curl -fsS "$url" >/dev/null; do
    if (( waited >= max_wait )); then
      echo "Qdrant did not become reachable at $url within ${max_wait}s." >&2
      docker logs --tail=80 "$QDRANT_CONTAINER" >&2 || true
      return 1
    fi
    sleep 1
    waited=$((waited + 1))
  done
}

neo4j_cypher() {
  if use_compose; then
    compose_cmd exec -T neo4j cypher-shell \
      -u neo4j \
      -p "$LOCALML_SCHEDULER_NEO4J_PASSWORD" \
      "RETURN 1;"
  else
    docker exec "$NEO4J_CONTAINER" cypher-shell \
      -u neo4j \
      -p "$LOCALML_SCHEDULER_NEO4J_PASSWORD" \
      "RETURN 1;"
  fi
}

wait_for_neo4j() {
  local waited=0
  local max_wait="${NEO4J_WAIT_SECONDS:-120}"

  until neo4j_cypher >/dev/null 2>&1; do
    if (( waited >= max_wait )); then
      echo "Neo4j did not become reachable on bolt port ${NEO4J_BOLT_PORT} within ${max_wait}s." >&2
      docker logs --tail=120 "$NEO4J_CONTAINER" >&2 || true
      return 1
    fi
    sleep 2
    waited=$((waited + 2))
  done
}

print_status() {
  if use_compose; then
    compose_cmd ps
  else
    docker ps -a \
      --filter "name=^/${QDRANT_CONTAINER}$" \
      --filter "name=^/${NEO4J_CONTAINER}$" \
      --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
  fi
}

print_devcontainer_hint() {
  cat <<EOF

Docker-host databases are ready.

From inside the devcontainer, run:

  export LOCALML_SCHEDULER_NEO4J_PASSWORD='${LOCALML_SCHEDULER_NEO4J_PASSWORD}'
  QDRANT_URL=http://${DOCKER_ACCESS_HOST}:${QDRANT_HTTP_PORT} \\
  GRAPH_DB_URI=bolt://${DOCKER_ACCESS_HOST}:${NEO4J_BOLT_PORT} \\
  ./bootstrap.sh

To only verify DB connectivity first:

  QDRANT_URL=http://${DOCKER_ACCESS_HOST}:${QDRANT_HTTP_PORT} \\
  GRAPH_DB_URI=bolt://${DOCKER_ACCESS_HOST}:${NEO4J_BOLT_PORT} \\
  MLEVOLVE_RUN_COMPARE=0 \\
  ./bootstrap.sh
EOF
}

cmd="${1:-up}"
case "$cmd" in
  up)
    require_docker
    if use_compose; then
      start_compose_stack
    else
      start_qdrant
      start_neo4j
    fi
    wait_for_qdrant
    wait_for_neo4j
    print_status
    print_devcontainer_hint
    ;;
  restart)
    require_docker
    if use_compose; then
      compose_cmd restart qdrant neo4j
    else
      docker restart "$QDRANT_CONTAINER" "$NEO4J_CONTAINER"
    fi
    wait_for_qdrant
    wait_for_neo4j
    print_status
    print_devcontainer_hint
    ;;
  status)
    require_docker
    print_status
    ;;
  logs)
    require_docker
    if use_compose; then
      compose_cmd logs --tail="${LOG_LINES:-120}" qdrant neo4j
    else
      docker logs --tail="${LOG_LINES:-120}" "$QDRANT_CONTAINER" || true
      docker logs --tail="${LOG_LINES:-120}" "$NEO4J_CONTAINER" || true
    fi
    ;;
  stop)
    require_docker
    if use_compose; then
      compose_cmd stop qdrant neo4j
    else
      docker stop "$QDRANT_CONTAINER" "$NEO4J_CONTAINER"
    fi
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: $cmd" >&2
    usage >&2
    exit 2
    ;;
esac

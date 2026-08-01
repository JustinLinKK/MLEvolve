#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-$ROOT/docker-compose.local.yml}"
COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-mlevolve}"
NEO4J_HARDWARE_CONTAINER="${NEO4J_HARDWARE_CONTAINER:-mlevolve-neo4j-hardware}"
NEO4J_IMAGE="${NEO4J_IMAGE:-neo4j:5.26}"
NEO4J_HARDWARE_DATA_VOLUME="${NEO4J_HARDWARE_DATA_VOLUME:-mlevolve_neo4j_hardware_data}"
NEO4J_HARDWARE_LOGS_VOLUME="${NEO4J_HARDWARE_LOGS_VOLUME:-mlevolve_neo4j_hardware_logs}"
NEO4J_HARDWARE_HTTP_PORT="${NEO4J_HARDWARE_HTTP_PORT:-7475}"
NEO4J_HARDWARE_BOLT_PORT="${NEO4J_HARDWARE_BOLT_PORT:-7688}"
HARDWARE_KNOWLEDGE_NEO4J_PASSWORD="${HARDWARE_KNOWLEDGE_NEO4J_PASSWORD:-test12345}"
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
It starts/reuses the independent hardware knowledge Neo4j database.

Environment overrides:
  HARDWARE_KNOWLEDGE_NEO4J_PASSWORD  Hardware Neo4j password. Default: test12345
  NEO4J_HARDWARE_BOLT_PORT                   Host hardware Neo4j Bolt port. Default: 7688
  NEO4J_HARDWARE_HTTP_PORT                   Host hardware Neo4j HTTP port. Default: 7475
  COMPOSE_PROJECT_NAME                       Compose project name. Default: mlevolve
  COMPOSE_FILE                               Compose file. Default: ./docker-compose.local.yml
  DOCKER_ACCESS_HOST                         Hostname reachable from this shell.
  NEO4J_HARDWARE_CONTAINER                   Container name. Default: mlevolve-neo4j-hardware
EOF
}

require_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    cat >&2 <<'EOF'
Docker is not available in this shell.

Rebuild the devcontainer after enabling docker-outside-of-docker, or open a
terminal on your Docker host and run this script there.
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

start_neo4j_hardware() {
  if container_exists "$NEO4J_HARDWARE_CONTAINER"; then
    docker start "$NEO4J_HARDWARE_CONTAINER" >/dev/null
    return 0
  fi

  docker run -d \
    --name "$NEO4J_HARDWARE_CONTAINER" \
    --restart unless-stopped \
    -p "${NEO4J_HARDWARE_HTTP_PORT}:7474" \
    -p "${NEO4J_HARDWARE_BOLT_PORT}:7687" \
    -e "NEO4J_AUTH=neo4j/${HARDWARE_KNOWLEDGE_NEO4J_PASSWORD}" \
    -v "${NEO4J_HARDWARE_DATA_VOLUME}:/data" \
    -v "${NEO4J_HARDWARE_LOGS_VOLUME}:/logs" \
    "$NEO4J_IMAGE" >/dev/null
}

start_compose_stack() {
  compose_cmd up -d neo4j-hardware
}

neo4j_cypher() {
  if use_compose; then
    compose_cmd exec -T neo4j-hardware cypher-shell \
      -u neo4j \
      -p "$HARDWARE_KNOWLEDGE_NEO4J_PASSWORD" \
      "RETURN 1;"
  else
    docker exec "$NEO4J_HARDWARE_CONTAINER" cypher-shell \
      -u neo4j \
      -p "$HARDWARE_KNOWLEDGE_NEO4J_PASSWORD" \
      "RETURN 1;"
  fi
}

wait_for_neo4j() {
  local waited=0
  local max_wait="${NEO4J_WAIT_SECONDS:-120}"

  until neo4j_cypher >/dev/null 2>&1; do
    if (( waited >= max_wait )); then
      echo "Hardware Neo4j did not become reachable on bolt port ${NEO4J_HARDWARE_BOLT_PORT} within ${max_wait}s." >&2
      docker logs --tail=120 "$NEO4J_HARDWARE_CONTAINER" >&2 || true
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
      --filter "name=^/${NEO4J_HARDWARE_CONTAINER}$" \
      --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
  fi
}

print_devcontainer_hint() {
  cat <<EOF

Hardware knowledge database is ready.

From inside the devcontainer, run:

  export HARDWARE_KNOWLEDGE_NEO4J_PASSWORD='${HARDWARE_KNOWLEDGE_NEO4J_PASSWORD}'
  HARDWARE_GRAPH_URI=bolt://${DOCKER_ACCESS_HOST}:${NEO4J_HARDWARE_BOLT_PORT} \\
  ./bootstrap.sh

To only verify DB connectivity first:

  HARDWARE_GRAPH_URI=bolt://${DOCKER_ACCESS_HOST}:${NEO4J_HARDWARE_BOLT_PORT} \\
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
      start_neo4j_hardware
    fi
    wait_for_neo4j
    print_status
    print_devcontainer_hint
    ;;
  restart)
    require_docker
    if use_compose; then
      compose_cmd restart neo4j-hardware
    else
      docker restart "$NEO4J_HARDWARE_CONTAINER"
    fi
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
      compose_cmd logs --tail="${LOG_LINES:-120}" neo4j-hardware
    else
      docker logs --tail="${LOG_LINES:-120}" "$NEO4J_HARDWARE_CONTAINER" || true
    fi
    ;;
  stop)
    require_docker
    if use_compose; then
      compose_cmd stop neo4j-hardware
    else
      docker stop "$NEO4J_HARDWARE_CONTAINER"
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

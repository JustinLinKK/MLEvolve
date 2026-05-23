#!/usr/bin/env bash
set -euo pipefail

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXP_ID="${EXP_ID:?Set EXP_ID to the competition task id}"
DATASET_DIR="${DATASET_DIR:-/datasets/mle-bench-data}"
SERVER_ID="${SERVER_ID:-111}"
TIME_LIMIT_SECS="${TIME_LIMIT_SECS:-43200}"
MEMORY_INDEX="${MEMORY_INDEX:-0}"
START_CPU_ID="${START_CPU_ID:-0}"
CPU_NUMBER="${CPU_NUMBER:-21}"
SCHEDULER_RUNTIME_ROOT="${SCHEDULER_RUNTIME_ROOT:-/runtime/localml_scheduler}"
if [ -z "${MLEVOLVE_CONFIG:-}" ]; then
    if [ -f "$ROOT/config.yaml" ]; then
        MLEVOLVE_CONFIG="$ROOT/config.yaml"
    else
        MLEVOLVE_CONFIG="$ROOT/config.example.yaml"
    fi
fi
GRADING_LOG_DIR="${GRADING_LOG_DIR:-$SCHEDULER_RUNTIME_ROOT/grading_servers}"
RUNS_ROOT="${RUNS_ROOT:-/results/runs}"
EXP_NAME="${EXP_NAME:-$EXP_ID}"

mkdir -p "$GRADING_LOG_DIR" "$SCHEDULER_RUNTIME_ROOT" "$RUNS_ROOT"
cd "$ROOT"

BASE_PORT=5005
GRADING_SERVER_PORT=$(( BASE_PORT + SERVER_ID ))
export GRADING_SERVER_PORT
export DATASET_DIR
export MEMORY_INDEX
export MLEVOLVE_CONFIG

grading_server_pid=""

wait_for_grading_server() {
    local deadline="${1:-30}"
    local waited=0
    while [ "$waited" -lt "$deadline" ]; do
        if "$PYTHON_BIN" - <<PY >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:${GRADING_SERVER_PORT}/health", timeout=1).read()
PY
        then
            return 0
        fi
        sleep 1
        waited=$(( waited + 1 ))
    done
    return 1
}

cleanup() {
    if [ -n "$grading_server_pid" ] && kill -0 "$grading_server_pid" 2>/dev/null; then
        kill -TERM "$grading_server_pid" 2>/dev/null || true
        wait "$grading_server_pid" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

env GRADING_SERVER_PORT="$GRADING_SERVER_PORT" \
    "$PYTHON_BIN" -u -m engine.validation.format_server \
        dataset_dir="$DATASET_DIR" \
        data_dir="none" \
        desc_file="none" \
        > "$GRADING_LOG_DIR/grading_server_${SERVER_ID}.log" 2>&1 &
grading_server_pid="$!"

if ! wait_for_grading_server "${GRADING_SERVER_WAIT_SECONDS:-30}"; then
    echo "Grading server did not become healthy on port ${GRADING_SERVER_PORT}" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="$MEMORY_INDEX" \
timeout --foreground --signal=TERM --kill-after=20s "${TIME_LIMIT_SECS}s" \
    "$PYTHON_BIN" run.py \
        exp_id="$EXP_ID" \
        dataset_dir="$DATASET_DIR" \
        data_dir="$DATASET_DIR/$EXP_ID/prepared/public" \
        desc_file="$DATASET_DIR/$EXP_ID/prepared/public/description.md" \
        exp_name="$EXP_NAME" \
        start_cpu_id="$START_CPU_ID" \
        cpu_number="$CPU_NUMBER" \
        log_dir="$RUNS_ROOT" \
        workspace_dir="$RUNS_ROOT" \
        scheduler.enabled=true \
        scheduler.settings.runtime_root="$SCHEDULER_RUNTIME_ROOT" \
        scheduler.settings.graph_db.enabled=false \
        scheduler.settings.graph_db.mode=off \
        scheduler.settings.graph_db.uri="bolt://neo4j:7687" \
        scheduler.settings.gpu_scheduler.backend_priority="[stream,cuda_process,exclusive]" \
        scheduler.settings.gpu_scheduler.concurrent_backend_allowlist="[stream]" \
        scheduler.settings.gpu_scheduler.submission_defaults.backend_allowlist="[stream,cuda_process]" \
        scheduler.settings.gpu_scheduler.mps.enabled=false \
        scheduler.settings.gpu_scheduler.stream.enabled=true \
        scheduler.settings.hardware_feature_db.url="http://qdrant:6333" \
        scheduler.runtime_root="$SCHEDULER_RUNTIME_ROOT"
RUN_EXIT=$?

if [ "$RUN_EXIT" -eq 130 ]; then
    exit 130
fi
if [ "$RUN_EXIT" -eq 124 ]; then
    echo "Timed out after ${TIME_LIMIT_SECS}s"
elif [ "$RUN_EXIT" -ne 0 ]; then
    echo "Run failed with exit code: $RUN_EXIT"
fi

"$PYTHON_BIN" utils/submission_fusion_utils.py \
    --task_id "$EXP_ID" \
    --exp_name "$EXP_NAME"

exit "$RUN_EXIT"

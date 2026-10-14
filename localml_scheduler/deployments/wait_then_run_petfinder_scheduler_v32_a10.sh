#!/usr/bin/env bash
# Start the strict 9/2-parity Scheduler+Predictor run only after a preserved
# diagnostic controller has exited.  This script never signals that controller.
set -euo pipefail

repo="${REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
launcher="${LAUNCHER:-$repo/localml_scheduler/deployments/run_petfinder_scheduler_v32_a10.sh}"
current_run_id="${CURRENT_RUN_ID:-petfinder_scheduler_v32_a10_20260915_094241}"
poll_seconds="${POLL_SECONDS:-60}"
settle_seconds="${SETTLE_SECONDS:-30}"
lock_dir="${WATCH_LOCK_DIR:-/tmp/mlevolve_petfinder_scheduler_v32_successor.lock}"
baseline_data_dir="${PETFINDER_BASELINE_DATA_DIR:-/root/downeyflyfan/mle-bench-data/petfinder-pawpularity-score/prepared/public}"
predictor_source_root="${PREDICTOR_V32_SOURCE_ROOT:-$repo/predictor_v3.2/source}"
predictor_weights="${PREDICTOR_V32_WEIGHTS:-$repo/predictor_v3.2/weights/predictor_v3.2_student.pt}"

test -x "$launcher"
test -r "$baseline_data_dir/train.csv"
test -r "$baseline_data_dir/test.csv"
test -r "$baseline_data_dir/sample_submission.csv"
[[ "$poll_seconds" =~ ^[1-9][0-9]*$ ]]
[[ "$settle_seconds" =~ ^[0-9]+$ ]]

if ! mkdir "$lock_dir" 2>/dev/null; then
    echo "successor watcher already exists: $lock_dir"
    exit 0
fi
trap 'rmdir "$lock_dir" 2>/dev/null || true' EXIT

while true; do
    while pgrep -f "run.py exp_name=${current_run_id}" >/dev/null; do
        echo "$(date -u +%FT%TZ) waiting for ${current_run_id}"
        sleep "$poll_seconds"
    done
    sleep "$settle_seconds"
    if ! pgrep -f "run.py exp_name=${current_run_id}" >/dev/null; then
        break
    fi
    echo "controller reappeared during settle period; continuing to wait"
done

echo "$(date -u +%FT%TZ) starting strict 9/2-parity Scheduler+Predictor run"
rmdir "$lock_dir"
trap - EXIT
exec env \
    REPO="$repo" \
    MLEVOLVE_EXECUTION_REPO="$repo" \
    PETFINDER_DATA_DIR="$baseline_data_dir" \
    PETFINDER_DESC_FILE="$baseline_data_dir/description.md" \
    PREDICTOR_V32_SOURCE_ROOT="$predictor_source_root" \
    PREDICTOR_V32_WEIGHTS="$predictor_weights" \
    PETFINDER_NODE_BUDGET=50 \
    PETFINDER_SEARCH_TIME_LIMIT=28800 \
    PREFLIGHT_ENABLED=false \
    QWEN_VLLM_URL="${QWEN_VLLM_URL:-http://mlevolve-qwen-a100.ecepxie.svc.cluster.local:8000/v1}" \
    "$launcher"

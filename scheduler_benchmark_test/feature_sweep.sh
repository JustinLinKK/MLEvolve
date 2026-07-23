#!/usr/bin/env bash
# Focused feature sweep for RAM preload, packed-width, and packed batch-threshold settings.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="$REPO/scheduler_benchmark_test"
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY="python3"
source "$BENCH_DIR/benchmark_runtime.sh"
bench_init_runtime "feature_sweep" "$BENCH_DIR"

FEATURE_DIR="${FEATURE_DIR:-/tmp/scheduler_benchmark_feature_sweep}"
TRACE="${TRACE:-$FEATURE_DIR/smoke_trace.jsonl}"
CODE_CACHE="$FEATURE_DIR/codes"
RESULTS_BASE="${RESULTS_BASE:-$REPO/results/scheduler_benchmark_test/feature_sweep}"
mkdir -p "$FEATURE_DIR" "$CODE_CACHE" "$RESULTS_BASE"

if ! "$PY" "$BENCH_DIR/check_benchmark_env.py" --trace "$TRACE" >/dev/null; then
    "$PY" "$BENCH_DIR/check_benchmark_env.py" --trace "$TRACE"
    exit 1
fi

SMOKE_DIR="$FEATURE_DIR" "$PY" "$BENCH_DIR/gen_smoke_trace.py"

run_case() {
    local id="$1"
    shift
    local cfg_dir="$RESULTS_BASE/$id"
    mkdir -p "$cfg_dir"
    bench_prepare_case "$id"
    echo "[$(date -Iseconds)] === feature case $id ==="
    bench_start_gpu_sampler "$cfg_dir/gpu_metrics.csv"
    if bench_run_logged "$cfg_dir/replay.log" \
        env REPLAY_WORKDIR_ROOT="$BENCH_CASE_WORKDIR_ROOT" \
        timeout 420 "$PY" "$BENCH_DIR/replay_scheduler.py" \
            --config-id "$id" \
            --trace "$TRACE" \
            --runtime-root "$BENCH_CASE_RUNTIME_ROOT" \
            --results-dir "$cfg_dir/results" \
            --summary "$cfg_dir/summary.json" \
            --code-cache-dir "$CODE_CACHE" \
            --duration-s 360 \
            --vram-budget-gib 28.0 \
            "$@"; then
        local rc=0
    else
        local rc=$?
    fi
    if [ -n "${BENCH_GPU_SAMPLER_PID:-}" ]; then
        bench_stop_pid "$BENCH_GPU_SAMPLER_PID"
        unset BENCH_GPU_SAMPLER_PID
    fi
    local done_count
    done_count=$(grep -o '"COMPLETED": *[0-9]*' "$cfg_dir/summary.json" 2>/dev/null | head -1 || echo "no_summary")
    echo "[$(date -Iseconds)] $id rc=$rc $done_count"
}

run_case PRELOAD_OFF \
    --mode serial_basic --backend exclusive --batch-search off \
    --cache-warm-top-k 0 --cache-entry-capacity 0 --cache-memory-budget-gib 0
run_case PRELOAD_ON \
    --mode serial_basic --backend exclusive --batch-search off \
    --cache-warm-top-k 4 --cache-entry-capacity 4 --cache-max-ram-percent 0.20 --cache-memory-budget-gib 6
run_case PAR_CACHE_OFF \
    --mode parallel_batch_optimized --backend cuda_process --batch-search power_of_two \
    --cache-warm-top-k 0 --cache-entry-capacity 0 --cache-max-ram-percent 0.0 --cache-memory-budget-gib 0 \
    --power-of-two-range-up 1 --power-of-two-range-down 1
run_case PAR_CACHE_ON \
    --mode parallel_batch_optimized --backend cuda_process --batch-search power_of_two \
    --cache-warm-top-k 4 --cache-entry-capacity 4 --cache-max-ram-percent 0.20 --cache-memory-budget-gib 6 \
    --power-of-two-range-up 1 --power-of-two-range-down 1
run_case PACK3_ON \
    --mode parallel_batch_optimized --backend cuda_process --batch-search power_of_two \
    --cache-warm-top-k 4 --cache-entry-capacity 4 --cache-max-ram-percent 0.20 --cache-memory-budget-gib 6 \
    --power-of-two-range-up 1 --power-of-two-range-down 1 \
    --max-packed-jobs-per-gpu 3
run_case PACK4_ON \
    --mode parallel_batch_optimized --backend cuda_process --batch-search power_of_two \
    --cache-warm-top-k 4 --cache-entry-capacity 4 --cache-max-ram-percent 0.20 --cache-memory-budget-gib 6 \
    --power-of-two-range-up 1 --power-of-two-range-down 1 \
    --max-packed-jobs-per-gpu 4
run_case BINARY_TIGHT \
    --mode parallel_batch_optimized --backend cuda_process --batch-search binary \
    --cache-warm-top-k 4 --cache-entry-capacity 4 --cache-max-ram-percent 0.20 --cache-memory-budget-gib 6 \
    --binary-range-up 8 --binary-range-down 4
run_case BINARY_WIDE \
    --mode parallel_batch_optimized --backend cuda_process --batch-search binary \
    --cache-warm-top-k 4 --cache-entry-capacity 4 --cache-max-ram-percent 0.20 --cache-memory-budget-gib 6 \
    --binary-range-up 16 --binary-range-down 8
run_case POW2_LOCAL \
    --mode parallel_batch_optimized --backend cuda_process --batch-search power_of_two \
    --cache-warm-top-k 4 --cache-entry-capacity 4 --cache-max-ram-percent 0.20 --cache-memory-budget-gib 6 \
    --power-of-two-range-up 1 --power-of-two-range-down 1
run_case POW2_WIDE \
    --mode parallel_batch_optimized --backend cuda_process --batch-search power_of_two \
    --cache-warm-top-k 4 --cache-entry-capacity 4 --cache-max-ram-percent 0.20 --cache-memory-budget-gib 6 \
    --power-of-two-range-up 2 --power-of-two-range-down 1

echo
echo "=== FEATURE SWEEP SUMMARY ==="
for id in PRELOAD_OFF PRELOAD_ON PAR_CACHE_OFF PAR_CACHE_ON PACK3_ON PACK4_ON BINARY_TIGHT BINARY_WIDE POW2_LOCAL POW2_WIDE; do
    cfg_dir="$RESULTS_BASE/$id"
    if [ -f "$cfg_dir/summary.json" ]; then
        summary_line=$("$PY" - "$cfg_dir/summary.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
hits = ((payload.get("cache_stats") or {}).get("hits"))
completed = ((payload.get("by_status") or {}).get("COMPLETED"))
group_sizes = payload.get("packed_group_size_counts") or {}
print(f"COMPLETED: {completed} hits: {hits} packed_group_size_counts: {group_sizes}")
PY
)
    else
        summary_line="COMPLETED: ? hits: ? packed_group_size_counts: ?"
    fi
    echo "  $id: $summary_line"
done
echo "Feature sweep results in $RESULTS_BASE"

#!/usr/bin/env bash
# Focused runner for cuda_process pack-3 and pack-4 scheduler cases.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="$REPO/scheduler_benchmark_test"
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY="python3"
source "$BENCH_DIR/benchmark_runtime.sh"
bench_init_runtime "cuda_process_pack34" "$BENCH_DIR"

TRACE="${TRACE:-$BENCH_DIR/workload_trace_W3.jsonl}"
CODE_CACHE="${CODE_CACHE:-$BENCH_DIR/replay_codes_W3}"
RESULTS_BASE="${RESULTS_BASE:-$REPO/results/scheduler_benchmark_test/cuda_process_pack34}"
mkdir -p "$RESULTS_BASE"

CONFIG_TIMEOUT="${CONFIG_TIMEOUT:-2700}"
COMMON_VRAM_BUDGET_GIB="${COMMON_VRAM_BUDGET_GIB:-30.0}"
COMMON_CACHE_WARM_POLICY="${COMMON_CACHE_WARM_POLICY:-top_k}"
COMMON_CACHE_WARM_TOP_K="${COMMON_CACHE_WARM_TOP_K:-4}"
COMMON_CACHE_ENTRY_CAPACITY="${COMMON_CACHE_ENTRY_CAPACITY:-8}"
COMMON_CACHE_MAX_RAM_PERCENT="${COMMON_CACHE_MAX_RAM_PERCENT:-0.20}"
COMMON_CACHE_MEMORY_BUDGET_GIB="${COMMON_CACHE_MEMORY_BUDGET_GIB:-30.0}"
COMMON_POWER_OF_TWO_RANGE_UP="${COMMON_POWER_OF_TWO_RANGE_UP:-4}"
COMMON_POWER_OF_TWO_RANGE_DOWN="${COMMON_POWER_OF_TWO_RANGE_DOWN:-1}"

if ! "$PY" "$BENCH_DIR/check_benchmark_env.py" --trace "$TRACE" >/dev/null; then
    "$PY" "$BENCH_DIR/check_benchmark_env.py" --trace "$TRACE"
    exit 1
fi

if [ ! -f "$TRACE" ]; then
    echo "[setup] generating W3 trace..."
    OUT_DIR="$BENCH_DIR" "$PY" "$BENCH_DIR/gen_trace_W3.py"
fi

run_sched() {
    local id="$1"
    local max_packed="$2"
    local cfg_dir="$RESULTS_BASE/$id"
    mkdir -p "$cfg_dir"
    bench_prepare_case "$id"

    echo "[$(date -Iseconds)] === $id  backend=cuda_process  max_packed=$max_packed ==="
    bench_start_gpu_sampler "$cfg_dir/gpu_metrics.csv"

    local t0
    t0=$(date +%s.%N)
    if bench_run_logged "$cfg_dir/replay.log" \
        env REPLAY_WORKDIR_ROOT="$BENCH_CASE_WORKDIR_ROOT" \
        timeout "$CONFIG_TIMEOUT" "$PY" "$BENCH_DIR/replay_scheduler.py" \
            --config-id "$id" \
            --mode parallel_batch_optimized \
            --backend cuda_process \
            --batch-search power_of_two \
            --trace "$TRACE" \
            --runtime-root "$BENCH_CASE_RUNTIME_ROOT" \
            --results-dir "$cfg_dir/results" \
            --summary "$cfg_dir/summary.json" \
            --code-cache-dir "$CODE_CACHE" \
            --duration-s $(( CONFIG_TIMEOUT - 60 )) \
            --vram-budget-gib "$COMMON_VRAM_BUDGET_GIB" \
            --cache-warm-policy "$COMMON_CACHE_WARM_POLICY" \
            --cache-warm-top-k "$COMMON_CACHE_WARM_TOP_K" \
            --cache-entry-capacity "$COMMON_CACHE_ENTRY_CAPACITY" \
            --cache-max-ram-percent "$COMMON_CACHE_MAX_RAM_PERCENT" \
            --cache-memory-budget-gib "$COMMON_CACHE_MEMORY_BUDGET_GIB" \
            --power-of-two-range-up "$COMMON_POWER_OF_TWO_RANGE_UP" \
            --power-of-two-range-down "$COMMON_POWER_OF_TWO_RANGE_DOWN" \
            --max-packed-jobs-per-gpu "$max_packed"; then
        local rc=0
    else
        local rc=$?
    fi
    local t1
    t1=$(date +%s.%N)
    local elapsed
    elapsed="$(bench_elapsed_seconds "$t0" "$t1")"
    if [ -n "${BENCH_GPU_SAMPLER_PID:-}" ]; then
        bench_stop_pid "$BENCH_GPU_SAMPLER_PID"
        unset BENCH_GPU_SAMPLER_PID
    fi
    echo "$elapsed" > "$cfg_dir/wall_clock.txt"
    echo "$rc" > "$cfg_dir/rc.txt"
    echo "[$(date -Iseconds)] $id rc=$rc elapsed=${elapsed}s"
}

run_sched W8 3
run_sched W9 4

echo "Run plots with:"
echo "  RESULTS_DIR=$RESULTS_BASE $PY $BENCH_DIR/plot_results.py"

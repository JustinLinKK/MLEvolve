#!/usr/bin/env bash
# Focused runner for cuda_process pack-3 and pack-4 scheduler cases.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="$REPO/scheduler_benchmark_test"
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

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

cleanup_gpu() {
    pkill -9 -f "replay_scheduler.py|replay_torch_mp.py|step_[0-9]\\+\\.py|nvidia-cuda-mps" 2>/dev/null || true
    sleep 3
}

run_sched() {
    local id="$1"
    local max_packed="$2"
    local cfg_dir="$RESULTS_BASE/$id"
    mkdir -p "$cfg_dir"
    cleanup_gpu
    rm -rf "/tmp/replay_workdirs/$id"

    echo "[$(date -Iseconds)] === $id  backend=cuda_process  max_packed=$max_packed ==="
    nohup nvidia-smi dmon -s pucvmet -d 1 -o T > "$cfg_dir/dmon.csv" 2>&1 &
    local dmon_pid=$!
    sleep 2

    local t0
    t0=$(date +%s.%N)
    timeout "$CONFIG_TIMEOUT" "$PY" "$BENCH_DIR/replay_scheduler.py" \
        --config-id "$id" \
        --mode parallel_batch_optimized \
        --backend cuda_process \
        --batch-search power_of_two \
        --trace "$TRACE" \
        --runtime-root "/tmp/scheduler_benchmark_runtime_$id" \
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
        --max-packed-jobs-per-gpu "$max_packed" \
        > "$cfg_dir/replay.log" 2>&1
    local rc=$?
    local t1
    t1=$(date +%s.%N)
    local elapsed
    elapsed=$("$PY" - <<PY
t0 = float("$t0")
t1 = float("$t1")
print(t1 - t0)
PY
)
    kill "$dmon_pid" 2>/dev/null || true
    wait "$dmon_pid" 2>/dev/null || true
    sleep 1
    echo "$elapsed" > "$cfg_dir/wall_clock.txt"
    echo "$rc" > "$cfg_dir/rc.txt"
    echo "[$(date -Iseconds)] $id rc=$rc elapsed=${elapsed}s"
}

run_sched W8 3
run_sched W9 4

echo "Run plots with:"
echo "  RESULTS_DIR=$RESULTS_BASE $PY $BENCH_DIR/plot_results.py"

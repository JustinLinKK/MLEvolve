#!/usr/bin/env bash
# Sweep the benchmark trace.
# Set NO_MPS=1 to use the no-MPS matrix, or use sweep_run_windows.sh.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="$REPO/scheduler_benchmark_test"
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY="python3"
source "$BENCH_DIR/benchmark_runtime.sh"
bench_init_runtime "sweep_run" "$BENCH_DIR"
NO_MPS="${NO_MPS:-0}"
HAS_MPS_CONTROL=0
if bench_has_mps_control; then
    HAS_MPS_CONTROL=1
fi

TRACE="$BENCH_DIR/workload_trace_W3.jsonl"
CODE_CACHE="$BENCH_DIR/replay_codes_W3"
if ! bench_mps_enabled; then
    RESULTS_BASE="${RESULTS_BASE:-$REPO/results/scheduler_benchmark_test/main_sweep_windows}"
else
    RESULTS_BASE="${RESULTS_BASE:-$REPO/results/scheduler_benchmark_test/main_sweep}"
fi
mkdir -p "$RESULTS_BASE"

CONFIG_TIMEOUT="${CONFIG_TIMEOUT:-2700}"
COMMON_VRAM_BUDGET_GIB="${COMMON_VRAM_BUDGET_GIB:-30.0}"
COMMON_CACHE_WARM_POLICY="${COMMON_CACHE_WARM_POLICY:-top_k}"
COMMON_CACHE_WARM_TOP_K="${COMMON_CACHE_WARM_TOP_K:-4}"
COMMON_CACHE_ENTRY_CAPACITY="${COMMON_CACHE_ENTRY_CAPACITY:-8}"
COMMON_CACHE_MAX_RAM_PERCENT="${COMMON_CACHE_MAX_RAM_PERCENT:-0.20}"
COMMON_CACHE_MEMORY_BUDGET_GIB="${COMMON_CACHE_MEMORY_BUDGET_GIB:-30.0}"
COMMON_BINARY_RANGE_UP="${COMMON_BINARY_RANGE_UP:-16}"
COMMON_BINARY_RANGE_DOWN="${COMMON_BINARY_RANGE_DOWN:-8}"
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
    local id="$1" mode="$2" backend="$3" probe="$4"
    shift 4
    local cfg_dir="$RESULTS_BASE/$id"
    mkdir -p "$cfg_dir"
    bench_prepare_case "$id"

    echo "[$(date -Iseconds)] === $id  mode=$mode  backend=$backend  probe=$probe ==="
    bench_start_gpu_sampler "$cfg_dir/gpu_metrics.csv"

    local t0
    t0=$(date +%s.%N)
    if bench_run_logged "$cfg_dir/replay.log" \
        env REPLAY_WORKDIR_ROOT="$BENCH_CASE_WORKDIR_ROOT" \
        timeout "$CONFIG_TIMEOUT" "$PY" "$BENCH_DIR/replay_scheduler.py" \
            --config-id "$id" \
            --mode "$mode" \
            --backend "$backend" \
            --batch-search "$probe" \
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
            --binary-range-up "$COMMON_BINARY_RANGE_UP" \
            --binary-range-down "$COMMON_BINARY_RANGE_DOWN" \
            --power-of-two-range-up "$COMMON_POWER_OF_TWO_RANGE_UP" \
            --power-of-two-range-down "$COMMON_POWER_OF_TWO_RANGE_DOWN" \
            "$@"; then
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

run_torchmp() {
    local id="$1" backend="$2" probe="$3"
    local cfg_dir="$RESULTS_BASE/$id"
    mkdir -p "$cfg_dir"
    bench_prepare_case "$id"

    echo "[$(date -Iseconds)] === $id  torch_mp  backend=$backend  probe=$probe ==="
    bench_start_gpu_sampler "$cfg_dir/gpu_metrics.csv"

    local t0
    t0=$(date +%s.%N)
    if bench_run_logged "$cfg_dir/replay.log" \
        env REPLAY_WORKDIR_ROOT="$BENCH_CASE_WORKDIR_ROOT" \
        timeout "$CONFIG_TIMEOUT" "$PY" "$BENCH_DIR/replay_torch_mp.py" \
            --config-id "$id" \
            --backend "$backend" \
            --batch-search "$probe" \
            --n-workers 2 \
            --trace "$TRACE" \
            --results-dir "$cfg_dir/results" \
            --summary "$cfg_dir/summary.json" \
            --code-cache-dir "$CODE_CACHE" \
            --duration-s $(( CONFIG_TIMEOUT - 60 )); then
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

if bench_mps_enabled && [ "$HAS_MPS_CONTROL" -ne 1 ]; then
    echo "MPS is not available in this environment. Use NO_MPS=1 or run scheduler_benchmark_test/sweep_run_windows.sh." >&2
    exit 1
fi

summary_ids=()

run_sched   B1  serial_basic             exclusive    off
summary_ids+=(B1)

if ! bench_mps_enabled; then
    run_sched   B3  serial_batch_optimized   exclusive    power_of_two
    run_sched   W1  parallel_default         cuda_process off
    run_sched   T2  parallel_default         stream       off
    run_sched   W4  parallel_batch_optimized stream       power_of_two \
        --cache-warm-top-k 0 --cache-entry-capacity 0 --cache-max-ram-percent 0.0 --cache-memory-budget-gib 0
    run_sched   W5  parallel_batch_optimized stream       power_of_two
    run_sched   W6  parallel_batch_optimized stream       power_of_two \
        --max-packed-jobs-per-gpu 3
    run_sched   W7  parallel_batch_optimized stream       power_of_two \
        --max-packed-jobs-per-gpu 4
    run_sched   W8  parallel_batch_optimized cuda_process power_of_two \
        --max-packed-jobs-per-gpu 3
    run_sched   W9  parallel_batch_optimized cuda_process power_of_two \
        --max-packed-jobs-per-gpu 4
    run_torchmp T11 stream power_of_two
    summary_ids+=(B3 W1 T2 W4 W5 W6 W7 W8 W9 T11)
else
    run_sched   B2  serial_batch_optimized   exclusive    binary
    run_sched   B3  serial_batch_optimized   exclusive    power_of_two
    run_sched   T1  parallel_default         mps       off
    run_sched   T2  parallel_default         stream    off
    run_sched   T4  parallel_batch_optimized mps       binary
    run_sched   T5  parallel_batch_optimized mps       power_of_two
    run_sched   T6  parallel_batch_optimized stream    binary
    run_sched   T7  parallel_batch_optimized stream    power_of_two
    run_torchmp T8  mps    binary
    run_torchmp T9  mps    power_of_two
    run_torchmp T10 stream binary
    run_torchmp T11 stream power_of_two
    summary_ids+=(B2 B3 T1 T2 T4 T5 T6 T7 T8 T9 T10 T11)
fi

echo "[$(date -Iseconds)] === SWEEP COMPLETE ==="
echo "=== per-config summary ==="
for id in "${summary_ids[@]}"; do
    cfg_dir="$RESULTS_BASE/$id"
    rc=$(cat "$cfg_dir/rc.txt" 2>/dev/null || echo "?")
    wc=$(cat "$cfg_dir/wall_clock.txt" 2>/dev/null || echo "?")
    n=$(grep -o '"COMPLETED": *[0-9]*' "$cfg_dir/summary.json" 2>/dev/null | head -1 || echo "?")
    echo "  $id: rc=$rc wall=${wc}s $n"
done
echo
echo "Run plots: $PY $BENCH_DIR/plot_results.py"

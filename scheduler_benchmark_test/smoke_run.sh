#!/usr/bin/env bash
# Smoke test for the benchmark harness.
# Set NO_MPS=1 to skip MPS-only configs, or use smoke_run_windows.sh.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="$REPO/scheduler_benchmark_test"
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY="python3"
NO_MPS="${NO_MPS:-0}"
HAS_MPS_CONTROL=0
if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    HAS_MPS_CONTROL=1
fi

SMOKE_DIR="${SMOKE_DIR:-/tmp/scheduler_benchmark_smoke}"
TRACE="$SMOKE_DIR/smoke_trace.jsonl"
CODE_CACHE="$SMOKE_DIR/codes"
RESULTS="$SMOKE_DIR/results"
mkdir -p "$SMOKE_DIR" "$CODE_CACHE" "$RESULTS"

COMMON_VRAM_BUDGET_GIB="${COMMON_VRAM_BUDGET_GIB:-28.0}"
COMMON_CACHE_WARM_POLICY="${COMMON_CACHE_WARM_POLICY:-top_k}"
COMMON_CACHE_MEMORY_BUDGET_GIB="${COMMON_CACHE_MEMORY_BUDGET_GIB:-6.0}"
COMMON_CACHE_MAX_RAM_PERCENT="${COMMON_CACHE_MAX_RAM_PERCENT:-0.20}"

if ! "$PY" "$BENCH_DIR/check_benchmark_env.py" --trace "$TRACE" >/dev/null; then
    "$PY" "$BENCH_DIR/check_benchmark_env.py" --trace "$TRACE"
    exit 1
fi

SMOKE_DIR="$SMOKE_DIR" "$PY" "$BENCH_DIR/gen_smoke_trace.py"

cleanup_gpu() {
    pkill -9 -f "replay_scheduler.py|replay_torch_mp.py|step_[0-9]\+\.py|nvidia-cuda-mps" 2>/dev/null || true
    if [ "$HAS_MPS_CONTROL" -eq 1 ]; then
        echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    fi
    sleep 2
}

run_sched_smoke() {
    local id="$1" mode="$2" backend="$3" probe="$4"
    shift 4
    local cfg_dir="$RESULTS/$id"
    mkdir -p "$cfg_dir"
    cleanup_gpu
    rm -rf "/tmp/replay_workdirs/${id}_smoke"
    echo "--- smoke $id  $mode  $backend  $probe ---"
    timeout 240 "$PY" "$BENCH_DIR/replay_scheduler.py" \
        --config-id "${id}_smoke" \
        --mode "$mode" \
        --backend "$backend" \
        --batch-search "$probe" \
        --trace "$TRACE" \
        --runtime-root "/tmp/scheduler_benchmark_runtime_${id}_smoke" \
        --results-dir "$cfg_dir/results" \
        --summary "$cfg_dir/summary.json" \
        --code-cache-dir "$CODE_CACHE" \
        --vram-budget-gib "$COMMON_VRAM_BUDGET_GIB" \
        --cache-warm-policy "$COMMON_CACHE_WARM_POLICY" \
        --cache-memory-budget-gib "$COMMON_CACHE_MEMORY_BUDGET_GIB" \
        --cache-max-ram-percent "$COMMON_CACHE_MAX_RAM_PERCENT" \
        --duration-s 220 \
        "$@" > "$cfg_dir/replay.log" 2>&1
    local rc=$?
    local n
    n=$(grep -o '"COMPLETED": *[0-9]*' "$cfg_dir/summary.json" 2>/dev/null | head -1)
    echo "  rc=$rc  $n"
}

run_tmp_smoke() {
    local id="$1" backend="$2" probe="$3"
    local cfg_dir="$RESULTS/$id"
    mkdir -p "$cfg_dir"
    cleanup_gpu
    rm -rf "/tmp/replay_workdirs/${id}_smoke"
    echo "--- smoke $id  torch_mp  $backend  $probe ---"
    timeout 240 "$PY" "$BENCH_DIR/replay_torch_mp.py" \
        --config-id "${id}_smoke" \
        --backend "$backend" \
        --batch-search "$probe" \
        --n-workers 2 \
        --trace "$TRACE" \
        --results-dir "$cfg_dir/results" \
        --summary "$cfg_dir/summary.json" \
        --code-cache-dir "$CODE_CACHE" \
        --duration-s 220 \
        > "$cfg_dir/replay.log" 2>&1
    local rc=$?
    local n
    n=$(grep -o '"COMPLETED": *[0-9]*' "$cfg_dir/summary.json" 2>/dev/null | head -1)
    echo "  rc=$rc  $n"
}

if [ "$NO_MPS" != "1" ] && [ "$HAS_MPS_CONTROL" -ne 1 ]; then
    echo "MPS is not available in this environment. Use NO_MPS=1 or run scheduler_benchmark_test/smoke_run_windows.sh." >&2
    exit 1
fi

run_sched_smoke B1 serial_basic     exclusive    off
run_sched_smoke T2 parallel_default stream       off
run_sched_smoke C1 serial_basic     exclusive    off \
    --cache-warm-top-k 0 --cache-entry-capacity 0

summary_ids=(B1 T2 C1)

if [ "$NO_MPS" = "1" ]; then
    run_sched_smoke W4 parallel_batch_optimized stream power_of_two \
        --cache-warm-top-k 0 --cache-entry-capacity 0 --cache-max-ram-percent 0.0 --cache-memory-budget-gib 0
    run_sched_smoke W5 parallel_batch_optimized stream power_of_two \
        --binary-range-up 16 --binary-range-down 8 --cache-warm-top-k 4 --cache-entry-capacity 4
    run_sched_smoke W6 parallel_batch_optimized stream power_of_two \
        --cache-warm-top-k 4 --cache-entry-capacity 4 --max-packed-jobs-per-gpu 3
    run_sched_smoke W7 parallel_batch_optimized stream power_of_two \
        --cache-warm-top-k 4 --cache-entry-capacity 4 --max-packed-jobs-per-gpu 4
    run_sched_smoke W8 parallel_batch_optimized cuda_process power_of_two \
        --cache-warm-top-k 4 --cache-entry-capacity 4 --max-packed-jobs-per-gpu 3
    run_sched_smoke W9 parallel_batch_optimized cuda_process power_of_two \
        --cache-warm-top-k 4 --cache-entry-capacity 4 --max-packed-jobs-per-gpu 4
    run_tmp_smoke   T11 stream power_of_two
    summary_ids+=(W4 W5 W6 W7 W8 W9 T11)
else
    run_sched_smoke T4 parallel_batch_optimized mps binary \
        --binary-range-up 16 --binary-range-down 8 --cache-warm-top-k 4 --cache-entry-capacity 4
    run_tmp_smoke   T8 mps binary
    summary_ids+=(T4 T8)
fi

echo
echo "=== SMOKE SUMMARY ==="
for id in "${summary_ids[@]}"; do
    cfg_dir="$RESULTS/$id"
    n=$(grep -o '"COMPLETED": *[0-9]*' "$cfg_dir/summary.json" 2>/dev/null | head -1 || echo "no_summary")
    echo "  $id: $n"
done
echo "Smoke results in $RESULTS"

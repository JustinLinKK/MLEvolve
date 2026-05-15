#!/usr/bin/env bash
set -euo pipefail

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BENCH_DIR="$ROOT/scheduler_benchmark_test"
PYTHON_BIN="${PYTHON_BIN:-python}"

source "$BENCH_DIR/benchmark_runtime.sh"
bench_init_runtime "nautilus_benchmark" "$BENCH_DIR"

BENCH_CONFIG_ID="${BENCH_CONFIG_ID:-nautilus_benchmark}"
BENCH_RUNNER="${BENCH_RUNNER:-scheduler}"
BENCH_MODE="${BENCH_MODE:-parallel_default}"
BENCH_BACKEND="${BENCH_BACKEND:-stream}"
BENCH_BATCH_SEARCH="${BENCH_BATCH_SEARCH:-off}"
BENCH_TIMEOUT_S="${BENCH_TIMEOUT_S:-2700}"
BENCH_DURATION_S="${BENCH_DURATION_S:-$(( BENCH_TIMEOUT_S - 60 ))}"
BENCH_RESULTS_ROOT="${BENCH_RESULTS_ROOT:-/results/scheduler_benchmark_test}"
BENCH_CODE_CACHE_DIR="${BENCH_CODE_CACHE_DIR:-$BENCH_RUN_ROOT/code_cache}"
BENCH_TRACE_KIND="${BENCH_TRACE_KIND:-w3}"
CASSAVA_ROOT="${CASSAVA_ROOT:-/datasets/cassava-leaf-disease-classification/prepared/public}"

mkdir -p "$BENCH_RESULTS_ROOT" "$BENCH_CODE_CACHE_DIR"
export REPO_ROOT="$ROOT"
export CASSAVA_ROOT

case "$BENCH_TRACE_KIND" in
    smoke)
        TRACE_DIR="${BENCH_TRACE_DIR:-$BENCH_RUN_ROOT/traces/smoke}"
        mkdir -p "$TRACE_DIR"
        TRACE_PATH="${BENCH_TRACE_PATH:-$TRACE_DIR/smoke_trace.jsonl}"
        if [ ! -f "$TRACE_PATH" ]; then
            SMOKE_DIR="$TRACE_DIR" "$PYTHON_BIN" "$BENCH_DIR/gen_smoke_trace.py"
        fi
        ;;
    w3)
        TRACE_DIR="${BENCH_TRACE_DIR:-$BENCH_RUN_ROOT/traces/w3}"
        mkdir -p "$TRACE_DIR"
        TRACE_PATH="${BENCH_TRACE_PATH:-$TRACE_DIR/workload_trace_W3.jsonl}"
        if [ ! -f "$TRACE_PATH" ]; then
            OUT_DIR="$TRACE_DIR" "$PYTHON_BIN" "$BENCH_DIR/gen_trace_W3.py"
        fi
        ;;
    *)
        TRACE_PATH="${BENCH_TRACE_PATH:?Set BENCH_TRACE_PATH when BENCH_TRACE_KIND is custom}"
        ;;
esac

"$PYTHON_BIN" "$BENCH_DIR/check_benchmark_env.py" --trace "$TRACE_PATH"

if [ "$BENCH_BACKEND" = "mps" ] && ! bench_mps_enabled; then
    echo "BENCH_BACKEND=mps requires BENCH_ENABLE_MPS=true" >&2
    exit 1
fi
if [ "$BENCH_BACKEND" = "mps" ] && ! bench_has_mps_control; then
    echo "nvidia-cuda-mps-control is unavailable in this image; cannot run an MPS benchmark job" >&2
    exit 1
fi

RESULT_DIR="$BENCH_RESULTS_ROOT/$BENCH_CONFIG_ID"
mkdir -p "$RESULT_DIR"
bench_prepare_case "$BENCH_CONFIG_ID"
bench_start_gpu_sampler "$RESULT_DIR/gpu_metrics.csv"

LOG_PATH="$RESULT_DIR/replay.log"
SUMMARY_PATH="$RESULT_DIR/summary.json"
T0="$(date +%s.%N)"

if [ "$BENCH_RUNNER" = "torch_mp" ]; then
    if bench_run_logged "$LOG_PATH" \
        env REPLAY_WORKDIR_ROOT="$BENCH_CASE_WORKDIR_ROOT" \
        timeout --foreground --signal=TERM --kill-after=20s "${BENCH_TIMEOUT_S}s" \
            "$PYTHON_BIN" "$BENCH_DIR/replay_torch_mp.py" \
            --config-id "$BENCH_CONFIG_ID" \
            --backend "$BENCH_BACKEND" \
            --batch-search "$BENCH_BATCH_SEARCH" \
            --n-workers "${BENCH_N_WORKERS:-2}" \
            --trace "$TRACE_PATH" \
            --results-dir "$RESULT_DIR/results" \
            --summary "$SUMMARY_PATH" \
            --code-cache-dir "$BENCH_CODE_CACHE_DIR" \
            --duration-s "$BENCH_DURATION_S"; then
        RC=0
    else
        RC=$?
    fi
else
    if bench_run_logged "$LOG_PATH" \
        env REPLAY_WORKDIR_ROOT="$BENCH_CASE_WORKDIR_ROOT" \
        timeout --foreground --signal=TERM --kill-after=20s "${BENCH_TIMEOUT_S}s" \
            "$PYTHON_BIN" "$BENCH_DIR/replay_scheduler.py" \
            --config-id "$BENCH_CONFIG_ID" \
            --mode "$BENCH_MODE" \
            --backend "$BENCH_BACKEND" \
            --batch-search "$BENCH_BATCH_SEARCH" \
            --trace "$TRACE_PATH" \
            --runtime-root "$BENCH_CASE_RUNTIME_ROOT" \
            --results-dir "$RESULT_DIR/results" \
            --summary "$SUMMARY_PATH" \
            --code-cache-dir "$BENCH_CODE_CACHE_DIR" \
            --duration-s "$BENCH_DURATION_S" \
            --vram-budget-gib "${BENCH_VRAM_BUDGET_GIB:-28.0}" \
            --max-packed-jobs-per-gpu "${BENCH_MAX_PACKED_JOBS_PER_GPU:-2}" \
            --cache-warm-policy "${BENCH_CACHE_WARM_POLICY:-top_k}" \
            --cache-warm-top-k "${BENCH_CACHE_WARM_TOP_K:-4}" \
            --cache-entry-capacity "${BENCH_CACHE_ENTRY_CAPACITY:-8}" \
            --cache-max-ram-percent "${BENCH_CACHE_MAX_RAM_PERCENT:-0.20}" \
            --cache-memory-budget-gib "${BENCH_CACHE_MEMORY_BUDGET_GIB:-6.0}" \
            --binary-range-up "${BENCH_BINARY_RANGE_UP:-16}" \
            --binary-range-down "${BENCH_BINARY_RANGE_DOWN:-8}" \
            --power-of-two-range-up "${BENCH_POWER_OF_TWO_RANGE_UP:-4}" \
            --power-of-two-range-down "${BENCH_POWER_OF_TWO_RANGE_DOWN:-1}" \
            --target-vram-fraction "${BENCH_TARGET_VRAM_FRACTION:-0.97}"; then
        RC=0
    else
        RC=$?
    fi
fi

T1="$(date +%s.%N)"
if [ -n "${BENCH_GPU_SAMPLER_PID:-}" ]; then
    bench_stop_pid "$BENCH_GPU_SAMPLER_PID"
    unset BENCH_GPU_SAMPLER_PID
fi

bench_elapsed_seconds "$T0" "$T1" > "$RESULT_DIR/wall_clock.txt"
printf '%s\n' "$RC" > "$RESULT_DIR/rc.txt"
printf '%s\n' "$RESULT_DIR"
exit "$RC"

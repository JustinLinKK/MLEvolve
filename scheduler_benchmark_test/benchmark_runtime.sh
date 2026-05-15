#!/usr/bin/env bash

if [ "${BENCH_RUNTIME_SH_LOADED:-0}" = "1" ]; then
    return 0
fi
BENCH_RUNTIME_SH_LOADED=1

bench_bool() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

bench_mps_enabled() {
    bench_bool "${BENCH_ENABLE_MPS:-false}"
}

bench_has_mps_control() {
    command -v nvidia-cuda-mps-control >/dev/null 2>&1
}

bench_init_runtime() {
    local script_name="$1"
    local bench_dir="$2"
    local default_enable_mps="true"
    if [ "${NO_MPS:-0}" = "1" ]; then
        default_enable_mps="false"
    fi

    BENCH_SCRIPT_NAME="$script_name"
    BENCH_DIR="$bench_dir"
    BENCH_RUN_ROOT="${BENCH_RUN_ROOT:-$BENCH_DIR/runtime}"
    BENCH_GPU_SAMPLER="${BENCH_GPU_SAMPLER:-query}"
    BENCH_CLEANUP_GRACE_SECONDS="${BENCH_CLEANUP_GRACE_SECONDS:-10}"
    BENCH_ENABLE_MPS="${BENCH_ENABLE_MPS:-$default_enable_mps}"

    mkdir -p "$BENCH_RUN_ROOT"
    local stamp
    stamp="$(date -u +"%Y%m%dT%H%M%SZ")"
    BENCH_SESSION_ID="${BENCH_SESSION_ID:-${BENCH_SCRIPT_NAME}-${stamp}-$$-${RANDOM}}"
    BENCH_SESSION_DIR="$BENCH_RUN_ROOT/$BENCH_SESSION_ID"
    BENCH_CASE_ROOT="$BENCH_SESSION_DIR/cases"
    BENCH_PID_DIR="$BENCH_SESSION_DIR/pids"
    mkdir -p "$BENCH_CASE_ROOT" "$BENCH_PID_DIR"

    BENCH_TRACKED_PIDS=()
    BENCH_CLEANUP_DONE=0
    export BENCH_SCRIPT_NAME BENCH_DIR BENCH_RUN_ROOT BENCH_GPU_SAMPLER BENCH_CLEANUP_GRACE_SECONDS BENCH_ENABLE_MPS
    export BENCH_SESSION_ID BENCH_SESSION_DIR BENCH_CASE_ROOT BENCH_PID_DIR

    trap 'bench_handle_signal INT' INT
    trap 'bench_handle_signal TERM' TERM
    trap 'bench_cleanup' EXIT
}

bench_prepare_case() {
    local case_id="$1"
    BENCH_CASE_ID="$case_id"
    BENCH_CASE_DIR="$BENCH_CASE_ROOT/$case_id"
    BENCH_CASE_RUNTIME_ROOT="$BENCH_CASE_DIR/runtime"
    BENCH_CASE_WORKDIR_ROOT="$BENCH_CASE_DIR/workdirs"
    BENCH_CASE_MPS_ROOT="$BENCH_CASE_DIR/mps"
    BENCH_CASE_MPS_PIPE_DIR="$BENCH_CASE_MPS_ROOT/pipe"
    BENCH_CASE_MPS_LOG_DIR="$BENCH_CASE_MPS_ROOT/log"
    mkdir -p "$BENCH_CASE_RUNTIME_ROOT" "$BENCH_CASE_WORKDIR_ROOT" "$BENCH_CASE_MPS_PIPE_DIR" "$BENCH_CASE_MPS_LOG_DIR"

    export BENCH_CASE_ID BENCH_CASE_DIR BENCH_CASE_RUNTIME_ROOT BENCH_CASE_WORKDIR_ROOT
    export BENCH_CASE_MPS_ROOT BENCH_CASE_MPS_PIPE_DIR BENCH_CASE_MPS_LOG_DIR
    export BENCH_MPS_PIPE_DIRECTORY="$BENCH_CASE_MPS_PIPE_DIR"
    export BENCH_MPS_LOG_DIRECTORY="$BENCH_CASE_MPS_LOG_DIR"
    export CUDA_MPS_PIPE_DIRECTORY="$BENCH_CASE_MPS_PIPE_DIR"
    export CUDA_MPS_LOG_DIRECTORY="$BENCH_CASE_MPS_LOG_DIR"
}

bench_register_pid() {
    local pid="${1:-}"
    if [ -z "$pid" ]; then
        return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    local existing
    for existing in "${BENCH_TRACKED_PIDS[@]:-}"; do
        if [ "$existing" = "$pid" ]; then
            return 0
        fi
    done
    BENCH_TRACKED_PIDS+=("$pid")
    printf '%s\n' "$pid" > "$BENCH_PID_DIR/$pid.pid"
}

bench_unregister_pid() {
    local pid="${1:-}"
    if [ -z "$pid" ]; then
        return 0
    fi
    local keep=()
    local existing
    for existing in "${BENCH_TRACKED_PIDS[@]:-}"; do
        if [ "$existing" != "$pid" ]; then
            keep+=("$existing")
        fi
    done
    BENCH_TRACKED_PIDS=("${keep[@]}")
    rm -f "$BENCH_PID_DIR/$pid.pid"
}

bench_wait_for_exit() {
    local pid="$1"
    local timeout_seconds="${2:-10}"
    local waited=0
    while kill -0 "$pid" 2>/dev/null; do
        if [ "$waited" -ge "$timeout_seconds" ]; then
            return 1
        fi
        sleep 1
        waited=$(( waited + 1 ))
    done
    return 0
}

bench_stop_pid() {
    local pid="$1"
    if ! kill -0 "$pid" 2>/dev/null; then
        bench_unregister_pid "$pid"
        return 0
    fi
    kill -TERM "$pid" 2>/dev/null || true
    if ! bench_wait_for_exit "$pid" "$BENCH_CLEANUP_GRACE_SECONDS"; then
        kill -KILL "$pid" 2>/dev/null || true
        bench_wait_for_exit "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
    bench_unregister_pid "$pid"
}

bench_cleanup() {
    if [ "${BENCH_CLEANUP_DONE:-0}" = "1" ]; then
        return 0
    fi
    BENCH_CLEANUP_DONE=1
    local idx
    for (( idx=${#BENCH_TRACKED_PIDS[@]}-1; idx>=0; idx-- )); do
        bench_stop_pid "${BENCH_TRACKED_PIDS[$idx]}"
    done
}

bench_handle_signal() {
    local signal_name="$1"
    bench_cleanup
    trap - EXIT
    if [ "$signal_name" = "INT" ]; then
        exit 130
    fi
    exit 143
}

bench_start_gpu_sampler() {
    local output_path="$1"
    local device_index="${2:-0}"
    local sampler_mode="${BENCH_GPU_SAMPLER:-query}"

    if [ "$sampler_mode" = "none" ]; then
        return 0
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[bench] nvidia-smi not found; skipping GPU sampler" >&2
        return 0
    fi

    case "$sampler_mode" in
        query)
            (
                printf '%s\n' "timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
                while true; do
                    nvidia-smi \
                        --id="$device_index" \
                        --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
                        --format=csv,noheader,nounits
                    sleep 1
                done
            ) > "$output_path" 2>&1 &
            ;;
        dmon)
            nvidia-smi -i "$device_index" dmon -s pucvmet -d 1 -o T > "$output_path" 2>&1 &
            ;;
        *)
            echo "[bench] unsupported BENCH_GPU_SAMPLER=$sampler_mode" >&2
            return 1
            ;;
    esac

    BENCH_GPU_SAMPLER_PID=$!
    export BENCH_GPU_SAMPLER_PID
    bench_register_pid "$BENCH_GPU_SAMPLER_PID"
}

bench_elapsed_seconds() {
    local started="$1"
    local finished="$2"
    awk -v t0="$started" -v t1="$finished" 'BEGIN { printf "%.6f\n", (t1 - t0) }'
}

bench_run_logged() {
    local log_path="$1"
    shift
    local had_errexit=0
    if [[ $- == *e* ]]; then
        had_errexit=1
        set +e
    fi
    "$@" > "$log_path" 2>&1 &
    local child_pid=$!
    bench_register_pid "$child_pid"
    wait "$child_pid"
    local rc=$?
    if [ "$had_errexit" -eq 1 ]; then
        set -e
    fi
    bench_unregister_pid "$child_pid"
    return "$rc"
}

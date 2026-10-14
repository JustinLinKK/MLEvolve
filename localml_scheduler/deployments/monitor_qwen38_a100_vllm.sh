#!/usr/bin/env bash
set -uo pipefail

DEPLOY_ROOT="${DEPLOY_ROOT:-/root/downeyflyfan/qwen38-v100-int8}"
PID_FILE="${PID_FILE:-$DEPLOY_ROOT/state/vllm-a100.pid}"
BOOTSTRAP_SCRIPT="${BOOTSTRAP_SCRIPT:-/root/downeyflyfan/.cache/mlevolve_a100_bootstrap_qwen38_a100.sh}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8000/health}"
PROCESS_MARKER="${PROCESS_MARKER:-vllm.entrypoints.cli.main}"
CURL_BIN="${CURL_BIN:-curl}"
CHECK_INTERVAL="${CHECK_INTERVAL:-20}"
RUN_ONCE="${RUN_ONCE:-0}"
SERVER_LOG="${SERVER_LOG:-$DEPLOY_ROOT/logs/vllm-local-int8-a100-tp1.log}"
CRASH_LOG_DIR="${CRASH_LOG_DIR:-/root/downeyflyfan/.cache/qwen-a100-crash-logs}"

server_process_is_alive() {
  local pid cmdline
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  case "$pid" in
    ''|*[!0-9]*) return 1 ;;
  esac
  kill -0 "$pid" 2>/dev/null || return 1
  cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
  [[ "$cmdline" == *"$PROCESS_MARKER"* ]]
}

preserve_server_log() {
  local stamp
  [[ -s "$SERVER_LOG" ]] || return 0
  mkdir -p "$CRASH_LOG_DIR"
  stamp="$(date -u +%Y%m%dT%H%M%S%NZ)"
  cp -- "$SERVER_LOG" "$CRASH_LOG_DIR/vllm-a100-crash-$stamp.log"
}

check_once() {
  if "$CURL_BIN" --fail --silent "$HEALTH_URL" >/dev/null 2>&1; then
    return 0
  fi
  if server_process_is_alive; then
    return 0
  fi
  preserve_server_log || true
  printf '%s vLLM server is dead; invoking bootstrap\n' "$(date -u +%FT%TZ)"
  "$BOOTSTRAP_SCRIPT"
}

if [[ "$RUN_ONCE" == 1 ]]; then
  check_once
  exit $?
fi

while true; do
  check_once || true
  sleep "$CHECK_INTERVAL"
done

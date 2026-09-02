#!/usr/bin/env bash
set -uo pipefail

STATE_DIR="${STATE_DIR:-/root/downeyflyfan/.cache/mlevolve_scheduler_current_a10_v1}"
WATCHDOG_STATE_DIR="${WATCHDOG_STATE_DIR:-/root/downeyflyfan/.cache/mlevolve_scheduler_watchdog_a10_v1}"
POLL_SECONDS="${POLL_SECONDS:-120}"
STALL_SECONDS="${STALL_SECONDS:-3600}"
STOP_GRACE_SECONDS="${STOP_GRACE_SECONDS:-30}"
RUN_ONCE="${RUN_ONCE:-0}"

mkdir -p "$WATCHDOG_STATE_DIR"

now_epoch() {
  if [[ -n "${NOW_EPOCH:-}" ]]; then
    printf '%s\n' "$NOW_EPOCH"
  else
    date +%s
  fi
}

read_required_file() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  head -n 1 "$path"
}

effective_node_count() {
  local runner_log="$1"
  local count
  count="$({ sed -n 's/.*Scheduler-controlled progress: \([0-9][0-9]*\)\/[0-9][0-9]* budget-counted nodes.*/\1/p' "$runner_log" 2>/dev/null || true; } | sort -n | tail -n 1)"
  printf '%s\n' "${count:-0}"
}

run_start_epoch() {
  local runner_log="$1"
  local timestamp
  timestamp="$(sed -n 's/^\[\([^]]*\)\] INFO: Starting run .*/\1/p' "$runner_log" 2>/dev/null | head -n 1)"
  timestamp="${timestamp%%,*}"
  [[ -n "$timestamp" ]] || return 1
  date -u -d "$timestamp" +%s 2>/dev/null
}

process_is_running() {
  local pid="$1"
  local state
  kill -0 "$pid" 2>/dev/null || return 1
  state="$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null || true)"
  [[ "$state" != "Z" && -n "$state" ]]
}

is_verified_scheduler() {
  local pid="$1"
  local cmdline
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  process_is_running "$pid" || return 1
  cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
  [[ "$cmdline" == *"run.py"* && "$cmdline" == *"petfinder_scheduler_profile_hkwd"* ]]
}

capture_diagnostics() {
  local comparison_root="$1"
  local phase="$2"
  local pid="$3"
  local count="$4"
  local elapsed="$5"
  local stamp stall_dir runner_log
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  stall_dir="$comparison_root/watchdog_stalls/$stamp"
  runner_log="$comparison_root/$phase.runner.out"
  mkdir -p "$stall_dir"

  {
    printf 'detected_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'comparison_root=%s\n' "$comparison_root"
    printf 'phase=%s\n' "$phase"
    printf 'scheduler_pid=%s\n' "$pid"
    printf 'effective_nodes=%s\n' "$count"
    printf 'seconds_without_progress=%s\n' "$elapsed"
    printf '\nPROCESS\n'
    ps -p "$pid" -o pid,ppid,lstart,etime,stat,pcpu,pmem,args --cols 5000 2>&1 || true
    printf '\nGPU\n'
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader 2>&1 || true
    printf '\nRUNNER_TAIL\n'
    tail -n 240 "$runner_log" 2>&1 || true
  } > "$stall_dir/diagnostics.txt"

  [[ -f "$runner_log" ]] && cp -- "$runner_log" "$stall_dir/runner.out"
  find "$comparison_root/$phase" -maxdepth 3 -type f \
    \( -name 'pipeline.sqlite3' -o -name 'journal.json' -o -name 'node_records.jsonl' -o -name 'MLEvolve.log' \) \
    -exec cp --parents -- '{}' "$stall_dir" \; 2>/dev/null || true
  printf '%s\n' "$stall_dir" > "$WATCHDOG_STATE_DIR/last_stall_dir"
}

stop_stalled_scheduler() {
  local pid="$1"
  local deadline
  if ! is_verified_scheduler "$pid"; then
    printf 'refusing to stop unverified process pid=%s\n' "$pid" >&2
    return 43
  fi

  kill -TERM "$pid"
  deadline=$(( $(date +%s) + STOP_GRACE_SECONDS ))
  while process_is_running "$pid" && (( $(date +%s) < deadline )); do
    sleep 1
  done
  if process_is_running "$pid"; then
    if ! is_verified_scheduler "$pid"; then
      printf 'refusing force-stop after process identity changed pid=%s\n' "$pid" >&2
      return 43
    fi
    kill -KILL "$pid"
  fi
  return 0
}

check_once() {
  local comparison_root phase pid runner_log current_count current_time
  local monitored_root last_count last_progress elapsed initial_progress

  comparison_root="$(read_required_file "$STATE_DIR/comparison_root")" || return 0
  phase="$(read_required_file "$STATE_DIR/active_phase")" || return 0
  pid="$(read_required_file "$STATE_DIR/scheduler.pid")" || return 0
  runner_log="$comparison_root/$phase.runner.out"
  current_count="$(effective_node_count "$runner_log")"
  current_time="$(now_epoch)"
  monitored_root="$(read_required_file "$WATCHDOG_STATE_DIR/monitored_root" 2>/dev/null || true)"

  if [[ "$monitored_root" != "$comparison_root" ]]; then
    initial_progress="$(run_start_epoch "$runner_log" 2>/dev/null || printf '%s\n' "$current_time")"
    if (( initial_progress > current_time )); then
      initial_progress="$current_time"
    fi
    printf '%s\n' "$comparison_root" > "$WATCHDOG_STATE_DIR/monitored_root"
    printf '%s\n' "$current_count" > "$WATCHDOG_STATE_DIR/last_count"
    printf '%s\n' "$initial_progress" > "$WATCHDOG_STATE_DIR/last_progress_epoch"
  fi

  last_count="$(read_required_file "$WATCHDOG_STATE_DIR/last_count" 2>/dev/null || printf '0\n')"
  last_progress="$(read_required_file "$WATCHDOG_STATE_DIR/last_progress_epoch" 2>/dev/null || printf '%s\n' "$current_time")"

  if (( current_count > last_count )); then
    last_count="$current_count"
    last_progress="$current_time"
    printf '%s\n' "$last_count" > "$WATCHDOG_STATE_DIR/last_count"
    printf '%s\n' "$last_progress" > "$WATCHDOG_STATE_DIR/last_progress_epoch"
  fi

  elapsed=$(( current_time - last_progress ))
  printf 'checked_utc=%s pid=%s effective_nodes=%s seconds_without_progress=%s\n' \
    "$(date -u +%FT%TZ)" "$pid" "$current_count" "$elapsed" > "$WATCHDOG_STATE_DIR/heartbeat"

  if (( elapsed < STALL_SECONDS )); then
    return 0
  fi

  if ! is_verified_scheduler "$pid"; then
    printf 'refusing to stop unverified process pid=%s\n' "$pid" >&2
    return 43
  fi

  capture_diagnostics "$comparison_root" "$phase" "$pid" "$current_count" "$elapsed"
  cat > "$WATCHDOG_STATE_DIR/STALL_DETECTED.json" <<EOF
{"reason":"no_effective_node_progress","scheduler_pid":$pid,"effective_nodes":$current_count,"seconds_without_progress":$elapsed,"comparison_root":"$comparison_root"}
EOF
  printf '%s\n' stalled_stopped > "$STATE_DIR/status"
  stop_stalled_scheduler "$pid" || return $?
  return 42
}

if [[ "$RUN_ONCE" == "1" ]]; then
  check_once
  exit $?
fi

while true; do
  check_once
  status=$?
  if (( status == 42 || status == 43 )); then
    exit "$status"
  fi
  sleep "$POLL_SECONDS"
done

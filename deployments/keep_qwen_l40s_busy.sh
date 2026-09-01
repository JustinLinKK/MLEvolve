#!/usr/bin/env bash
set -euo pipefail

endpoint="${ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}"
model_name="${MODEL_NAME:-qwen3.8-27b-int8-l40s}"
max_tokens="${MAX_TOKENS:-1024}"
retry_seconds="${RETRY_SECONDS:-2}"
concurrency="${CONCURRENCY:-16}"

if [[ ! "$max_tokens" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_TOKENS must be a positive integer." >&2
  exit 2
fi
if [[ ! "$concurrency" =~ ^[1-9][0-9]*$ ]]; then
  echo "CONCURRENCY must be a positive integer." >&2
  exit 2
fi

printf -v payload \
  '{"model":"%s","messages":[{"role":"user","content":"Write a detailed technical analysis of profile-based GPU scheduling, including admission control, runtime estimation, and live telemetry."}],"max_tokens":%s,"temperature":0.7,"stream":false}' \
  "$model_name" "$max_tokens"

run_once() {
  curl --fail --silent --show-error \
    --header 'Content-Type: application/json' \
    --data-binary "$payload" \
    "$endpoint" \
    >/dev/null
}

if [[ "${ONESHOT:-0}" == 1 ]]; then
  run_once
  exit 0
fi

echo "l40s_goal_filler"
run_worker() {
  while true; do
    if ! run_once; then
      sleep "$retry_seconds"
    fi
  done
}

worker_pids=()
cleanup_workers() {
  local pid
  for pid in "${worker_pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait "${worker_pids[@]}" 2>/dev/null || true
}
trap cleanup_workers EXIT

for _ in $(seq 1 "$concurrency"); do
  run_worker &
  worker_pids+=("$!")
done
wait "${worker_pids[@]}"

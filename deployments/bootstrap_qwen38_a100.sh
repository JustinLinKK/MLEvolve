#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT="${DEPLOY_ROOT:-/root/downeyflyfan/qwen38-v100-int8}"
MODEL_DIR="${MODEL_DIR:-$DEPLOY_ROOT/models/Qwen3.8-27B-INT8-W8A16-MTP}"
RUNTIME_DIR="${RUNTIME_DIR:-/tmp/mlevolve-vllm-a100}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache-a100}"
VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/vllm-cache-a100}"
LOG_DIR="${LOG_DIR:-$DEPLOY_ROOT/logs}"
STATE_DIR="${STATE_DIR:-$DEPLOY_ROOT/state}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b-int8-a100}"
SERVER_LOG="$LOG_DIR/vllm-local-int8-a100-tp1.log"

mkdir -p "$LOG_DIR" "$STATE_DIR" "$VLLM_CACHE_ROOT"
test -d "$MODEL_DIR"

if ! "$RUNTIME_DIR/bin/python" -c 'import vllm' >/dev/null 2>&1; then
  export UV_CACHE_DIR
  uv venv --python python3 --seed "$RUNTIME_DIR"
  uv pip install --python "$RUNTIME_DIR/bin/python" vllm==0.27.1
fi

if curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  exit 0
fi

nohup env CUDA_VISIBLE_DEVICES=0 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT" \
  "$RUNTIME_DIR/bin/python" -c 'from vllm.entrypoints.cli.main import main; main()' \
    serve "$MODEL_DIR" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --language-model-only \
    --skip-mm-profiling \
    --mamba-cache-mode align \
    --enable-prefix-caching \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    > "$SERVER_LOG" 2>&1 < /dev/null &
server_pid=$!
printf '%s\n' "$server_pid" > "$STATE_DIR/vllm-a100.pid"

for _ in $(seq 1 360); do
  if curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    exit 0
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -n 120 "$SERVER_LOG" >&2
    exit 1
  fi
  sleep 5
done

tail -n 120 "$SERVER_LOG" >&2
exit 1

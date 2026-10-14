#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT="${DEPLOY_ROOT:-/root/downeyflyfan/qwen38-v100-int8}"
MODEL_DIR="$DEPLOY_ROOT/models/Qwen3.8-27B-INT8-W8A16-MTP"
RUNTIME_DIR="${RUNTIME_DIR:-/tmp/mlevolve-vllm-l40s}"
LOG_DIR="$DEPLOY_ROOT/logs"
PORT=8000
ACCELERATOR="${ACCELERATOR:?ACCELERATOR is required}"
MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:?MODEL_CUDA_VISIBLE_DEVICES is required}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:?TENSOR_PARALLEL_SIZE is required}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
RUN_LABEL="${ACCELERATOR}-tp${TENSOR_PARALLEL_SIZE}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b-int8-l40s}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

mkdir -p "$RUNTIME_DIR" "$LOG_DIR" "$DEPLOY_ROOT/results"
test -d "$MODEL_DIR"

if ! "$RUNTIME_DIR/bin/python" -c 'import vllm' >/dev/null 2>&1; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache-l40s}"
  uv venv --python python3 --seed "$RUNTIME_DIR"
  uv pip install --python "$RUNTIME_DIR/bin/python" vllm==0.27.1
fi
RUNTIME_PYTHON="$RUNTIME_DIR/bin/python"

if ! curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  nohup env CUDA_VISIBLE_DEVICES="$MODEL_CUDA_VISIBLE_DEVICES" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    "$RUNTIME_PYTHON" -c 'from vllm.entrypoints.cli.main import main; main()' serve "$MODEL_DIR" \
      --host "$VLLM_HOST" \
      --port "$PORT" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --pipeline-parallel-size 1 \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-model-len "$MAX_MODEL_LEN" \
      --language-model-only \
      --skip-mm-profiling \
      --mamba-cache-mode align \
      --enforce-eager \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_coder \
      --default-chat-template-kwargs '{"enable_thinking":false}' \
      > "$LOG_DIR/vllm-exact-int8-${RUN_LABEL}.log" 2>&1 < /dev/null &
fi

for _ in $(seq 1 180); do
  if curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    if ! "$RUNTIME_PYTHON" -c 'import matplotlib, requests' >/dev/null 2>&1; then
      uv pip install --python "$RUNTIME_PYTHON" matplotlib requests
    fi
    nohup env MODEL_PATH="$MODEL_DIR" \
      MODEL_NAME="$SERVED_MODEL_NAME" \
      OUT="$DEPLOY_ROOT/results/qwen38_${RUN_LABEL}_benchmark.json" \
      PLOT="$DEPLOY_ROOT/results/qwen38_${RUN_LABEL}_benchmark.png" \
      "$RUNTIME_PYTHON" /bootstrap/benchmark.py \
      > "$LOG_DIR/qwen38_${RUN_LABEL}_benchmark.log" 2>&1 < /dev/null &
    exit 0
  fi
  sleep 5
done

tail -n 80 "$LOG_DIR/vllm-exact-int8-${RUN_LABEL}.log" >&2
exit 1

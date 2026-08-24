#!/usr/bin/env bash
set -euo pipefail

VLLM_VENV="${VLLM_VENV:-/tmp/qwen38-vllm-v100-probe-venv}"
MODEL_DIR="${MODEL_DIR:-models/Qwen3.8-27B}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
EAGER_FLAG=()
if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  EAGER_FLAG=(--enforce-eager)
fi

exec env \
  CUDA_VISIBLE_DEVICES=0,1 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$VLLM_VENV/bin/vllm" serve "$MODEL_DIR" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --served-model-name qwen3.8-27b-fp16 \
  --dtype float16 \
  --tensor-parallel-size 2 \
  --max-model-len 1024 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --mamba-cache-mode align \
  --skip-mm-profiling \
  "${EAGER_FLAG[@]}"

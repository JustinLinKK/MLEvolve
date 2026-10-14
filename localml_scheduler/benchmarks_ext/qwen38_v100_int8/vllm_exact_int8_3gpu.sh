#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VLLM_BIN="${VLLM_BIN:-vllm}"
MODEL_ID="lued/Qwen3.8-27B-INT8-W8A16-MTP"
MODEL_DIR="${MODEL_DIR:-models/Qwen3.8-27B-INT8-W8A16-MTP}"
HF_HOME="${HF_HOME:-$PWD/cache}"
PORT="${PORT:-8000}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-3}"

if [[ "${DOWNLOAD_ONLY:-0}" == "1" ]]; then
  HF_HOME="$HF_HOME" "$PYTHON_BIN" - "$MODEL_ID" "$MODEL_DIR" <<'PY'
from pathlib import Path
import sys
from huggingface_hub import snapshot_download

snapshot_download(repo_id=sys.argv[1], local_dir=Path(sys.argv[2]))
PY
  exit 0
fi

exec env \
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$VLLM_BIN" serve "$MODEL_DIR" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --served-model-name qwen3.8-27b-int8-w8a16 \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.92 \
  --mamba-cache-mode align \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}'

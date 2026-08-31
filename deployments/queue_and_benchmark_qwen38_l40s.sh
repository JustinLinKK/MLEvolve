#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_ROOT="/root/downeyflyfan/qwen38-v100-int8"

ssh Nautilus-V100 'hostname; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader'
scp "$ROOT/deployments/bootstrap_qwen38_l40s.sh" Nautilus-V100:"$REMOTE_ROOT/bootstrap_qwen38_l40s.sh"
ssh Nautilus-V100 "chmod 755 $REMOTE_ROOT/bootstrap_qwen38_l40s.sh && $REMOTE_ROOT/bootstrap_qwen38_l40s.sh"
scp "$ROOT/benchmarks/qwen38_v100_int8/benchmark.py" Nautilus-V100:"$REMOTE_ROOT/benchmark_qwen38_l40s.py"
ssh Nautilus-V100 "PYTHONPATH=$REMOTE_ROOT/runtime-vllm-l40s \
  MODEL_PATH=$REMOTE_ROOT/models/Qwen3.8-27B-INT8-W8A16-MTP \
  MODEL_NAME=qwen3.8-27b-int8-w8a16-l40s-tp1 \
  OUT=$REMOTE_ROOT/results/qwen38_l40s_1gpu_benchmark.json \
  PLOT=$REMOTE_ROOT/results/qwen38_l40s_1gpu_benchmark.png \
  python3 $REMOTE_ROOT/benchmark_qwen38_l40s.py"
scp Nautilus-V100:"$REMOTE_ROOT/results/qwen38_l40s_1gpu_benchmark.json" "$ROOT/records/"
scp Nautilus-V100:"$REMOTE_ROOT/results/qwen38_l40s_1gpu_benchmark.png" "$ROOT/records/"

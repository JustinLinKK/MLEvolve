#!/usr/bin/env bash
# Run locally once. Nautilus-A10's ProxyCommand waits for the queued pod.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_ROOT="${REMOTE_ROOT:-/root/qwen38-v100-int8}"
RESULT_DIR="$REMOTE_ROOT/results/a10_tp4"

ssh Nautilus-A10 'hostname; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader'
scp "$ROOT/localml_scheduler/benchmarks_ext/qwen38_v100_int8/benchmark.py" \
  "$ROOT/localml_scheduler/benchmarks_ext/qwen38_v100_int8/a10_backend_runner.py" \
  "$ROOT/localml_scheduler/deployments/run_qwen38_a10_comparison.sh" \
  Nautilus-A10:"$REMOTE_ROOT/"
ssh Nautilus-A10 "chmod 755 $REMOTE_ROOT/run_qwen38_a10_comparison.sh && \
  BENCHMARK_PY=$REMOTE_ROOT/benchmark.py RUNNER_PY=$REMOTE_ROOT/a10_backend_runner.py \
  $REMOTE_ROOT/run_qwen38_a10_comparison.sh"

mkdir -p "$ROOT/records/qwen38_a10_tp4"
scp -r Nautilus-A10:"$RESULT_DIR/." "$ROOT/records/qwen38_a10_tp4/"
scp Nautilus-A10:"$REMOTE_ROOT/logs/qwen38_a10_tp4_vllm.log" \
  "$REMOTE_ROOT/logs/qwen38_a10_tp4_tensorrt_llm.log" \
  "$ROOT/records/qwen38_a10_tp4/"

#!/usr/bin/env bash
set -euo pipefail

# Serve the text-only Qwen3.8-27B W8A16 checkpoint on ABA GPU 1.  That GPU is
# in Exclusive Process mode, which fits one long-lived vLLM server.  GPU 0 is
# left to concurrent MLEvolve candidate processes because it is in Default mode.
repo="${REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
cache_root="${ABA_CACHE_ROOT:-/data1/yufan/MLEvolve/.cache}"
runtime_root="${QWEN_RUNTIME_ROOT:-$cache_root/qwen38_vllm}"
python_bin="${QWEN_PYTHON_BIN:-$runtime_root/.venv-cu129/bin/python}"
cuda_compat_dir="${QWEN_CUDA_COMPAT_DIR:-$runtime_root/cuda-compat-12-9/extracted/usr/local/cuda-12.9/compat}"
model_dir="${QWEN_MODEL_DIR:-$cache_root/models/Qwen3.8-27B-INT8-W8A16-MTP}"
state_dir="${QWEN_STATE_DIR:-$runtime_root/state}"
log_file="${QWEN_LOG_FILE:-$state_dir/vllm.log}"
pid_file="${QWEN_PID_FILE:-$state_dir/vllm.pid}"
port="${QWEN_PORT:-8000}"
served_model_name="${QWEN_SERVED_MODEL_NAME:-qwen3.8-27b-int8-a100}"
max_model_len="${QWEN_MAX_MODEL_LEN:-262144}"

test -f "$repo/run.py"
test -x "$python_bin"
test -r "$cuda_compat_dir/libcuda.so.1"
test -d "$model_dir"
test -r "$model_dir/config.json"
test -r "$model_dir/model.safetensors.index.json"

test "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ge 2
test "$(nvidia-smi --id=0 --query-gpu=name --format=csv,noheader | xargs)" = "NVIDIA A100 80GB PCIe"
test "$(nvidia-smi --id=1 --query-gpu=name --format=csv,noheader | xargs)" = "NVIDIA A100 80GB PCIe"
test "$(nvidia-smi --id=0 --query-gpu=memory.total --format=csv,noheader,nounits | xargs)" -eq 81920
test "$(nvidia-smi --id=1 --query-gpu=memory.total --format=csv,noheader,nounits | xargs)" -eq 81920

mkdir -p "$state_dir"
if curl --fail --silent --max-time 5 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
  exit 0
fi

if test -s "$pid_file"; then
  old_pid="$(cat "$pid_file")"
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "A recorded Qwen process is alive but its health endpoint is unavailable: $old_pid" >&2
    exit 70
  fi
fi

nohup env \
  LD_LIBRARY_PATH="$cuda_compat_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  CUDA_VISIBLE_DEVICES=1 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_CACHE_ROOT="$runtime_root/cache" \
  "$python_bin" -m vllm.entrypoints.cli.main serve "$model_dir" \
    --host 127.0.0.1 \
    --port "$port" \
    --served-model-name "$served_model_name" \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --max-model-len "$max_model_len" \
    --language-model-only \
    --skip-mm-profiling \
    --mamba-cache-mode align \
    --enable-prefix-caching \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    > "$log_file" 2>&1 < /dev/null &
server_pid=$!
printf '%s\n' "$server_pid" > "$pid_file"

for _ in $(seq 1 360); do
  if curl --fail --silent --max-time 5 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
    exit 0
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -n 160 "$log_file" >&2
    exit 71
  fi
  sleep 5
done

tail -n 160 "$log_file" >&2
exit 72

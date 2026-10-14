#!/usr/bin/env bash
# Run on Nautilus-A10 after the pod is Ready. It owns only PIDs it starts.
set -euo pipefail

DEPLOY_ROOT="${DEPLOY_ROOT:-/root/qwen38-v100-int8}"
MODEL_DIR="${MODEL_DIR:-$DEPLOY_ROOT/models/Qwen3.8-27B-INT8-W8A16-MTP}"
BENCHMARK_PY="${BENCHMARK_PY:-$DEPLOY_ROOT/benchmark_qwen38_a10.py}"
RUNNER_PY="${RUNNER_PY:-$DEPLOY_ROOT/a10_backend_runner.py}"
RESULT_DIR="$DEPLOY_ROOT/results/a10_tp4"
LOG_DIR="$DEPLOY_ROOT/logs"
MAX_START_SECONDS="${MAX_START_SECONDS:-900}"

mkdir -p "$RESULT_DIR" "$LOG_DIR"
test -d "$MODEL_DIR"
test -f "$BENCHMARK_PY"
test -f "$RUNNER_PY"

wait_for_health() {
  local port="$1"
  local elapsed=0
  until curl --fail --silent "http://127.0.0.1:${port}/health" >/dev/null; do
    if (( elapsed >= MAX_START_SECONDS )); then
      return 1
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
}

model_name() {
  local port="$1"
  curl --fail --silent "http://127.0.0.1:${port}/v1/models" | python3 -c \
    'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])'
}

start_backend() {
  local backend="$1"
  local port="$2"
  local pid_file="$RESULT_DIR/${backend}.pid"
  local log_file="$LOG_DIR/qwen38_a10_tp4_${backend}.log"

  if curl --fail --silent "http://127.0.0.1:${port}/health" >/dev/null; then
    echo "refusing to overwrite an existing service on port ${port}" >&2
    return 1
  fi

  if [[ "$backend" == "vllm" ]]; then
    command -v vllm >/dev/null || python3 -m pip install --no-cache-dir vllm==0.27.1
    nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_WORKER_MULTIPROC_METHOD=spawn \
      VLLM_USE_FLASHINFER_SAMPLER=0 \
      vllm serve "$MODEL_DIR" --host=127.0.0.1 --port="$port" \
      --served-model-name=qwen3.8-27b-int8-w8a16-a10-tp4 \
      --tensor-parallel-size=4 --gpu-memory-utilization=0.90 \
      --max-model-len=4096 --language-model-only --skip-mm-profiling \
      --mamba-cache-mode=align \
      --speculative-config='{"method":"mtp","num_speculative_tokens":3}' \
      >"$log_file" 2>&1 < /dev/null &
  elif [[ "$backend" == "tensorrt_llm" ]]; then
    command -v trtllm-serve >/dev/null || \
      python3 -m pip install --no-cache-dir tensorrt_llm
    nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 \
      trtllm-serve "$MODEL_DIR" --backend=pytorch --host=127.0.0.1 --port="$port" \
      --tp_size=4 --max_seq_len=4096 --trust_remote_code --reasoning_parser=qwen3 \
      >"$log_file" 2>&1 < /dev/null &
  else
    echo "unknown backend: $backend" >&2
    return 2
  fi
  echo "$!" > "$pid_file"
}

stop_owned_backend() {
  local backend="$1"
  local pid_file="$RESULT_DIR/${backend}.pid"
  [[ -f "$pid_file" ]] || return 0
  local pid
  pid="$(<"$pid_file")"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    wait "$pid" 2>/dev/null || true
  fi
  rm -f "$pid_file"
}

run_backend() {
  local backend="$1"
  local port="$2"
  local output="$RESULT_DIR/${backend}.json"
  local plot="$RESULT_DIR/${backend}.png"
  local status="$RESULT_DIR/${backend}.status.json"

  if ! start_backend "$backend" "$port" || ! wait_for_health "$port"; then
    python3 - "$backend" "$status" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[2]).write_text(json.dumps({"backend": sys.argv[1], "ok": False}))
PY
    stop_owned_backend "$backend"
    return 0
  fi

  local served_model
  served_model="$(model_name "$port")"
  if ! MODEL_PATH="$MODEL_DIR" MODEL_NAME="$served_model" \
    URL="http://127.0.0.1:${port}/v1/chat/completions" OUT="$output" PLOT="$plot" \
    python3 "$BENCHMARK_PY"; then
    python3 - "$backend" "$status" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[2]).write_text(json.dumps({"backend": sys.argv[1], "ok": False}))
PY
    stop_owned_backend "$backend"
    return 0
  fi

  python3 - "$backend" "$output" "$status" <<'PY'
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[2]).read_text())
Path(sys.argv[3]).write_text(json.dumps({
    "backend": sys.argv[1], "ok": True,
    "tokens_per_second": report["median"]["tokens_per_second"],
    "ttft_seconds": report["median"]["ttft_seconds"],
}))
PY
  stop_owned_backend "$backend"
}

run_backend vllm 8000
run_backend tensorrt_llm 8001

winner="$(python3 - "$RUNNER_PY" "$RESULT_DIR" <<'PY'
import importlib.util
import json
import sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("a10_backend_runner", sys.argv[1])
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
results = module.load_status_results(Path(sys.argv[2]))
print(module.select_fastest(results))
PY
)"

echo "$winner" | tee "$RESULT_DIR/winner.txt"
if [[ "$winner" == "vllm" ]]; then
  start_backend vllm 8000
  wait_for_health 8000
else
  start_backend tensorrt_llm 8001
  wait_for_health 8001
fi

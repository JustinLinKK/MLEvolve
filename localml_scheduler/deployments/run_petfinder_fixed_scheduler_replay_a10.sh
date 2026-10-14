#!/usr/bin/env bash
# Execute a frozen PetFinder baseline candidate under Scheduler-only admission.
# Scheduler state and candidate workspace stay on tmpfs so Ceph metadata latency
# cannot become a false scheduler result; every artifact is copied on exit.
set -euo pipefail

sources="${1:?pass frozen source directory}"
durable_root="${2:?pass durable result directory}"
repo="${REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${A10_PYTHON_BIN:-/dev/shm/mlevolve_a10_clean_venv/bin/python}"
predictor_root="${PREDICTOR_V32_SOURCE_ROOT:-/dev/shm/mlevolve_predictor_v32_20260916/source}"
predictor_weights="${PREDICTOR_V32_WEIGHTS:-/dev/shm/mlevolve_predictor_v32_20260916/weights/predictor_v3.2_student.pt}"
data_root="${PETFINDER_DATA_DIR:-/dev/shm/mlevolve_petfinder_public_seed42}"
limit="${PETFINDER_NODE_BUDGET:-1}"
time_limit="${PETFINDER_SEARCH_TIME_LIMIT:-7200}"
run_name="petfinder_fixed_scheduler_replay_$(date -u +%Y%m%d_%H%M%S)"
volatile_root="/dev/shm/mlevolve_${run_name}"

test -x "$python_bin"
test -d "$sources"
test -f "$sources/manifest.json"
test -d "$data_root"
test -d "$predictor_root"
test -f "$predictor_weights"
test -f "$repo/localml_scheduler/benchmark_test/run_petfinder_fixed_baseline_replay.py"
[[ "$limit" =~ ^[1-9][0-9]*$ ]]
[[ "$time_limit" =~ ^[1-9][0-9]*$ ]]
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qx 'NVIDIA A10'

mkdir -p "$durable_root"
persist() {
  local status="$?"
  if test -d "$volatile_root"; then
    cp -a "$volatile_root/." "$durable_root/"
  fi
  exit "$status"
}
trap persist EXIT

env \
  CUDA_VISIBLE_DEVICES=0 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONHASHSEED=42 \
  PYTHONPATH="$repo:$predictor_root${PYTHONPATH:+:$PYTHONPATH}" \
  "$python_bin" -u "$repo/localml_scheduler/benchmark_test/run_petfinder_fixed_baseline_replay.py" \
  --sources "$sources" \
  --data-root "$data_root" \
  --output-root "$volatile_root" \
  --candidate-python "$python_bin" \
  --predictor-v32-source-root "$predictor_root" \
  --predictor-v32-artifact "$predictor_weights" \
  --limit "$limit" \
  --time-limit-seconds "$time_limit"

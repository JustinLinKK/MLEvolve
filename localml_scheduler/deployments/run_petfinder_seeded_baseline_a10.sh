#!/usr/bin/env bash
# Scheduler-free PetFinder control.  It uses the same local Qwen service and
# seed contract as the treatment, while leaving only the scheduler disabled.
set -euo pipefail

repo="${REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
execution_repo="${MLEVOLVE_EXECUTION_REPO:-$repo}"
python_bin="${A10_PYTHON_BIN:-/dev/shm/mlevolve_a10_clean_venv/bin/python}"
agent_url="${QWEN_VLLM_URL:-http://mlevolve-qwen-a100.ecepxie.svc.cluster.local:8000/v1}"
data_dir="${PETFINDER_DATA_DIR:-/dev/shm/mlevolve_petfinder_public_seed42}"
desc_file="${PETFINDER_DESC_FILE:-$data_dir/description.md}"
torch_hub_dir="${TORCH_HUB_DIR:-/root/.cache/torch}"
node_budget="${PETFINDER_NODE_BUDGET:-1}"
search_time_limit="${PETFINDER_SEARCH_TIME_LIMIT:-10800}"
run_stamp="$(date -u +%Y%m%d_%H%M%S)"
run_name="petfinder_seeded_baseline_a10_${run_stamp}"
durable_root="${DURABLE_ROOT:-$repo/runs/$run_name}"
runtime_root="/dev/shm/mlevolve_$run_name"
volatile_root="$runtime_root/experiment"

test -x "$python_bin"
test -f "$execution_repo/run.py"
test -r "$data_dir/train.csv"
test -r "$data_dir/test.csv"
test -r "$data_dir/sample_submission.csv"
test -d "$data_dir/train"
test -d "$data_dir/test"
test -r "$desc_file"
[[ "$node_budget" =~ ^[1-9][0-9]*$ ]]
[[ "$search_time_limit" =~ ^[1-9][0-9]*$ ]]
curl --fail --silent --show-error --max-time 10 "${agent_url%/v1}/health" >/dev/null
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 1
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qx 'NVIDIA A10'

mkdir -p "$durable_root" "$volatile_root"

persist_runtime() {
  local status="$?"
  if test -d "$volatile_root"; then
    cp -a "$volatile_root/." "$durable_root/"
  fi
  if test -d "$runtime_root/pipeline"; then
    cp -a "$runtime_root/pipeline" "$durable_root/"
  fi
  exit "$status"
}
trap persist_runtime EXIT

cd "$execution_repo"
env \
  CUDA_VISIBLE_DEVICES=0 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONHASHSEED=42 \
  MLEVOLVE_CONFIG="$execution_repo/config/config.yaml" \
  MLEVOLVE_PIPELINE_SQLITE_DIR="$runtime_root/pipeline" \
  PYTHONPATH="$execution_repo${PYTHONPATH:+:$PYTHONPATH}" \
  "$python_bin" -u run.py \
  "exp_name=$run_name" \
  "log_dir=$volatile_root" "workspace_dir=$volatile_root" \
  "data_dir=$data_dir" "desc_file=$desc_file" \
  "torch_hub_dir=$torch_hub_dir" \
  exec.timeout=7200 \
  agent.steps="$node_budget" agent.time_limit="$search_time_limit" agent.seed=42 \
  agent.initial_drafts=1 agent.data_preview=true agent.clean_task_description=false agent.require_cuda_execution=true agent.check_data_leakage=true \
  agent.use_stepwise_generation=false agent.use_diff_mode=true agent.use_global_memory=false \
  experiment.mode=baseline \
  scheduler.enabled=false \
  hardware_knowledge.enabled=false agent.hardware_context_enabled=false \
  preflight.enabled=false \
  agent.pipeline_decision_enabled=false agent.review.enabled=false \
  agent.code.provider=openai-compatible agent.feedback.provider=openai-compatible \
  agent.code.model=qwen3.8-27b-int8-a100 agent.feedback.model=qwen3.8-27b-int8-a100 \
  agent.code.temp=0.7 agent.feedback.temp=0.7 \
  "agent.code.base_url=$agent_url" "agent.feedback.base_url=$agent_url" \
  agent.code.api_key=EMPTY agent.feedback.api_key=EMPTY \
  use_grading_server=false \
  agent.search.parallel_search_num=null agent.search.num_gpus=1

#!/usr/bin/env bash
# Run the controlled 50-node PetFinder Predictor v3.2 scheduler experiment on one A10.
# Hot SQLite scheduler/pipeline state is kept off Ceph; results remain durable.
set -euo pipefail

repo="${REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
execution_repo="${MLEVOLVE_EXECUTION_REPO:-$repo}"
python_bin="${A10_PYTHON_BIN:-/root/downeyflyfan/mlevolve_a10_baseline_20260829/.venv/bin/python}"
predictor_root="${PREDICTOR_V32_SOURCE_ROOT:-/tmp/Predictor/predictor_v3.2/source}"
predictor_weights="${PREDICTOR_V32_WEIGHTS:-/tmp/Predictor/predictor_v3.2/weights/predictor_v3.2_student.pt}"
preflight_root="${MODEL_PREFLIGHT_SOURCE_ROOT:-$execution_repo/Preflight_checker/src}"
# This Scheduler-only study still includes the local pre-flight checker.  It
# screens generated-code faults before GPU admission; HWKD remains disabled
# separately in the launch arguments below.
preflight_enabled="${PREFLIGHT_ENABLED:-true}"
agent_url="${QWEN_VLLM_URL:-http://mlevolve-qwen-a100.ecepxie.svc.cluster.local:8000/v1}"
# A fully materialized local cache avoids long-lived CephFS metadata stalls in
# DataLoader workers.  It must contain the canonical PetFinder public files.
data_dir="${PETFINDER_DATA_DIR:-/root/downeyflyfan/mle-bench-data/petfinder-pawpularity-score/prepared/public}"
desc_file="${PETFINDER_DESC_FILE:-$data_dir/description.md}"
# Keep torchvision source and its canonical checkpoints off CephFS.  The
# caller stages this cache before launch; this override only selects it.
torch_hub_dir="${TORCH_HUB_DIR:-/tmp/mlevolve_torch_hub}"
# Match the historical 9/2 Pathfinder baseline node budget.  The only
# treatment components are the scheduler and Predictor v3.2.  Pre-flight is
# available as an explicit opt-in for a separate checker experiment.
node_budget="${PETFINDER_NODE_BUDGET:-50}"
search_time_limit="${PETFINDER_SEARCH_TIME_LIMIT:-28800}"
run_stamp="$(date -u +%Y%m%d_%H%M%S)"
run_name="petfinder_scheduler_v32_a10_${run_stamp}"
durable_root="${DURABLE_ROOT:-$repo/runs/$run_name}"
runtime_root="/dev/shm/mlevolve_$run_name"
volatile_root="$runtime_root/experiment"

test -x "$python_bin"
test -f "$execution_repo/run.py"
test -d "$predictor_root"
test -f "$predictor_weights"
[[ "$preflight_enabled" == "true" || "$preflight_enabled" == "false" ]]
if [[ "$preflight_enabled" == "true" ]]; then
  test -d "$preflight_root/model_preflight"
fi
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
  if test -d "$runtime_root/scheduler_runtime"; then
    cp -a "$runtime_root/scheduler_runtime" "$durable_root/"
  fi
  # The pipeline SQLite journal is a sibling of the experiment directory.
  # Preserve it with the scheduler state so per-node outcomes remain
  # reconstructible after /dev/shm is reclaimed.
  if test -d "$runtime_root/pipeline"; then
    cp -a "$runtime_root/pipeline" "$durable_root/"
  fi
  exit "$status"
}
trap persist_runtime EXIT

# The controller creates isolated process groups for its scheduler workers.
# Killing only this launcher would otherwise orphan the controller, leaving it
# able to dispatch more work after the caller believes the experiment stopped.
# Walk only descendants of this run's controller, from leaves to root, so a
# signal never reaches another workload on the shared GPU node.
stop_controller_tree() {
  local parent_pid="$1"
  local signal_name="$2"
  local child_pid
  for child_pid in $(pgrep -P "$parent_pid" 2>/dev/null || true); do
    stop_controller_tree "$child_pid" "$signal_name"
  done
  kill "-$signal_name" "$parent_pid" 2>/dev/null || true
}

controller_pid=""
handle_stop_signal() {
  if [[ -n "$controller_pid" ]] && kill -0 "$controller_pid" 2>/dev/null; then
    stop_controller_tree "$controller_pid" TERM
    # Give Python and its worker supervisor a bounded chance to flush SQLite
    # and result records, then enforce the requested stop for this tree only.
    for _ in 1 2 3; do
      kill -0 "$controller_pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$controller_pid" 2>/dev/null; then
      stop_controller_tree "$controller_pid" KILL
    fi
    wait "$controller_pid" 2>/dev/null || true
  fi
  exit 143
}
trap handle_stop_signal INT TERM

# Preserve the baseline's OpenAI-compatible client contract while targeting
# the same local A100 vLLM service and model.
cd "$execution_repo"
env \
  CUDA_VISIBLE_DEVICES=0 \
  MLEVOLVE_CONFIG="$execution_repo/config/config.yaml" \
  MLEVOLVE_PIPELINE_SQLITE_DIR="$runtime_root/pipeline" \
  PYTHONPATH="$execution_repo:$predictor_root${PYTHONPATH:+:$PYTHONPATH}" \
  "$python_bin" -u run.py \
  "exp_name=$run_name" \
  "log_dir=$volatile_root" "workspace_dir=$volatile_root" \
  "data_dir=$data_dir" "desc_file=$desc_file" \
  "torch_hub_dir=$torch_hub_dir" \
  exec.timeout=7200 \
  agent.steps="$node_budget" agent.time_limit="$search_time_limit" agent.seed=42 \
  agent.initial_drafts=3 agent.data_preview=true agent.clean_task_description=false agent.require_cuda_execution=true agent.check_data_leakage=true \
  agent.use_diff_mode=true agent.use_global_memory=false \
  experiment.mode=hardware_aware \
  scheduler.enabled=true \
  "scheduler.runtime_root=$runtime_root/scheduler_runtime" \
  scheduler.settings.prediction.mode=ml_predictor \
  scheduler.settings.prediction.require_ml_prediction=true \
  "scheduler.settings.prediction.ml.v32_source_root=$predictor_root" \
  "scheduler.settings.prediction.ml.v32_artifact_path=$predictor_weights" \
  scheduler.settings.prediction.ml.v32_capture_device=cuda \
  scheduler.settings.prediction.ml.v32_async_source_capture=false \
  scheduler.settings.prediction.ml.source_conversion_timeout_seconds=60 \
  scheduler.settings.gpu_scheduler.enabled=true \
  scheduler.settings.gpu_scheduler.mode=parallel_time_aware \
  scheduler.settings.gpu_scheduler.scheduler_decision_mode=backend_aware \
  scheduler.settings.gpu_scheduler.packing_backend=cuda_process \
  scheduler.settings.gpu_scheduler.submission_defaults.backend_allowlist='[cuda_process]' \
  scheduler.settings.gpu_scheduler.parallel_job_cap=null \
  hardware_knowledge.enabled=false agent.hardware_context_enabled=false \
  "preflight.enabled=$preflight_enabled" \
  agent.pipeline_decision_enabled=false agent.review.enabled=false \
  agent.code.provider=openai-compatible agent.feedback.provider=openai-compatible \
  agent.code.model=qwen3.8-27b-int8-a100 agent.feedback.model=qwen3.8-27b-int8-a100 \
  agent.code.temp=0.7 agent.feedback.temp=0.7 \
  "agent.code.base_url=$agent_url" "agent.feedback.base_url=$agent_url" \
  agent.code.api_key=EMPTY agent.feedback.api_key=EMPTY \
  agent.search.parallel_search_num=null agent.search.num_gpus=1 &
controller_pid=$!
controller_status=0
# A foreground `wait "$controller_pid"` defers Bash signal traps until the
# controller exits.  Poll through short-lived sleep children instead, so TERM
# reaches `handle_stop_signal` promptly and cannot leave a live controller
# dispatching work after the launcher was told to stop.
while kill -0 "$controller_pid" 2>/dev/null; do
  controller_state="$(ps -o stat= -p "$controller_pid" 2>/dev/null || true)"
  [[ "$controller_state" == *Z* ]] && break
  sleep 0.2 &
  wait "$!" || true
done
wait "$controller_pid" || controller_status=$?
exit "$controller_status"

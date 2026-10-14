#!/usr/bin/env bash
set -euo pipefail

phase="${1:-}"
case "$phase" in
  baseline|scheduler) ;;
  *) echo "usage: $0 baseline|scheduler" >&2; exit 64 ;;
esac

current_repo="${REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
cache_root="${ABA_CACHE_ROOT:-/data1/yufan/MLEvolve/.cache}"
original_baseline_repo="${ORIGINAL_MLEVOLVE_BASELINE_REPO:-$cache_root/original_mlevolve_baseline_source}"
python_bin="${ABA_MLEVOLVE_PYTHON_BIN:-$current_repo/.venv/bin/python}"
data_dir="${PETFINDER_DATA_DIR:-$cache_root/data/petfinder-pawpularity-score}"
desc_file="${PETFINDER_DESC_FILE:-$data_dir/description.md}"
predictor_root="${PREDICTOR_V32_SOURCE_ROOT:-$cache_root/predictor_v3.2/source}"
predictor_weights="${PREDICTOR_V32_WEIGHTS:-$cache_root/predictor_v3.2/weights/predictor_v3.2_transfer_A100.pt}"
agent_url="${QWEN_VLLM_URL:-http://127.0.0.1:8000/v1}"
model_name="${QWEN_SERVED_MODEL_NAME:-qwen3.8-27b-int8-a100}"
pair_root="${PAIR_ROOT:?PAIR_ROOT must identify this controlled comparison}"
torch_hub_dir="${TORCH_HUB_DIR:-$cache_root/torch}"
search_time_limit="${PETFINDER_SEARCH_TIME_LIMIT:-86400}"

if [[ "$phase" == baseline ]]; then
  node_budget="${BASELINE_NODE_BUDGET:-50}"
  phase_repo="$original_baseline_repo"
else
  node_budget="${SCHEDULER_NODE_BUDGET:-50}"
  phase_repo="$current_repo"
fi

run_name="petfinder_aba_a100_${phase}_seed42"
durable_root="$pair_root/$phase"
runtime_root="/dev/shm/${run_name}_$$"
volatile_root="$runtime_root/experiment"

test -x "$python_bin"
test -f "$phase_repo/run.py"
test -r "$phase_repo/config/config.yaml"
test -r "$data_dir/train.csv"
test -r "$data_dir/test.csv"
test -r "$data_dir/sample_submission.csv"
test -d "$data_dir/train"
test -d "$data_dir/test"
test -r "$desc_file"
[[ "$node_budget" =~ ^[1-9][0-9]*$ ]]
[[ "$search_time_limit" =~ ^[1-9][0-9]*$ ]]
curl --fail --silent --show-error --max-time 10 "${agent_url%/v1}/health" >/dev/null

test "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ge 2
test "$(nvidia-smi --id=0 --query-gpu=name --format=csv,noheader | xargs)" = "NVIDIA A100 80GB PCIe"
test "$(nvidia-smi --id=0 --query-gpu=memory.total --format=csv,noheader,nounits | xargs)" -eq 81920
compute_mode="$(nvidia-smi --id=0 --query-gpu=compute_mode --format=csv,noheader | xargs)"
if [[ "$compute_mode" == "Exclusive_Process" ]]; then
  echo "ABA GPU 0 is in Exclusive_Process mode; concurrent candidate processes would fail." >&2
  exit 73
fi
experiment_gpu_used="$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if (( experiment_gpu_used > 256 )); then
  echo "ABA GPU 0 is not idle (${experiment_gpu_used} MiB used); refusing to mix workloads." >&2
  exit 73
fi

if [[ "$phase" == scheduler ]]; then
  test -d "$predictor_root/perfseer_v32"
  test -r "$predictor_weights"
fi

mkdir -p "$durable_root" "$volatile_root" "$torch_hub_dir"

persist_runtime() {
  local status="$?"
  if test -d "$volatile_root"; then
    cp -a "$volatile_root/." "$durable_root/"
  fi
  if test -d "$runtime_root/scheduler_runtime"; then
    cp -a "$runtime_root/scheduler_runtime" "$durable_root/"
  fi
  if test -d "$runtime_root/pipeline"; then
    cp -a "$runtime_root/pipeline" "$durable_root/"
  fi
  printf '%s\n' "$status" > "$durable_root/exit_code"
  exit "$status"
}
trap persist_runtime EXIT

common_args=(
  "exp_name=$run_name"
  "log_dir=$volatile_root"
  "workspace_dir=$volatile_root"
  "data_dir=$data_dir"
  "dataset_dir=$data_dir"
  "desc_file=$desc_file"
  "torch_hub_dir=$torch_hub_dir"
  "exec.timeout=7200"
  "agent.steps=$node_budget"
  "agent.time_limit=$search_time_limit"
  "agent.seed=42"
  "agent.initial_drafts=3"
  "agent.data_preview=true"
  "agent.check_data_leakage=true"
  "agent.use_stepwise_generation=false"
  "agent.use_diff_mode=true"
  "agent.use_global_memory=false"
  "agent.code.provider=openai-compatible"
  "agent.feedback.provider=openai-compatible"
  "agent.code.model=$model_name"
  "agent.feedback.model=$model_name"
  "agent.code.temp=0.7"
  "agent.feedback.temp=0.7"
  "agent.code.base_url=$agent_url"
  "agent.feedback.base_url=$agent_url"
  "agent.code.api_key=EMPTY"
  "agent.feedback.api_key=EMPTY"
  "use_grading_server=false"
  "agent.search.parallel_search_num=null"
  "agent.search.num_gpus=1"
)

baseline_args=(
  "scheduler.enabled=false"
)

scheduler_args=(
    "experiment.mode=baseline"
    "vllm_client.default_completion_tokens=8192"
    "context_cache.enabled=false"
    "context_cache.local_pack_cache_enabled=false"
    "context_cache.provider_prompt_cache_enabled=false"
    "hardware_knowledge.enabled=false"
    "agent.hardware_context_enabled=false"
    "preflight.enabled=false"
    "lesson_profiles.read_enabled=false"
    "lesson_profiles.write_enabled=false"
    "agent.pipeline_decision_enabled=false"
    "agent.review.enabled=false"
    "agent.cuda_docs.enabled=false"
    "scheduler.enabled=true"
    "scheduler.runtime_root=$runtime_root/scheduler_runtime"
    "scheduler.settings.runtime_root=$runtime_root/scheduler_runtime"
    "scheduler.settings.prediction.mode=ml_predictor"
    "scheduler.settings.prediction.require_ml_prediction=true"
    "scheduler.settings.prediction.ml.v32_source_root=$predictor_root"
    "scheduler.settings.prediction.ml.v32_artifact_path=$predictor_weights"
    "scheduler.settings.prediction.ml.v32_capture_device=cuda"
    "scheduler.settings.prediction.ml.v32_async_source_capture=false"
    "scheduler.settings.prediction.ml.source_conversion_timeout_seconds=60"
    "scheduler.settings.gpu_scheduler.enabled=true"
    "scheduler.settings.gpu_scheduler.mode=parallel_time_aware"
    "scheduler.settings.gpu_scheduler.scheduler_decision_mode=backend_aware"
    "scheduler.settings.gpu_scheduler.packing_backend=cuda_process"
    "scheduler.settings.gpu_scheduler.submission_defaults.backend_allowlist=[cuda_process]"
)

if [[ "$phase" == baseline ]]; then
  variant_args=("${baseline_args[@]}")
  phase_pythonpath="$phase_repo"
else
  variant_args=("${scheduler_args[@]}")
  phase_pythonpath="$phase_repo:$predictor_root"
fi

printf '%s\n' "$phase_repo" > "$durable_root/source_root"
printf '%s\n' "$compute_mode" > "$durable_root/experiment_gpu_compute_mode"

cd "$phase_repo"
env \
  CUDA_VISIBLE_DEVICES=0 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONHASHSEED=42 \
  MLEVOLVE_CONFIG="$phase_repo/config/config.yaml" \
  MLEVOLVE_PIPELINE_SQLITE_DIR="$runtime_root/pipeline" \
  PYTHONPATH="$phase_pythonpath${PYTHONPATH:+:$PYTHONPATH}" \
  "$python_bin" -u run.py "${common_args[@]}" "${variant_args[@]}"

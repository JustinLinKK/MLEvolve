#!/usr/bin/env bash
set -euo pipefail

current_repo="${REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
cache_root="${ABA_CACHE_ROOT:-/data1/yufan/MLEvolve/.cache}"
python_bin="${ABA_MLEVOLVE_PYTHON_BIN:-$current_repo/.venv/bin/python}"
data_dir="${PETFINDER_DATA_DIR:-$cache_root/data/petfinder-pawpularity-score}"
desc_file="${PETFINDER_DESC_FILE:-$data_dir/description.md}"
agent_url="${QWEN_VLLM_URL:-http://127.0.0.1:8000/v1}"
model_name="${QWEN_SERVED_MODEL_NAME:-qwen3.8-27b-int8-a100}"
node_budget="${PREFLIGHT_NODE_BUDGET:-50}"
search_time_limit="${PETFINDER_SEARCH_TIME_LIMIT:-86400}"
torch_hub_dir="${TORCH_HUB_DIR:-$cache_root/torch}"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_name="petfinder_aba_a100_preflight_only_seed42"
durable_root="${PREFLIGHT_RUN_ROOT:-$cache_root/experiments/${run_name}_${run_stamp}}"
current_pointer="$cache_root/petfinder_aba_a100_preflight_only_current"
runtime_root="/dev/shm/${run_name}_$$"
volatile_root="$runtime_root/experiment"

test -x "$python_bin"
test -r "$current_repo/run.py"
test -r "$current_repo/config/config.yaml"
test -r "$data_dir/train.csv"
test -r "$data_dir/test.csv"
test -r "$data_dir/sample_submission.csv"
test -d "$data_dir/train"
test -d "$data_dir/test"
test -r "$desc_file"
[[ "$node_budget" =~ ^[1-9][0-9]*$ ]]
[[ "$search_time_limit" =~ ^[1-9][0-9]*$ ]]
test "$node_budget" -eq 50

# Physical GPU 1 hosts the local Qwen vLLM service; physical GPU 0 is reserved
# for candidate execution. Refuse to mix with an unidentified GPU 0 workload.
curl --fail --silent --show-error --max-time 10 "${agent_url%/v1}/health" >/dev/null
test "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ge 2
test "$(nvidia-smi --id=0 --query-gpu=name --format=csv,noheader | xargs)" = "NVIDIA A100 80GB PCIe"
test "$(nvidia-smi --id=1 --query-gpu=name --format=csv,noheader | xargs)" = "NVIDIA A100 80GB PCIe"
test "$(nvidia-smi --id=0 --query-gpu=memory.total --format=csv,noheader,nounits | xargs)" -eq 81920
test "$(nvidia-smi --id=1 --query-gpu=memory.total --format=csv,noheader,nounits | xargs)" -eq 81920
candidate_compute_mode="$(nvidia-smi --id=0 --query-gpu=compute_mode --format=csv,noheader | xargs)"
agent_compute_mode="$(nvidia-smi --id=1 --query-gpu=compute_mode --format=csv,noheader | xargs)"
if [[ "$candidate_compute_mode" == "Exclusive_Process" ]]; then
  echo "ABA GPU 0 is in Exclusive_Process mode; concurrent candidate processes would fail." >&2
  exit 73
fi
test "$agent_compute_mode" = "Exclusive_Process"
candidate_gpu_used="$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
agent_gpu_used="$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if (( candidate_gpu_used > 256 )); then
  echo "ABA GPU 0 is not idle (${candidate_gpu_used} MiB used); refusing to mix workloads." >&2
  exit 73
fi
if (( agent_gpu_used < 60000 )); then
  echo "ABA GPU 1 does not look like the expected resident Qwen service (${agent_gpu_used} MiB used)." >&2
  exit 73
fi

mkdir -p "$durable_root" "$volatile_root" "$torch_hub_dir"
ln -sfn "$durable_root" "$current_pointer"

persist_runtime() {
  local exit_status="$?"
  if test -d "$volatile_root"; then
    cp -a "$volatile_root/." "$durable_root/"
  fi
  if test -d "$runtime_root/pipeline"; then
    cp -a "$runtime_root/pipeline" "$durable_root/"
  fi
  printf '%s\n' "$exit_status" > "$durable_root/exit_code"
  exit "$exit_status"
}
trap persist_runtime EXIT

run_args=(
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
  "agent.clean_task_description=true"
  "agent.prompt_compatibility_profile=original_mlevolve_20260924"
  "agent.check_data_leakage=true"
  "agent.require_cuda_execution=false"
  "agent.use_stepwise_generation=false"
  "agent.use_diff_mode=true"
  "agent.use_global_memory=false"
  "experiment.mode=hardware_aware"
  "scheduler.enabled=false"
  "context_cache.enabled=false"
  "context_cache.local_pack_cache_enabled=false"
  "context_cache.provider_prompt_cache_enabled=false"
  "hardware_knowledge.enabled=false"
  "agent.hardware_context_enabled=false"
  "preflight.enabled=true"
  "preflight.enabled_modes=[hardware_aware]"
  "preflight.policy_mode=balanced"
  "preflight.target_profile=nvidia/a100_80gb"
  "preflight.prompt_contract_enabled=false"
  "preflight.require_adapter_for_generated=false"
  "preflight.max_repair_rounds=1"
  "preflight.fail_open_on_internal_error=false"
  "lesson_profiles.read_enabled=false"
  "lesson_profiles.write_enabled=false"
  "agent.pipeline_decision_enabled=false"
  "agent.review.enabled=false"
  "agent.cuda_docs.enabled=false"
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
  "vllm_client.default_completion_tokens=8192"
  "use_grading_server=false"
  "agent.search.parallel_search_num=null"
  "agent.search.num_gpus=1"
)

# Validate the generated-code prompt surface before the first live model call.
# The generator prompt must remain frozen. Candidates that define a
# CandidateAdapter receive the full CPU preflight; baseline-style candidates
# without one still receive the static/hardware fallback instead of an
# adapter-only repair that would confound this checker-only ablation.
PROMPT_AUDIT_PATH="$durable_root/prompt_contract_audit.json" \
PREFLIGHT_RUN_ARGS="$(printf '%s\n' "${run_args[@]}")" \
PYTHONPATH="$current_repo${PYTHONPATH:+:$PYTHONPATH}" \
"$python_bin" - <<'PY'
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from agents.draft_agent import feature_prompt_contracts
from agents.prompts.impl_guideline import get_impl_guideline_from_agent

agent = SimpleNamespace(
    cfg=SimpleNamespace(
        scheduler=SimpleNamespace(
            enabled=False,
            settings=SimpleNamespace(prediction=SimpleNamespace(mode="")),
        ),
        preflight=SimpleNamespace(
            enabled=True,
            prompt_contract_enabled=False,
        ),
        exec=SimpleNamespace(timeout=7200),
        pretrain_model_dir="",
        exp_id="petfinder-pawpularity-score",
    ),
    acfg=SimpleNamespace(
        time_limit=86400,
        steps=50,
        prompt_compatibility_profile="original_mlevolve_20260924",
        expose_prediction=False,
        k_fold_validation=0,
    ),
    start_time=time.time(),
    current_step=0,
)

sections = feature_prompt_contracts(agent)
guideline = get_impl_guideline_from_agent(agent)
rendered = "\n".join(
    [name for name in sections]
    + [line for lines in sections.values() for line in lines]
    + [line for lines in guideline.values() for line in lines]
)
required = [
    "Preserve Task Semantics in Local Validation",
    "Make Validation Auditable",
]
required_absent = ["CPU preflight adapter contract", "CandidateAdapter"]
forbidden = [
    *required_absent,
    "Backend design contract",
    "Scheduler Model Family Contract",
]
missing = [marker for marker in required if marker not in rendered]
unexpected = [marker for marker in forbidden if marker in rendered]
audit = {
    "accepted": not missing and not unexpected,
    "required_markers": required,
    "forbidden_markers": forbidden,
    "missing_markers": missing,
    "unexpected_markers": unexpected,
    "feature_sections": list(sections),
    "run_args": os.environ["PREFLIGHT_RUN_ARGS"].splitlines(),
}
Path(os.environ["PROMPT_AUDIT_PATH"]).write_text(
    json.dumps(audit, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
if not audit["accepted"]:
    raise SystemExit(f"preflight-only prompt gate failed: {audit}")
PY

printf '%s\n' "$current_repo" > "$durable_root/source_root"
printf '%s\n' "$candidate_compute_mode" > "$durable_root/candidate_gpu_compute_mode"
printf '%s\n' "$agent_compute_mode" > "$durable_root/agent_gpu_compute_mode"
printf '%s\n' "${run_args[@]}" > "$durable_root/run_args.txt"

cd "$current_repo"
env \
  CUDA_VISIBLE_DEVICES=0 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONHASHSEED=42 \
  MLEVOLVE_CONFIG="$current_repo/config/config.yaml" \
  MLEVOLVE_PIPELINE_SQLITE_DIR="$runtime_root/pipeline" \
  PYTHONPATH="$current_repo${PYTHONPATH:+:$PYTHONPATH}" \
  "$python_bin" -u run.py "${run_args[@]}"

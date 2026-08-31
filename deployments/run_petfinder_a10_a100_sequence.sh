#!/usr/bin/env bash
set -euo pipefail

repo="${REPO:-/root/downeyflyfan/.cache/mlevolve_a10_a100_comparison_20260831}"
baseline_repo="${BASELINE_REPO:-/root/downeyflyfan/mlevolve_a10_baseline_20260829}"
python_bin="${A10_PYTHON_BIN:-/tmp/mlevolve-a10-run-venv/bin/python}"
python_fallback="/root/downeyflyfan/mlevolve_a10_baseline_20260829/.venv/bin/python"
state_dir="${STATE_DIR:-/root/downeyflyfan/.cache/mlevolve_a100_a10_sequence}"
agent_root="http://mlevolve-qwen-a100.ecepxie.svc.cluster.local:8000"
agent_base_url="${agent_root}/v1"
model_name="qwen3.8-27b-int8-a100"

if [[ ! -x "$python_bin" ]]; then
  python_bin="$python_fallback"
fi
if [[ ! -x "$python_bin" ]]; then
  echo "No usable A10 Python environment found." >&2
  exit 1
fi

mkdir -p "$state_dir" "$repo/runs"
if ! curl --fail --silent --show-error --max-time 10 "$agent_root/health" >/dev/null; then
  echo "A100 agent health check failed." >&2
  exit 1
fi
if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -ne 1 ]] || \
   ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -qx 'NVIDIA A10'; then
  echo "Refusing to run: this comparison requires exactly one NVIDIA A10." >&2
  exit 1
fi

if [[ -s "$state_dir/comparison_root" ]]; then
  comparison_root="$(cat "$state_dir/comparison_root")"
else
  comparison_root="$repo/runs/a100_agent_a10_comparison_$(date -u +%Y%m%d_%H%M%S)"
  printf '%s\n' "$comparison_root" > "$state_dir/comparison_root"
fi
mkdir -p "$comparison_root"

run_phase() {
  local phase="$1"
  local mode="$2"
  local phase_root="$comparison_root/$phase"
  local phase_log="$comparison_root/${phase}.runner.out"
  local phase_repo="$repo"
  local config_path="$repo/config/config.yaml"
  mkdir -p "$phase_root"
  printf '%s\n' "$phase" > "$state_dir/active_phase"

  local common=(
    "exp_name=petfinder_${phase}_a100_agent_a10"
    "log_dir=$phase_root"
    "workspace_dir=$phase_root"
    "agent.steps=50"
    "agent.time_limit=43200"
    "agent.seed=42"
    "agent.code.model=$model_name"
    "agent.code.base_url=$agent_base_url"
    "agent.code.api_key=EMPTY"
    "agent.feedback.model=$model_name"
    "agent.feedback.base_url=$agent_base_url"
    "agent.feedback.api_key=EMPTY"
    "agent.search.parallel_search_num=null"
    "agent.search.num_gpus=1"
  )

  local variant=()
  if [[ "$phase" == baseline ]]; then
    phase_repo="$baseline_repo"
    config_path="$baseline_repo/config/config.yaml"
    variant=(
      "scheduler.enabled=false"
      "agent.code.provider=openai-compatible"
      "agent.feedback.provider=openai-compatible"
      "agent.use_stepwise_generation=false"
    )
  else
    variant=(
      "experiment.mode=$mode"
      "scheduler.enabled=true"
      "scheduler.settings.prediction.mode=branch_profile"
      "scheduler.settings.gpu_scheduler.parallel_job_cap=null"
      "scheduler.runtime_root=$phase_root/scheduler_runtime"
      "hardware_knowledge.enabled=true"
      "hardware_knowledge.settings.runtime_root=$phase_root/hardware_knowledge_runtime"
      "preflight.enabled=true"
      "preflight.target_profile=nvidia/a10_24gb"
      "agent.hardware_context_enabled=true"
      "agent.pipeline_decision_enabled=true"
      "agent.review.enabled=true"
      "agent.code.provider=vllm"
      "agent.feedback.provider=vllm"
    )
  fi

  cd "$phase_repo"
  set +e
  env CUDA_VISIBLE_DEVICES=0 MLEVOLVE_CONFIG="$config_path" \
    "$python_bin" run.py "${common[@]}" "${variant[@]}" \
    > "$phase_log" 2>&1
  local status=$?
  set -e
  if [[ "$status" -eq 0 ]]; then
    local journal
    journal="$(find "$phase_root" -type f -path '*/logs/journal.json' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
    local retained=0
    if [[ -n "$journal" ]]; then
      retained="$($python_bin -c 'import json,sys; print(max(0, len(json.load(open(sys.argv[1])).get("nodes", [])) - 1))' "$journal")"
    fi
    if [[ "$retained" -ne 50 ]]; then
      echo "$phase exited without 50 retained nodes (found $retained)." >&2
      status=75
    fi
  fi
  printf '%s\n' "$status" > "$comparison_root/${phase}.exit_code"
  if [[ "$status" -ne 0 ]]; then
    echo "$phase failed with exit code $status; see $phase_log" >&2
    return "$status"
  fi
}

if [[ ! -f "$comparison_root/baseline.exit_code" ]] || \
   [[ "$(cat "$comparison_root/baseline.exit_code")" != 0 ]]; then
  run_phase baseline baseline
fi
if [[ ! -f "$comparison_root/scheduler_profile_hkwd.exit_code" ]] || \
   [[ "$(cat "$comparison_root/scheduler_profile_hkwd.exit_code")" != 0 ]]; then
  run_phase scheduler_profile_hkwd hardware_aware
fi

printf 'complete\n' > "$state_dir/status"
printf '%s\n' "$comparison_root" > "$state_dir/completed_root"

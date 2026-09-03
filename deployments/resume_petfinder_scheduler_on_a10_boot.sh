#!/usr/bin/env bash
# Resume the persistent PetFinder scheduler run after an A10 pod replacement.
# The journal and source tree live on the /root persistent volume.
set -euo pipefail

repo=/root/downeyflyfan/.cache/mlevolve_a10_a100_comparison_20260831
outer_root="$repo/runs/a100_agent_a10_scheduler_preflightpureconfigfix_20260902_191617"
phase=scheduler_profile_hkwd
phase_root="$outer_root/$phase"
run_root="$phase_root/20260902_191623_petfinder_scheduler_profile_hkwd_a100_agent_a10_preflightpureconfigfix"
state_dir=/root/downeyflyfan/.cache/mlevolve_scheduler_current_a10_v7
# Watchdog observations are disposable and must not block resume on Ceph.
watch_dir=/dev/shm/mlevolve_scheduler_watchdog_a10_v7
python_bin=/tmp/mlevolve-a10-run-venv/bin/python
python_fallback=/root/downeyflyfan/mlevolve_a10_baseline_20260829/.venv/bin/python

if [[ ! -x "$python_bin" ]]; then
    python_bin="$python_fallback"
fi
test -x "$python_bin"
test -f "$run_root/logs/journal.json"
mkdir -p "$state_dir" "$watch_dir"

if "$python_bin" - "$run_root/logs/journal.json" <<'PY'
import json
import sys

from engine.node_accounting import count_budget_nodes_from_json

raise SystemExit(0 if count_budget_nodes_from_json(sys.argv[1]) >= 50 else 1)
PY
then
    printf 'complete\n' > "$state_dir/status"
    exit 0
fi

cd "$repo"
nohup env CUDA_VISIBLE_DEVICES=0 MLEVOLVE_CONFIG="$repo/config/config.yaml" \
    PYTHONPATH="$repo/nn-model-preflight-checker/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$python_bin" run.py \
    exp_name=petfinder_scheduler_profile_hkwd_a100_agent_a10_preflightpureconfigfix \
    log_dir="$phase_root" workspace_dir="$phase_root" \
    agent.steps=50 agent.time_limit=43200 agent.seed=42 \
    agent.code.model=qwen3.8-27b-int8-a100 \
    agent.code.base_url=http://mlevolve-qwen-a100.ecepxie.svc.cluster.local:8000/v1 agent.code.api_key=EMPTY \
    agent.feedback.model=qwen3.8-27b-int8-a100 \
    agent.feedback.base_url=http://mlevolve-qwen-a100.ecepxie.svc.cluster.local:8000/v1 agent.feedback.api_key=EMPTY \
    agent.search.parallel_search_num=null agent.search.num_gpus=1 \
    experiment.mode=hardware_aware scheduler.enabled=true scheduler.settings.prediction.mode=branch_profile \
    scheduler.settings.gpu_scheduler.parallel_job_cap=null \
    scheduler.runtime_root="$phase_root/scheduler_runtime" \
    hardware_knowledge.enabled=true hardware_knowledge.settings.runtime_root="$phase_root/hardware_knowledge_runtime" \
    preflight.enabled=true preflight.target_profile=nvidia/a10_24gb preflight.max_repair_rounds=4 \
    agent.hardware_context_enabled=true agent.pipeline_decision_enabled=true \
    agent.review.enabled=true agent.review.max_repair_rounds=4 \
    agent.cuda_docs.enabled=true agent.cuda_docs.rollout_mode=debug_cached \
    agent.code.provider=vllm agent.feedback.provider=vllm \
    resume_journal="$run_root/logs/journal.json" \
    > "$outer_root/$phase.runner.out" 2>&1 < /dev/null &
scheduler_pid=$!
printf '%s\n' "$scheduler_pid" > "$state_dir/scheduler.pid"
printf 'running\n' > "$state_dir/status"

nohup env STATE_DIR="$state_dir" WATCHDOG_STATE_DIR="$watch_dir" \
    POLL_SECONDS=120 STALL_SECONDS=3600 STOP_GRACE_SECONDS=30 \
    bash "$repo/deployments/monitor_petfinder_scheduler_progress.sh" \
    > "$outer_root/watchdog.out" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$state_dir/watchdog.pid"

# Keep the development pod reachable after the experiment process terminates.
wait "$scheduler_pid" || true
tail -f /dev/null

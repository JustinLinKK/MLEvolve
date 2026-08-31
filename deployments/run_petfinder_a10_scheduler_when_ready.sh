#!/usr/bin/env bash
set -euo pipefail

namespace="${NAUTILUS_NAMESPACE:-ecepxie}"
a10_selector="app=gpu-dev-a10-experiment"
l40s_selector="app=gpu-dev-l40s-1gpu"
repo="/root/downeyflyfan/mlevolve_a10_scheduler_active_20260831"
python_bin="${A10_PYTHON_BIN:-/tmp/mlevolve-a10-run-venv/bin/python}"
python_fallback="/root/downeyflyfan/mlevolve_a10_baseline_20260829/.venv/bin/python"
state_dir="/root/downeyflyfan/.cache/mlevolve_goal_controller"
service_url="http://mlevolve-qwen-l40s.ecepxie.svc.cluster.local:8000"

running_pod() {
  local selector="$1"
  kubectl -n "$namespace" get pods -l "$selector" -o jsonpath='{range .items[?(@.status.phase=="Running")]}{.metadata.name}{"\n"}{end}' 2>/dev/null | head -n 1
}

start_a10_filler() {
  local pod="$1"
  kubectl -n "$namespace" exec "$pod" -- env STATE_DIR="$state_dir" bash -lc '
    set -e
    mkdir -p "$STATE_DIR"
    if test -s "$STATE_DIR/a10_filler.pid"; then
      pid=$(cat "$STATE_DIR/a10_filler.pid")
      if kill -0 "$pid" 2>/dev/null && tr "\0" " " < "/proc/$pid/cmdline" | grep -q a10_goal_filler; then
        exit 0
      fi
    fi
    nohup python3 -u -c '\''import torch; marker="a10_goal_filler"; n=24576; a=torch.randn((n,n),device="cuda",dtype=torch.float16); b=torch.randn((n,n),device="cuda",dtype=torch.float16); print(marker,flush=True); exec("while True:\n torch.mm(a,b,out=a)")'\'' \
      > "$STATE_DIR/a10_filler.log" 2>&1 < /dev/null &
    echo $! > "$STATE_DIR/a10_filler.pid"
  '
}

stop_owned_filler() {
  local pod="$1"
  local name="$2"
  kubectl -n "$namespace" exec "$pod" -- env STATE_DIR="$state_dir" FILLER_NAME="$name" bash -lc '
    pid_file="$STATE_DIR/${FILLER_NAME}_filler.pid"
    test -s "$pid_file" || exit 0
    pid=$(cat "$pid_file")
    if kill -0 "$pid" 2>/dev/null && tr "\0" " " < "/proc/$pid/cmdline" | grep -q "${FILLER_NAME}_goal_filler"; then
      kill "$pid"
    fi
  ' || true
}

start_l40s_filler() {
  local pod="$1"
  kubectl -n "$namespace" exec "$pod" -- env STATE_DIR="$state_dir" bash -lc '
    set -e
    mkdir -p "$STATE_DIR"
    if test -s "$STATE_DIR/l40s_filler.pid"; then
      pid=$(cat "$STATE_DIR/l40s_filler.pid")
      if kill -0 "$pid" 2>/dev/null && tr "\0" " " < "/proc/$pid/cmdline" | grep -q l40s_goal_filler; then
        exit 0
      fi
    fi
    nohup python3 -u -c '\''import requests,time; marker="l40s_goal_filler"; print(marker,flush=True); p={"model":"qwen3.8-27b-int8-l40s","messages":[{"role":"user","content":"Write a compact technical analysis of profile-based GPU scheduling."}],"max_tokens":512,"temperature":0.7}; exec("while True:\n try: requests.post(\"http://127.0.0.1:8000/v1/chat/completions\",json=p,timeout=600).raise_for_status()\n except Exception as e: print(repr(e),flush=True); time.sleep(2)")'\'' \
      > "$STATE_DIR/l40s_filler.log" 2>&1 < /dev/null &
    echo $! > "$STATE_DIR/l40s_filler.pid"
  '
}

agent_healthy() {
  local pod="$1"
  kubectl -n "$namespace" exec "$pod" -- curl --fail --silent --max-time 5 http://127.0.0.1:8000/health >/dev/null 2>&1
}

launch_or_resume_scheduler() {
  local pod="$1"
  kubectl -n "$namespace" exec "$pod" -- env \
    REPO="$repo" PYTHON_BIN="$python_bin" PYTHON_FALLBACK="$python_fallback" STATE_DIR="$state_dir" bash -lc '
    set -euo pipefail
    if ! test -x "$PYTHON_BIN"; then PYTHON_BIN="$PYTHON_FALLBACK"; fi
    mkdir -p "$STATE_DIR" "$REPO/runs"
    resolve_run_root() {
      local outer="$1"
      if test -d "$outer/logs"; then
        printf "%s\n" "$outer"
        return
      fi
      local nested
      nested=$(find "$outer" -mindepth 1 -maxdepth 1 -type d -name "*_petfinder_scheduler_profile_hkwd_a10_l40s" -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -n 1 | cut -d" " -f2-)
      printf "%s\n" "${nested:-$outer}"
    }
    if test -s "$STATE_DIR/active_run"; then
      outer_root=$(cat "$STATE_DIR/active_run")
    else
      outer_root="$REPO/runs/a10_scheduler_profile_hkwd_l40s_$(date -u +%Y%m%d_%H%M%S)"
      printf "%s\n" "$outer_root" > "$STATE_DIR/active_run"
    fi
    mkdir -p "$outer_root"
    run_root=$(resolve_run_root "$outer_root")
    if test -s "$STATE_DIR/scheduler.pid"; then
      old_pid=$(cat "$STATE_DIR/scheduler.pid")
      if kill -0 "$old_pid" 2>/dev/null && tr "\0" " " < "/proc/$old_pid/cmdline" | grep -q "run.py"; then
        printf "already running pid=%s run=%s\n" "$old_pid" "$run_root"
        exit 0
      fi
    fi
    if test -f "$run_root/logs/comparison_metrics.json" && \
       "$PYTHON_BIN" -c '\''import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("node_count") == 50 else 1)'\'' \
         "$run_root/logs/comparison_metrics.json"; then
      printf "already complete run=%s\n" "$run_root"
      exit 0
    fi
    resume_arg=()
    if test -f "$run_root/logs/journal.json"; then
      resume_arg=("resume_journal=$run_root/logs/journal.json")
    fi
    cd "$REPO"
    nohup env CUDA_VISIBLE_DEVICES=0 MLEVOLVE_CONFIG="$REPO/config/config.yaml" \
      "$PYTHON_BIN" run.py \
      "log_dir=$outer_root" \
      "workspace_dir=$outer_root" \
      "scheduler.runtime_root=$run_root/scheduler_runtime" \
      "hardware_knowledge.settings.runtime_root=$run_root/hardware_knowledge_runtime" \
      "${resume_arg[@]}" \
      > "$run_root/runner.out" 2>&1 < /dev/null &
    echo $! > "$STATE_DIR/scheduler.pid"
    printf "launched pid=%s run=%s resume=%s\n" "$!" "$run_root" "${resume_arg[*]:-no}"
  '
}

scheduler_state() {
  local pod="$1"
  kubectl -n "$namespace" exec "$pod" -- env \
    PYTHON_BIN="$python_bin" PYTHON_FALLBACK="$python_fallback" STATE_DIR="$state_dir" bash -lc '
    if ! test -x "$PYTHON_BIN"; then PYTHON_BIN="$PYTHON_FALLBACK"; fi
    if ! test -s "$STATE_DIR/active_run"; then
      echo "waiting nodes=0"
      exit 0
    fi
    outer_root=$(cat "$STATE_DIR/active_run")
    if test -d "$outer_root/logs"; then
      run_root="$outer_root"
    else
      run_root=$(find "$outer_root" -mindepth 1 -maxdepth 1 -type d -name "*_petfinder_scheduler_profile_hkwd_a10_l40s" -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -n 1 | cut -d" " -f2-)
      run_root=${run_root:-$outer_root}
    fi
    nodes=$("$PYTHON_BIN" -c '\''import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(max(0,len(json.load(open(p)).get("nodes",[]))-1) if p.is_file() else 0)'\'' "$run_root/logs/journal.json")
    metrics_nodes=$("$PYTHON_BIN" -c '\''import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(json.load(open(p)).get("node_count",0) if p.is_file() else 0)'\'' "$run_root/logs/comparison_metrics.json")
    if test "$nodes" -ge 50 && test "$metrics_nodes" -eq 50; then
      echo "complete nodes=$nodes run=$run_root"
      exit 0
    fi
    if test -s "$STATE_DIR/scheduler.pid"; then
      pid=$(cat "$STATE_DIR/scheduler.pid")
      if kill -0 "$pid" 2>/dev/null && tr "\0" " " < "/proc/$pid/cmdline" | grep -q "run.py"; then
        echo "running nodes=$nodes pid=$pid run=$run_root"
        exit 0
      fi
    fi
    echo "stopped nodes=$nodes run=$run_root"
  '
}

while true; do
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  a10_pod=$(running_pod "$a10_selector")
  l40s_pod=$(running_pod "$l40s_selector")
  healthy=false
  if test -n "$l40s_pod" && agent_healthy "$l40s_pod"; then
    healthy=true
  fi
  printf '%s A10=%s L40S=%s agent_healthy=%s\n' "$now" "${a10_pod:-pending}" "${l40s_pod:-pending}" "$healthy"

  if test -n "$a10_pod" && test "$healthy" != true; then
    start_a10_filler "$a10_pod"
  fi
  if test -n "$l40s_pod" && test "$healthy" = true && test -z "$a10_pod"; then
    start_l40s_filler "$l40s_pod"
  fi

  if test -n "$a10_pod" && test "$healthy" = true; then
    stop_owned_filler "$a10_pod" a10
    stop_owned_filler "$l40s_pod" l40s
    if ! kubectl -n "$namespace" exec "$a10_pod" -- nvidia-smi --query-gpu=name --format=csv,noheader | grep -qx 'NVIDIA A10'; then
      echo "Refusing to launch: experiment pod is not on an NVIDIA A10." >&2
      exit 1
    fi
    launch_or_resume_scheduler "$a10_pod"
    state=$(scheduler_state "$a10_pod")
    printf '%s scheduler=%s\n' "$now" "$state"
    if [[ "$state" == complete\ * ]]; then
      exit 0
    fi
  fi
  sleep 30
done

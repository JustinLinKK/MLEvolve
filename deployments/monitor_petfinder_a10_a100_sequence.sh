#!/usr/bin/env bash

set -u

namespace="${NAMESPACE:-ecepxie}"
a10_pod="${A10_POD:-gpu-dev-a10-experiment-746c959459-2xwdd}"
a100_pod="${A100_POD:-}"
l40s_pod="${L40S_POD:-gpu-dev-l40s-1gpu-8599df9fd4-nqp2g}"
state_dir="${STATE_DIR:-/root/downeyflyfan/.cache/mlevolve_a100_a10_sequence_v5}"
controller_script="${CONTROLLER_SCRIPT:-/root/downeyflyfan/.cache/mlevolve_a10_a100_comparison_20260831/deployments/run_petfinder_a10_a100_sequence.sh}"
interval_seconds="${INTERVAL_SECONDS:-60}"
max_iterations="${MAX_ITERATIONS:-0}"
iteration=0

while true; do
    date -u +%Y-%m-%dT%H:%M:%SZ
    kubectl exec -n "$namespace" "$a10_pod" -- env STATE_DIR="$state_dir" CONTROLLER_SCRIPT="$controller_script" bash -lc '
        root=$(cat "$STATE_DIR/comparison_root" 2>/dev/null || true)
        phase=$(cat "$STATE_DIR/active_phase" 2>/dev/null || echo unknown)
        journal=$(find "$root/$phase" -type f -path "*/logs/journal.json" -printf "%T@ %p\n" 2>/dev/null | sort -nr | head -n 1 | cut -d" " -f2-)
        budget=0
        attempts=0
        if test -n "$journal"; then
            if test "$phase" = baseline; then
                repo=/root/downeyflyfan/.cache/mlevolve_a10_baseline_budgeted_20260831
            else
                repo=/root/downeyflyfan/.cache/mlevolve_a10_a100_comparison_20260831
            fi
            budget=$(cd "$repo" && /tmp/mlevolve-a10-run-venv/bin/python -c "import sys; from engine.node_accounting import count_budget_nodes_from_json; print(count_budget_nodes_from_json(sys.argv[1]))" "$journal" 2>/dev/null || echo 0)
            attempts=$(jq "[(.nodes // [])[] | select(.stage != \"root\")] | length" "$journal" 2>/dev/null || echo 0)
        fi
        pid=$(cat "$STATE_DIR/controller.pid" 2>/dev/null || echo 0)
        status=$(cat "$STATE_DIR/status" 2>/dev/null || true)
        if kill -0 "$pid" 2>/dev/null && \
           tr "\0" " " < "/proc/$pid/cmdline" 2>/dev/null | grep -Fq run_petfinder_a10_a100_sequence.sh; then
            controller=running
        else
            controller=stopped
            if test "$status" != complete && test -f "$CONTROLLER_SCRIPT"; then
                nohup env STATE_DIR="$STATE_DIR" bash "$CONTROLLER_SCRIPT" \
                    > "$STATE_DIR/controller.out" 2>&1 < /dev/null &
                pid=$!
                printf "%s\n" "$pid" > "$STATE_DIR/controller.pid"
                controller=restarted
            fi
        fi
        echo "phase=$phase budget_nodes=$budget attempts=$attempts controller=$controller root=$root"
        nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader
    '
    current_a100_pod="$a100_pod"
    if [[ -z "$current_a100_pod" ]]; then
        current_a100_pod="$(
            kubectl get pods -n "$namespace" -l app=mlevolve-a100-1gpu \
                --sort-by=.metadata.creationTimestamp \
                -o jsonpath='{.items[-1:].metadata.name}' 2>/dev/null || true
        )"
    fi
    if [[ -n "$current_a100_pod" ]]; then
        kubectl exec -n "$namespace" "$current_a100_pod" -- nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader
    else
        echo "A100 deployment pod not found"
    fi
    kubectl exec -n "$namespace" "$l40s_pod" -- nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader
    iteration=$((iteration + 1))
    if [[ "$max_iterations" -gt 0 ]] && [[ "$iteration" -ge "$max_iterations" ]]; then
        break
    fi
    sleep "$interval_seconds"
done

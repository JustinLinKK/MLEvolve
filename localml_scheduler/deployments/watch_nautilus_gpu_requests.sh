#!/usr/bin/env bash
# Persistently detect allocation of the team's Nautilus GPU requests.
set -euo pipefail

interval_seconds="${1:-60}"
state_dir="${NAUTILUS_GPU_MONITOR_DIR:-.cache/nautilus_gpu_monitor}"
namespace="ecepxie"
apps=(gpu-dev gpu-dev-l40s-1gpu gpu-dev-a10 gpu-dev-rtx4090 gpu-dev2)

mkdir -p "$state_dir"
state_file="$state_dir/state.tsv"
next_file="$state_dir/state.next.tsv"
events_file="$state_dir/events.log"

snapshot() {
    : > "$next_file"
    for app in "${apps[@]}"; do
        pods="$(kubectl get pods -n "$namespace" -l "app=$app" \
            -o jsonpath='{range .items[*]}{.metadata.name}{"|"}{.status.phase}{"|"}{.spec.nodeName}{"|"}{.status.containerStatuses[0].ready}{"\n"}{end}' \
            2>/dev/null || true)"
        if [[ -z "$pods" ]]; then
            printf '%s|NO_POD|||\n' "$app" >> "$next_file"
        else
            while IFS= read -r pod; do
                printf '%s|%s\n' "$app" "$pod" >> "$next_file"
            done <<< "$pods"
        fi
    done
    sort -o "$next_file" "$next_file"
}

snapshot
if [[ ! -f "$state_file" ]]; then
    mv "$next_file" "$state_file"
    printf '%s monitor initialized\n' "$(date -Is)" >> "$events_file"
fi

while true; do
    sleep "$interval_seconds"
    snapshot
    if ! cmp -s "$state_file" "$next_file"; then
        changes="$(diff -u "$state_file" "$next_file" || true)"
        {
            date -Is
            printf '%s\n' "$changes"
        } >> "$events_file"
        while IFS= read -r line; do
            [[ "$line" == +* && "$line" != +++* && "$line" == *'|Running|'* ]] || continue
            message="${line#+}"
            printf 'READY %s %s\n' "$(date -Is)" "$message" | tee -a "$events_file"
            if command -v notify-send >/dev/null 2>&1; then
                notify-send 'Nautilus GPU request ready' "$message" || true
            fi
        done <<< "$changes"
        mv "$next_file" "$state_file"
    else
        rm -f "$next_file"
    fi
done

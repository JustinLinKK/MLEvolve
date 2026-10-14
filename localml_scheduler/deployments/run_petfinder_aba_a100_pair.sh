#!/usr/bin/env bash
set -euo pipefail

repo="${REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
cache_root="${ABA_CACHE_ROOT:-/data1/yufan/MLEvolve/.cache}"
pair_root="${PAIR_ROOT:-$cache_root/experiments/petfinder_aba_a100_$(date -u +%Y%m%dT%H%M%SZ)}"
phase_launcher="$repo/localml_scheduler/deployments/run_petfinder_aba_a100_phase.sh"
qwen_launcher="$repo/localml_scheduler/deployments/run_qwen38_int8_aba_a100.sh"

test -x "$phase_launcher"
test -x "$qwen_launcher"
mkdir -p "$pair_root"
printf '%s\n' "$pair_root" > "$cache_root/petfinder_aba_a100_current_pair"

"$qwen_launcher"

for phase in baseline scheduler; do
  printf '%s\n' "$phase" > "$pair_root/active_phase"
  if ! PAIR_ROOT="$pair_root" "$phase_launcher" "$phase" \
      > "$pair_root/${phase}.controller.log" 2>&1; then
    printf 'failed:%s\n' "$phase" > "$pair_root/status"
    exit 1
  fi
done

printf 'complete\n' > "$pair_root/status"
printf '%s\n' "$pair_root"

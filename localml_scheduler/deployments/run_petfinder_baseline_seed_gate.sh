#!/usr/bin/env bash
# Run the recorded first scored baseline candidate as a reproducibility gate.
set -euo pipefail

source_file="${1:?pass the recorded candidate source path}"
run_root="${2:?pass the volatile run root}"
data_root="${PETFINDER_DATA_ROOT:-/dev/shm/mlevolve_petfinder_public_seed42}"
python_bin="${PETFINDER_PYTHON:-/dev/shm/mlevolve_a10_clean_venv/bin/python}"

test -f "$source_file"
test -d "$data_root"
test -x "$python_bin"

workspace="$run_root/workspace"
logs="$run_root/logs"
mkdir -p "$workspace" "$logs"
ln -sfn "$data_root" "$workspace/input"
cp "$source_file" "$workspace/candidate.py"

cd "$workspace"
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 "$python_bin" -u candidate.py \
  >"$logs/stdout.log" 2>"$logs/stderr.log"

score="$(rg -o 'Final Validation Score: [0-9.]+' "$logs/stdout.log" | tail -n 1 || true)"
printf '%s\n' "$score" >"$logs/final_score.txt"

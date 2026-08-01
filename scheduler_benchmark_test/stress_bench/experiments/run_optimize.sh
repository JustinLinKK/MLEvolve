#!/usr/bin/env bash
# Goal: show a saturation-gated placement policy reduces TOTAL TRAINING TIME (sum of per-job
# training seconds) by >=10% vs memory-fit packing, on the same 16 jobs with the same (accurate
# profile) predictor. Only difference = --placement-policy {pack|gated}.
set -uo pipefail
REPO=/root/downeyflyfan/perfseer_test/exp_run
BENCH=/root/downeyflyfan/perfseer_test/exp30
TRACE=$BENCH/short_matched.jsonl
PY=/usr/bin/python3
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
for POL in pack gated; do
  echo "=== policy=$POL $(date -Is) ==="
  rm -rf "$BENCH/opt_$POL" "/tmp/sbOpt_$POL"
  $PY -m scheduler_benchmark_test.stress_bench.run_bench \
    --condition scheduler_profile --trace "$TRACE" --outdir "$BENCH/opt_$POL" \
    --max-parallel 4 --vram-budget-gib 20 --timeout-s 3000 \
    --placement-policy "$POL" --runtime-root "/tmp/sbOpt_$POL" > "$BENCH/opt_$POL.log" 2>&1
  echo "exit=$? $(date -Is)"
  sleep 15
done
echo "=== ALL DONE $(date -Is) ==="

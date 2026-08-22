#!/usr/bin/env bash
# Fan a10_other_labels_job.yaml out into N single-GPU jobs over disjoint shards.
#
# The manifest holds one Job pinned to shards 0-3. This rewrites the name and the
# SHARD_START/SHARD_END pair for each slice, so the only thing that varies
# between jobs is which shard indices they walk. Everything else -- spec filter,
# output dir, profiler flags -- stays byte-identical, which is what makes the
# shard split coherent.
#
#   apply_a10_other_labels.sh            # apply all 4 slices
#   apply_a10_other_labels.sh 0          # apply only slice 0 (shards 0-3)
#   apply_a10_other_labels.sh 1 2 3      # apply slices 1..3
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
MANIFEST="$HERE/a10_other_labels_job.yaml"
NS=ecepxie
SHARDS_PER_JOB=4

slices=("$@")
[ ${#slices[@]} -eq 0 ] && slices=(0 1 2 3)

for s in "${slices[@]}"; do
  start=$((s * SHARDS_PER_JOB))
  end=$((start + SHARDS_PER_JOB - 1))
  sed -e "s/perfseer-a10-other-s0/perfseer-a10-other-s$s/" \
      -e "s/{name: SHARD_START, value: \"0\"}/{name: SHARD_START, value: \"$start\"}/" \
      -e "s/{name: SHARD_END, value: \"3\"}/{name: SHARD_END, value: \"$end\"}/" \
      "$MANIFEST" | kubectl apply -n "$NS" -f -
  echo "  slice $s -> shards $start-$end"
done

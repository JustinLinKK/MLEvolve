#!/usr/bin/env bash
# Start the ten-node Scheduler replay only after a fresh frozen-source baseline
# control agrees with the already preserved one-node Scheduler result.
set -euo pipefail

baseline_root="${1:?pass the fresh baseline-control output root}"
scheduler_gate_root="${2:?pass the completed one-node Scheduler gate root}"
sources="${3:?pass frozen source directory}"
ten_node_root="${4:?pass target ten-node output root}"
repo="${5:?pass MLEvolve execution repository}"
tolerance="${METRIC_TOLERANCE:-0.01}"
python_bin="${A10_PYTHON_BIN:-/dev/shm/mlevolve_a10_clean_venv/bin/python}"

test -r "$baseline_root/launcher.pid"
test -d "$scheduler_gate_root"
test -d "$sources"
test -f "$sources/manifest.json"
test -x "$python_bin"
test -f "$repo/localml_scheduler/deployments/run_petfinder_fixed_scheduler_replay_a10.sh"

baseline_pid="$(cat "$baseline_root/launcher.pid")"
while kill -0 "$baseline_pid" 2>/dev/null; do
  sleep 45
done

for _ in $(seq 1 24); do
  test -f "$baseline_root/logs/final_score.txt" && break
  sleep 5
done

"$python_bin" - "$baseline_root" "$scheduler_gate_root" "$tolerance" <<'PY'
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys

baseline_root = Path(sys.argv[1])
scheduler_root = Path(sys.argv[2])
tolerance = float(sys.argv[3])
score_path = baseline_root / "logs" / "final_score.txt"
summary_path = scheduler_root / "summary.json"
decision_path = baseline_root / "control_pair_decision.json"
decision: dict[str, object] = {
    "baseline_score_present": score_path.is_file(),
    "scheduler_summary_present": summary_path.is_file(),
    "tolerance": tolerance,
}

baseline_metric = None
if score_path.is_file():
    match = re.search(r"Final Validation Score:\s*([0-9.]+)", score_path.read_text())
    if match:
        baseline_metric = float(match.group(1))
decision["baseline_metric"] = baseline_metric

scheduler_metric = None
scheduler_status = None
if summary_path.is_file():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    job = ((summary.get("jobs") or [None])[0] or {})
    scheduler_status = job.get("status")
    scheduler_metric = (job.get("metadata") or {}).get("best_metric")
decision["scheduler_metric"] = scheduler_metric
decision["scheduler_status"] = scheduler_status

decision["absolute_metric_difference"] = (
    abs(float(baseline_metric) - float(scheduler_metric))
    if isinstance(baseline_metric, (int, float)) and isinstance(scheduler_metric, (int, float))
    else None
)
decision["accepted"] = bool(
    scheduler_status == "COMPLETED"
    and isinstance(baseline_metric, (int, float))
    and isinstance(scheduler_metric, (int, float))
    and math.isfinite(float(baseline_metric))
    and math.isfinite(float(scheduler_metric))
    and float(decision["absolute_metric_difference"]) <= tolerance
)
decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

accepted="$("$python_bin" - "$baseline_root/control_pair_decision.json" <<'PY'
import json
from pathlib import Path
import sys

print("1" if json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("accepted") else "0")
PY
)"
if test "$accepted" != 1; then
  exit 0
fi

test ! -e "$ten_node_root"
mkdir -p "$ten_node_root"
nohup env \
  PETFINDER_NODE_BUDGET=10 \
  PETFINDER_SEARCH_TIME_LIMIT=7200 \
  REPO="$repo" \
  bash "$repo/localml_scheduler/deployments/run_petfinder_fixed_scheduler_replay_a10.sh" \
  "$sources" "$ten_node_root" \
  >"$ten_node_root/launcher.log" 2>&1 < /dev/null &
echo "$!" > "$ten_node_root/launcher.pid"

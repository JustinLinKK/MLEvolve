#!/usr/bin/env bash
# Continue the frozen PetFinder Scheduler replay only after its one-node
# reproducibility gate agrees with the recorded historical metric.
set -euo pipefail

gate_root="${1:?pass one-node gate output root}"
sources="${2:?pass frozen source directory}"
ten_node_root="${3:?pass ten-node output root}"
repo="${4:?pass MLEvolve execution repository}"
expected_metric="${EXPECTED_METRIC:-18.1561}"
tolerance="${METRIC_TOLERANCE:-0.01}"
python_bin="${A10_PYTHON_BIN:-/dev/shm/mlevolve_a10_clean_venv/bin/python}"

test -r "$gate_root/launcher.pid"
test -d "$sources"
test -f "$sources/manifest.json"
test -x "$python_bin"
test -f "$repo/localml_scheduler/deployments/run_petfinder_fixed_scheduler_replay_a10.sh"

gate_pid="$(cat "$gate_root/launcher.pid")"
while kill -0 "$gate_pid" 2>/dev/null; do
  sleep 45
done

for _ in $(seq 1 24); do
  test -f "$gate_root/summary.json" && break
  sleep 5
done

"$python_bin" - "$gate_root" "$expected_metric" "$tolerance" <<'PY'
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

root = Path(sys.argv[1])
expected = float(sys.argv[2])
tolerance = float(sys.argv[3])
summary_path = root / "summary.json"
decision_path = root / "gate_decision.json"
decision: dict[str, object] = {
    "expected_metric": expected,
    "tolerance": tolerance,
    "summary_present": summary_path.is_file(),
}
if summary_path.is_file():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    job = ((summary.get("jobs") or [None])[0] or {})
    metric = (job.get("metadata") or {}).get("best_metric")
    decision.update({"status": job.get("status"), "observed_metric": metric})
    decision["accepted"] = (
        job.get("status") == "COMPLETED"
        and isinstance(metric, (int, float))
        and math.isfinite(metric)
        and abs(metric - expected) <= tolerance
    )
else:
    decision.update({"status": "missing_summary", "observed_metric": None, "accepted": False})
decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

accepted="$("$python_bin" - "$gate_root/gate_decision.json" <<'PY'
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

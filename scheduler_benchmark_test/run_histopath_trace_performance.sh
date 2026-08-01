#!/usr/bin/env bash
# Run the captured histopathologic-cancer-detection replay through both the
# scheduler and the simple multiprocess baseline, then print comparable metrics.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PRESET="quick"
OUTPUT_ROOT="$ROOT/runs/histopath_trace_performance_$(date +%Y%m%d_%H%M%S)"
SOURCE_RUN=""
FIXTURE_DIR=""
PARALLELISM="2"
GENERATE_PLOTS=0
BASELINE_JOB_TIMEOUT_SECONDS=""

usage() {
  cat <<'EOF'
Usage:
  bash scheduler_benchmark_test/run_histopath_trace_performance.sh [options]

Options:
  --preset quick|smoke|full          Preset passed to both replay wrappers. Defaults to quick.
  --source-run PATH                  Original profile_scheduler_compare run root.
  --fixture-dir PATH                 Fixture directory used by both replay wrappers.
  --output-root PATH                 Parent directory for scheduler/ and baseline/ outputs.
  --parallelism N                    Multiprocess baseline slots. Defaults to 2.
  --baseline-job-timeout-seconds N   Optional per-job timeout for the multiprocess baseline.
  --no-baseline-job-timeout          Do not pass a per-job timeout to the baseline. This is the default.
  --plots                            Generate comparison plot summaries.
  --skip-plots, --no-plots           Skip plot summaries. This is the default.
  -h, --help                         Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset)
      PRESET="${2:?--preset requires a value}"
      shift 2
      ;;
    --source-run)
      SOURCE_RUN="$(realpath -m "${2:?--source-run requires a value}")"
      shift 2
      ;;
    --fixture-dir)
      FIXTURE_DIR="$(realpath -m "${2:?--fixture-dir requires a value}")"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$(realpath -m "${2:?--output-root requires a value}")"
      shift 2
      ;;
    --parallelism)
      PARALLELISM="${2:?--parallelism requires a value}"
      shift 2
      ;;
    --baseline-job-timeout-seconds)
      BASELINE_JOB_TIMEOUT_SECONDS="${2:?--baseline-job-timeout-seconds requires a value}"
      shift 2
      ;;
    --no-baseline-job-timeout)
      BASELINE_JOB_TIMEOUT_SECONDS=""
      shift
      ;;
    --plots)
      GENERATE_PLOTS=1
      shift
      ;;
    --skip-plots|--no-plots)
      GENERATE_PLOTS=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCHEDULER_OUTPUT_ROOT="$OUTPUT_ROOT/scheduler"
BASELINE_OUTPUT_ROOT="$OUTPUT_ROOT/baseline"

common_args=(--preset "$PRESET")
if [[ -n "$SOURCE_RUN" ]]; then
  common_args+=(--source-run "$SOURCE_RUN")
fi
if [[ -n "$FIXTURE_DIR" ]]; then
  common_args+=(--fixture-dir "$FIXTURE_DIR")
fi
if [[ "$GENERATE_PLOTS" -eq 0 ]]; then
  common_args+=(--skip-plots)
fi

scheduler_cmd=(
  bash scheduler_benchmark_test/run_histopath_scheduler_replay.sh
  "${common_args[@]}"
  --output-root "$SCHEDULER_OUTPUT_ROOT"
)

baseline_cmd=(
  bash scheduler_benchmark_test/run_histopath_multiprocess_baseline.sh
  "${common_args[@]}"
  --output-root "$BASELINE_OUTPUT_ROOT"
  --parallelism "$PARALLELISM"
)
if [[ -n "$BASELINE_JOB_TIMEOUT_SECONDS" ]]; then
  baseline_cmd+=(-- --job-timeout-seconds "$BASELINE_JOB_TIMEOUT_SECONDS")
fi

echo "==> Running scheduler replay"
"${scheduler_cmd[@]}"

echo "==> Running multiprocess baseline"
"${baseline_cmd[@]}"

SCHEDULER_METRICS="$SCHEDULER_OUTPUT_ROOT/logs/comparison_metrics.json"
BASELINE_METRICS="$BASELINE_OUTPUT_ROOT/logs/comparison_metrics.json"

python - "$SCHEDULER_METRICS" "$BASELINE_METRICS" <<'PY'
from __future__ import annotations

from pathlib import Path
import json
import sys

scheduler_path = Path(sys.argv[1])
baseline_path = Path(sys.argv[2])


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


scheduler = load(scheduler_path)
baseline = load(baseline_path)
keys = [
    "total_wall_time_seconds",
    "candidate_execution_makespan_seconds",
    "total_candidate_execution_time_seconds",
    "completed_job_count",
    "failed_job_count",
    "candidate_execution_parallelism_ratio",
    "avg_gpu_util_percent",
    "peak_gpu_memory_used_mb",
]

print("\nTrace performance metrics:")
print(f"  scheduler metrics: {scheduler_path}")
print(f"  baseline metrics:  {baseline_path}")
for key in keys:
    print(f"  {key}: scheduler={fmt(scheduler.get(key))} baseline={fmt(baseline.get(key))}")

scheduler_wall = scheduler.get("total_wall_time_seconds")
baseline_wall = baseline.get("total_wall_time_seconds")
if isinstance(scheduler_wall, (int, float)) and isinstance(baseline_wall, (int, float)) and scheduler_wall > 0:
    print(f"  baseline/scheduler wall ratio: {baseline_wall / scheduler_wall:.3f}x")
PY

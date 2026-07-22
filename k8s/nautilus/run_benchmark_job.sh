#!/usr/bin/env bash
set -euo pipefail

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
HISTOPATH_DATA_ROOT="${HISTOPATH_DATA_ROOT:-/datasets/histopathologic-cancer-detection/prepared/public}"
STANDARD_BENCH_RESULTS_ROOT="${STANDARD_BENCH_RESULTS_ROOT:-/results/scheduler_benchmark_test/standard_histopath_v1}"
STANDARD_BENCH_FIXTURE="${STANDARD_BENCH_FIXTURE:-$ROOT/scheduler_benchmark_test/fixtures/standard_histopath_v1}"
STANDARD_BENCH_REPETITIONS="${STANDARD_BENCH_REPETITIONS:-3}"
STANDARD_BENCH_RUNNER_MODE="${STANDARD_BENCH_RUNNER_MODE:-real}"

mkdir -p "$STANDARD_BENCH_RESULTS_ROOT"
cd "$ROOT"
export HISTOPATH_DATA_ROOT
export PYTHONUNBUFFERED=1

"$PYTHON_BIN" -m scheduler_benchmark_test.standard.generate_fixture \
    --check \
    --output "$STANDARD_BENCH_FIXTURE" \
    --data-root "$HISTOPATH_DATA_ROOT"

if [ "${STANDARD_BENCH_VALIDATE_ONLY:-false}" = "true" ]; then
    exec "$PYTHON_BIN" -m scheduler_benchmark_test.standard.validate \
        --fixture "$STANDARD_BENCH_FIXTURE" \
        --data-root "$HISTOPATH_DATA_ROOT" \
        --output-root "$STANDARD_BENCH_RESULTS_ROOT/validation" \
        --resume
fi

RUN_ARGS=(
    -m scheduler_benchmark_test.standard.run_benchmark
    --fixture "$STANDARD_BENCH_FIXTURE"
    --data-root "$HISTOPATH_DATA_ROOT"
    --output-root "$STANDARD_BENCH_RESULTS_ROOT"
    --repetitions "$STANDARD_BENCH_REPETITIONS"
    --runner-mode "$STANDARD_BENCH_RUNNER_MODE"
    --resume
)

if [ -n "${STANDARD_BENCH_ARMS:-}" ]; then
    read -r -a configured_arms <<< "$STANDARD_BENCH_ARMS"
    RUN_ARGS+=(--arms "${configured_arms[@]}")
fi

exec "$PYTHON_BIN" "${RUN_ARGS[@]}"

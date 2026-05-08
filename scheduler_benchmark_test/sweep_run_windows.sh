#!/usr/bin/env bash
# No-MPS full sweep runner. Uses the windows/no-MPS matrix.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
export NO_MPS=1
export RESULTS_BASE="${RESULTS_BASE:-$REPO/results/scheduler_benchmark_test/main_sweep_windows}"

bash "$SCRIPT_DIR/sweep_run.sh" "$@"

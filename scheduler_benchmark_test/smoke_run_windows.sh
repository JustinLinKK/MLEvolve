#!/usr/bin/env bash
# No-MPS smoke runner. Useful on Windows-like or MPS-unavailable environments.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NO_MPS=1
export SMOKE_DIR="${SMOKE_DIR:-/tmp/scheduler_benchmark_smoke_windows}"

bash "$SCRIPT_DIR/smoke_run.sh" "$@"

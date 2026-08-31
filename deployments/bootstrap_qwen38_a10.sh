#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT="${DEPLOY_ROOT:-/root/qwen38-v100-int8}"
mkdir -p "$DEPLOY_ROOT"

install -m 644 /bootstrap/benchmark.py "$DEPLOY_ROOT/benchmark.py"
install -m 644 /bootstrap/a10_backend_runner.py "$DEPLOY_ROOT/a10_backend_runner.py"
install -m 755 /bootstrap/run_qwen38_a10_comparison.sh \
  "$DEPLOY_ROOT/run_qwen38_a10_comparison.sh"

exec env \
  BENCHMARK_PY="$DEPLOY_ROOT/benchmark.py" \
  RUNNER_PY="$DEPLOY_ROOT/a10_backend_runner.py" \
  "$DEPLOY_ROOT/run_qwen38_a10_comparison.sh"

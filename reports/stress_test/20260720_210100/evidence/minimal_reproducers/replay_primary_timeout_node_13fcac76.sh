#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="/workspaces/MLEvolve/runs/stress_workflow_fix20_pass/20260719_030703_stress_workflow_fix20_pass"
SCRIPT="${RUN_DIR}/workspace/runfile_0_13fcac766cae492480e5292d94ce021b_9143be55b70e49a88198002a0e4f3301.py"
WORKSPACE="${RUN_DIR}/workspace"

cd "${WORKSPACE}"
timeout --foreground --signal=TERM --kill-after=10s 120s python "${SCRIPT}"

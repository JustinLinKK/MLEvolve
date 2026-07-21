#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="/workspaces/MLEvolve/runs/stress_workflow_fix20_pass/20260719_030703_stress_workflow_fix20_pass"
SCRIPT="${RUN_DIR}/workspace/runfile_9_890c89f51f354cd8aed66d22bc5fcdfc_e4944714a34a41b192ef31fa1c0b4514.py"
WORKSPACE="${RUN_DIR}/workspace"

cd "${WORKSPACE}"
timeout --foreground --signal=TERM --kill-after=10s 120s python "${SCRIPT}"

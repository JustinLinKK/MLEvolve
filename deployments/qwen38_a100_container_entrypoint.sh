#!/usr/bin/env bash
set -euo pipefail

INIT_SCRIPT="${INIT_SCRIPT:-/init/init.sh}"
BOOTSTRAP_SCRIPT="${BOOTSTRAP_SCRIPT:-/opt/mlevolve-a100/bootstrap_qwen38_a100.sh}"
MONITOR_SCRIPT="${MONITOR_SCRIPT:-/opt/mlevolve-a100/monitor_qwen38_a100_vllm.sh}"

"$INIT_SCRIPT"
"$BOOTSTRAP_SCRIPT"
exec "$MONITOR_SCRIPT"

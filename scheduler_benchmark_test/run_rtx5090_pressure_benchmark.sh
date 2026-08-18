#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/artifacts/rtx5090-${RUN_STAMP}}"
MODE_TIMEOUT="${MODE_TIMEOUT:-5400}"
MODES="${MODES:-baseline,warm,cold}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUNNER_PID=""

mkdir -p "${OUTPUT_ROOT}"
printf '%s\n' "RTX 5090 benchmark artifacts: ${OUTPUT_ROOT}"

finalize() {
    local exit_code=$?
    trap - EXIT INT TERM
    if [[ -n "${RUNNER_PID}" ]] && kill -0 "${RUNNER_PID}" 2>/dev/null; then
        kill -TERM -- "-${RUNNER_PID}" 2>/dev/null || kill -TERM "${RUNNER_PID}" 2>/dev/null || true
        wait "${RUNNER_PID}" 2>/dev/null || true
    fi
    if [[ -f "${OUTPUT_ROOT}/trace.json" ]]; then
        (
            cd "${REPO_ROOT}"
            "${PYTHON_BIN}" -m scheduler_benchmark_test.rtx5090_pressure_benchmark analyze \
                --output-root "${OUTPUT_ROOT}" --deadline "${MODE_TIMEOUT}"
        ) || true
    fi
    printf '%s\n' "Final artifacts: ${OUTPUT_ROOT}"
    exit "${exit_code}"
}
trap finalize EXIT INT TERM

cd "${REPO_ROOT}"
setsid "${PYTHON_BIN}" -m scheduler_benchmark_test.rtx5090_pressure_benchmark full \
    --output-root "${OUTPUT_ROOT}" \
    --modes "${MODES}" \
    --mode-timeout "${MODE_TIMEOUT}" \
    "$@" &
RUNNER_PID=$!
wait "${RUNNER_PID}"
RUNNER_PID=""


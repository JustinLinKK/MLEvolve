#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE_TIMEOUT="${MODE_TIMEOUT:-900}"
MODES="${MODES:-baseline,warm,cold}"
RUNNER_PID=""

if [[ -z "${PRESSURE_RUN:-}" ]]; then
    PRESSURE_RUN="$(find "${SCRIPT_DIR}/artifacts" -mindepth 1 -maxdepth 1 -type d -name 'rtx5090-*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
fi
if [[ -n "${PRESSURE_RUN:-}" ]]; then
    OUTPUT_ROOT="${OUTPUT_ROOT:-${PRESSURE_RUN}/quality-audit}"
else
    RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
    OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/artifacts/model-quality-${RUN_STAMP}}"
fi

mkdir -p "${OUTPUT_ROOT}"
printf '%s\n' "Model-quality audit artifacts: ${OUTPUT_ROOT}"

finalize() {
    local exit_code=$?
    trap - EXIT INT TERM
    if [[ -n "${RUNNER_PID}" ]] && kill -0 "${RUNNER_PID}" 2>/dev/null; then
        kill -TERM -- "-${RUNNER_PID}" 2>/dev/null || kill -TERM "${RUNNER_PID}" 2>/dev/null || true
        wait "${RUNNER_PID}" 2>/dev/null || true
    fi
    if [[ -f "${OUTPUT_ROOT}/quality-trace.json" ]]; then
        (
            cd "${REPO_ROOT}"
            "${PYTHON_BIN}" -m scheduler_benchmark_test.model_quality_benchmark analyze \
                --output-root "${OUTPUT_ROOT}"
        ) || true
    fi
    printf '%s\n' "Quality chart: ${OUTPUT_ROOT}/quality_accuracy_comparison.png"
    printf '%s\n' "Quality report: ${OUTPUT_ROOT}/QUALITY_REPORT.md"
    exit "${exit_code}"
}
trap finalize EXIT INT TERM

cd "${REPO_ROOT}"
setsid "${PYTHON_BIN}" -m scheduler_benchmark_test.model_quality_benchmark run \
    --output-root "${OUTPUT_ROOT}" \
    --modes "${MODES}" \
    --mode-timeout "${MODE_TIMEOUT}" \
    "$@" &
RUNNER_PID=$!
wait "${RUNNER_PID}"
RUNNER_PID=""


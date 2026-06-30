#!/usr/bin/env bash
# Run one MLEvolve baseline measurement and write comparable performance artifacts.

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash measure_baseline_performance.sh <competition-id> [options] [-- extra.omegaconf=overrides]

Required:
  <competition-id>            Kaggle/MLE-bench competition slug.

Options:
  --dataset-root PATH         MLE-bench dataset root. Defaults to ./data/mle-bench.
  --run-root PATH             Output root. Defaults to runs/baseline_performance_<competition>_<timestamp>.
  --server-id N               Validation server id. Port is 5005 + N. Defaults to 111.
  --steps N                   Optional agent.steps override.
  --initial-drafts N          Optional agent.initial_drafts override.
  --seed N                    Optional shared agent.seed override.
  --timeout-seconds N         Run timeout. 0 disables timeout. Defaults to 0.
  --memory-index N            CUDA_VISIBLE_DEVICES value. Defaults to 0.
  --start-cpu-id N            Optional start_cpu_id override.
  --cpu-number N              Optional cpu_number override.
  --no-validation-server      Do not start engine.validation.format_server.
  --dry-run                   Print commands and write command.txt without running.
  -h, --help                  Show this help.

The baseline run is written under:
  <run-root>/baseline/
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

COMPETITION_ID=""
DATASET_ROOT="$ROOT/data/mle-bench"
RUN_ROOT=""
SERVER_ID=111
STEPS=""
INITIAL_DRAFTS=""
SEED=""
TIMEOUT_SECONDS=0
MEMORY_INDEX=0
START_CPU_ID=""
CPU_NUMBER=""
START_VALIDATION_SERVER=1
DRY_RUN=0
EXTRA_OVERRIDES=()

if [[ $# -gt 0 && "$1" != -* ]]; then
  COMPETITION_ID="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --competition|--competition-id|--contest)
      COMPETITION_ID="${2:?$1 requires a value}"
      shift 2
      ;;
    --dataset-root|--data-dir)
      DATASET_ROOT="${2:?$1 requires a value}"
      shift 2
      ;;
    --run-root)
      RUN_ROOT="${2:?--run-root requires a value}"
      shift 2
      ;;
    --server-id)
      SERVER_ID="${2:?--server-id requires a value}"
      shift 2
      ;;
    --steps)
      STEPS="${2:?--steps requires a value}"
      shift 2
      ;;
    --initial-drafts)
      INITIAL_DRAFTS="${2:?--initial-drafts requires a value}"
      shift 2
      ;;
    --seed)
      SEED="${2:?--seed requires a value}"
      shift 2
      ;;
    --timeout-seconds)
      TIMEOUT_SECONDS="${2:?--timeout-seconds requires a value}"
      shift 2
      ;;
    --memory-index)
      MEMORY_INDEX="${2:?--memory-index requires a value}"
      shift 2
      ;;
    --start-cpu-id)
      START_CPU_ID="${2:?--start-cpu-id requires a value}"
      shift 2
      ;;
    --cpu-number)
      CPU_NUMBER="${2:?--cpu-number requires a value}"
      shift 2
      ;;
    --no-validation-server)
      START_VALIDATION_SERVER=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_OVERRIDES=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$COMPETITION_ID" ]]; then
  echo "Missing competition id." >&2
  usage >&2
  exit 2
fi

DATASET_ROOT="$(realpath -m "$DATASET_ROOT")"
PUBLIC_DIR="$DATASET_ROOT/$COMPETITION_ID/prepared/public"
DESCRIPTION_FILE="$PUBLIC_DIR/description.md"

if [[ -z "$RUN_ROOT" ]]; then
  RUN_ROOT="$ROOT/runs/baseline_performance_${COMPETITION_ID}_$(date +%Y%m%d_%H%M%S)"
else
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
fi
mkdir -p "$RUN_ROOT"

quote_cmd() {
  printf '%q ' "$@"
}

GRADING_SERVER_PID=""
GRADING_SERVER_PORT=$((5005 + SERVER_ID))

cleanup() {
  if [[ -n "$GRADING_SERVER_PID" ]] && kill -0 "$GRADING_SERVER_PID" 2>/dev/null; then
    kill -TERM "$GRADING_SERVER_PID" 2>/dev/null || true
    wait "$GRADING_SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

wait_for_validation_server() {
  local waited=0
  local max_wait="${1:-30}"
  while [[ "$waited" -lt "$max_wait" ]]; do
    if python - <<PY >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:${GRADING_SERVER_PORT}/health", timeout=1).read()
PY
    then
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  return 1
}

start_validation_server() {
  if [[ "$START_VALIDATION_SERVER" -eq 0 ]]; then
    return 0
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Would start validation server on port $GRADING_SERVER_PORT for DATASET_DIR=$DATASET_ROOT"
    return 0
  fi

  local log_dir="$RUN_ROOT/validation_server"
  mkdir -p "$log_dir"
  export DATASET_DIR="$DATASET_ROOT"
  export GRADING_SERVER_PORT

  echo "==> Starting validation server on port $GRADING_SERVER_PORT"
  python -u -m engine.validation.format_server \
    dataset_dir="$DATASET_ROOT" \
    data_dir="none" \
    desc_file="none" \
    > "$log_dir/server_${SERVER_ID}.log" 2>&1 &
  GRADING_SERVER_PID="$!"

  if ! wait_for_validation_server "${GRADING_SERVER_WAIT_SECONDS:-30}"; then
    echo "Validation server did not become healthy. See $log_dir/server_${SERVER_ID}.log" >&2
    exit 1
  fi
}

run_baseline() {
  local mode_root="$RUN_ROOT/baseline"
  local stdout_log="$mode_root/stdout.log"
  local exit_file="$mode_root/exit_code.txt"
  mkdir -p "$mode_root"

  local -a cmd=(
    python "$ROOT/run.py"
    "exp_id=$COMPETITION_ID"
    "dataset_dir=$DATASET_ROOT"
    "data_dir=$PUBLIC_DIR"
    "desc_file=$DESCRIPTION_FILE"
    "exp_name=${COMPETITION_ID}_baseline"
    "log_dir=$mode_root/runs"
    "workspace_dir=$mode_root/runs"
  )

  if [[ -n "$STEPS" ]]; then
    cmd+=("agent.steps=$STEPS")
  fi
  if [[ -n "$INITIAL_DRAFTS" ]]; then
    cmd+=("agent.initial_drafts=$INITIAL_DRAFTS")
  fi
  if [[ -n "$SEED" ]]; then
    cmd+=("agent.seed=$SEED")
  fi
  if [[ -n "$START_CPU_ID" ]]; then
    cmd+=("start_cpu_id=$START_CPU_ID")
  fi
  if [[ -n "$CPU_NUMBER" ]]; then
    cmd+=("cpu_number=$CPU_NUMBER")
  fi

  cmd+=("${EXTRA_OVERRIDES[@]}")

  {
    echo "mode: baseline"
    echo "started_at: $(date -Iseconds)"
    echo -n "command: "
    quote_cmd "${cmd[@]}"
    echo
  } > "$mode_root/command.txt"

  echo
  echo "==> Running baseline"
  echo "    output: $stdout_log"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    quote_cmd "${cmd[@]}"
    echo
    echo "dry-run" > "$exit_file"
    return 0
  fi

  set +e
  if [[ "$TIMEOUT_SECONDS" -gt 0 ]]; then
    CUDA_VISIBLE_DEVICES="$MEMORY_INDEX" timeout --foreground --signal=TERM --kill-after=10s "${TIMEOUT_SECONDS}s" "${cmd[@]}" >"$stdout_log" 2>&1
  else
    CUDA_VISIBLE_DEVICES="$MEMORY_INDEX" "${cmd[@]}" >"$stdout_log" 2>&1
  fi
  local exit_code=$?
  set -e

  echo "$exit_code" > "$exit_file"
  echo "    exit_code: $exit_code"
  return "$exit_code"
}

{
  echo "created_at: $(date -Iseconds)"
  echo "competition_id: $COMPETITION_ID"
  echo "dataset_root: $DATASET_ROOT"
  echo "public_dir: $PUBLIC_DIR"
  echo "description_file: $DESCRIPTION_FILE"
  echo "steps: ${STEPS:-<config default>}"
  echo "initial_drafts: ${INITIAL_DRAFTS:-<config default>}"
  echo "seed: ${SEED:-<config default>}"
  echo "start_cpu_id: ${START_CPU_ID:-<config default>}"
  echo "cpu_number: ${CPU_NUMBER:-<config default>}"
  echo "timeout_seconds: $TIMEOUT_SECONDS"
  echo "validation_server_port: $GRADING_SERVER_PORT"
  echo "extra_overrides: ${EXTRA_OVERRIDES[*]:-}"
  echo "run_root: $RUN_ROOT"
} > "$RUN_ROOT/manifest.txt"

if [[ "$DRY_RUN" -eq 0 ]]; then
  if [[ ! -d "$PUBLIC_DIR" ]]; then
    echo "Prepared public directory does not exist: $PUBLIC_DIR" >&2
    exit 1
  fi
  if [[ ! -f "$DESCRIPTION_FILE" ]]; then
    echo "Prepared description file does not exist: $DESCRIPTION_FILE" >&2
    exit 1
  fi
fi

start_validation_server
run_baseline

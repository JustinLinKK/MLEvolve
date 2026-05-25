#!/usr/bin/env bash
# Prepare one Kaggle/MLE-bench competition, then run MLEvolve twice:
# once in baseline mode and once in hardware-aware mode.
#
# Example:
#   bash compare_hardware_awareness.sh dogs-vs-cats-redux-kernels-edition \
#     --dataset-root /mle-bench/data \
#     --steps 5

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash compare_hardware_awareness.sh <competition-id> [options] [-- extra.omegaconf=overrides]

The script follows the README boot flow:
  1. prepare an MLE-bench competition dataset with Kaggle credentials;
  2. run the same prepared competition through MLEvolve in baseline mode;
  3. run it again in hardware_aware mode.

Required:
  <competition-id>            Kaggle/MLE-bench competition slug, e.g. dogs-vs-cats-redux-kernels-edition.

Options:
  --dataset-root PATH         MLE-bench dataset root. Defaults to ./data/mle-bench.
  --kaggle-json PATH          Kaggle credentials. Defaults to ./kaggle.json, then ~/.kaggle/kaggle.json.
  --config PATH               MLEvolve config. Defaults to config.yaml, then config.example.yaml.
  --run-root PATH             Comparison output root. Defaults to runs/hardware_awareness_compare_<competition>_<timestamp>.
  --server-id N               Validation server id. Port is 5005 + N. Defaults to 111.
  --steps N                   Agent steps for each mode. Defaults to 1.
  --initial-drafts N          Initial drafts for each mode. Defaults to 0.
  --seed N                    Shared seed for both modes. Defaults to 42.
  --timeout-seconds N         Per-mode timeout. 0 disables timeout. Defaults to 0.
  --memory-index N            CUDA_VISIBLE_DEVICES value. Defaults to 0.
  --start-cpu-id N            start_cpu_id override. Defaults to 0.
  --cpu-number N              cpu_number override. Defaults to 21.
  --skip-prepare              Reuse an existing prepared dataset.
  --keep-raw                  Keep raw MLE-bench download files after prepare.
  --verify-checksums          Let mlebench verify checksums. Default skips verification.
  --hardware-first            Run hardware_aware before baseline.
  --disable-scheduler-baseline
                              Disable scheduler execution for baseline. By default both modes use scheduler rounds.
  --no-validation-server      Do not start engine.validation.format_server.
  --plot-output-dir PATH      Graph output directory. Defaults to <run-root>/comparison_plots.
  --skip-plots, --no-plots    Do not generate comparison graph images after both runs.
  --dry-run                   Print commands without preparing or running.
  -h, --help                  Show this help.

Extra overrides after -- are appended to both run.py invocations exactly as provided.
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

COMPETITION_ID=""
DATASET_ROOT="$ROOT/data/mle-bench"
KAGGLE_JSON=""
CONFIG_PATH=""
RUN_ROOT=""
SERVER_ID=111
STEPS=1
INITIAL_DRAFTS=0
SEED=42
TIMEOUT_SECONDS=0
MEMORY_INDEX=0
START_CPU_ID=0
CPU_NUMBER=21
SKIP_PREPARE=0
KEEP_RAW=0
VERIFY_CHECKSUMS=0
HARDWARE_FIRST=0
DISABLE_SCHEDULER_BASELINE=0
START_VALIDATION_SERVER=1
GENERATE_PLOTS=1
PLOT_OUTPUT_DIR=""
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
    --kaggle-json)
      KAGGLE_JSON="${2:?--kaggle-json requires a value}"
      shift 2
      ;;
    --config)
      CONFIG_PATH="${2:?--config requires a value}"
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
    --skip-prepare)
      SKIP_PREPARE=1
      shift
      ;;
    --keep-raw)
      KEEP_RAW=1
      shift
      ;;
    --verify-checksums)
      VERIFY_CHECKSUMS=1
      shift
      ;;
    --hardware-first)
      HARDWARE_FIRST=1
      shift
      ;;
    --keep-scheduler-baseline)
      # Backward-compatible no-op: baseline now keeps the scheduler by default.
      DISABLE_SCHEDULER_BASELINE=0
      shift
      ;;
    --disable-scheduler-baseline)
      DISABLE_SCHEDULER_BASELINE=1
      shift
      ;;
    --no-validation-server)
      START_VALIDATION_SERVER=0
      shift
      ;;
    --plot-output-dir)
      PLOT_OUTPUT_DIR="${2:?--plot-output-dir requires a value}"
      shift 2
      ;;
    --skip-plots|--no-plots)
      GENERATE_PLOTS=0
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

if [[ -z "$CONFIG_PATH" ]]; then
  if [[ -f "$ROOT/config.yaml" ]]; then
    CONFIG_PATH="$ROOT/config.yaml"
  else
    CONFIG_PATH="$ROOT/config.example.yaml"
  fi
else
  CONFIG_PATH="$(realpath "$CONFIG_PATH")"
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file does not exist: $CONFIG_PATH" >&2
  exit 2
fi
export MLEVOLVE_CONFIG="$CONFIG_PATH"

if [[ -z "$KAGGLE_JSON" ]]; then
  if [[ -f "$ROOT/kaggle.json" ]]; then
    KAGGLE_JSON="$ROOT/kaggle.json"
  elif [[ -f "$HOME/.kaggle/kaggle.json" ]]; then
    KAGGLE_JSON="$HOME/.kaggle/kaggle.json"
  fi
else
  KAGGLE_JSON="$(realpath "$KAGGLE_JSON")"
fi

if [[ -n "$KAGGLE_JSON" ]]; then
  if [[ ! -f "$KAGGLE_JSON" ]]; then
    echo "Kaggle credentials file does not exist: $KAGGLE_JSON" >&2
    exit 2
  fi
  chmod 600 "$KAGGLE_JSON" 2>/dev/null || true
  export KAGGLE_CONFIG_DIR="$(dirname "$KAGGLE_JSON")"
elif [[ "$SKIP_PREPARE" -eq 0 ]]; then
  echo "No kaggle.json found. Pass --kaggle-json PATH or set up ~/.kaggle/kaggle.json." >&2
  exit 2
fi

if [[ -z "$RUN_ROOT" ]]; then
  RUN_ROOT="$ROOT/runs/hardware_awareness_compare_${COMPETITION_ID}_$(date +%Y%m%d_%H%M%S)"
else
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
fi
mkdir -p "$RUN_ROOT"

if [[ -z "$PLOT_OUTPUT_DIR" ]]; then
  PLOT_OUTPUT_DIR="$RUN_ROOT/comparison_plots"
else
  PLOT_OUTPUT_DIR="$(realpath -m "$PLOT_OUTPUT_DIR")"
fi

quote_cmd() {
  printf '%q ' "$@"
}

prepare_dataset() {
  if [[ "$SKIP_PREPARE" -eq 1 ]]; then
    echo "==> Skipping dataset prepare; expecting $PUBLIC_DIR"
    return 0
  fi

  local -a prepare_cmd=(mlebench prepare -c "$COMPETITION_ID" --data-dir "$DATASET_ROOT")
  if [[ "$KEEP_RAW" -eq 1 ]]; then
    prepare_cmd+=(--keep-raw)
  fi
  if [[ "$VERIFY_CHECKSUMS" -eq 0 ]]; then
    prepare_cmd+=(--skip-verification)
  fi

  echo "==> Preparing Kaggle competition through mlebench"
  echo "    competition: $COMPETITION_ID"
  echo "    dataset_root: $DATASET_ROOT"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    quote_cmd "${prepare_cmd[@]}"
    echo
    return 0
  fi

  mkdir -p "$DATASET_ROOT"
  "${prepare_cmd[@]}"
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

run_mode() {
  local mode="$1"
  local mode_root="$RUN_ROOT/$mode"
  local stdout_log="$mode_root/stdout.log"
  local exit_file="$mode_root/exit_code.txt"
  local scheduler_runtime="$mode_root/scheduler_runtime"
  mkdir -p "$mode_root"

  local -a cmd=(
    python "$ROOT/run.py"
    "exp_id=$COMPETITION_ID"
    "dataset_dir=$DATASET_ROOT"
    "data_dir=$PUBLIC_DIR"
    "desc_file=$DESCRIPTION_FILE"
    "exp_name=${COMPETITION_ID}_${mode}"
    "log_dir=$mode_root/runs"
    "workspace_dir=$mode_root/runs"
    "experiment.mode=$mode"
    "agent.steps=$STEPS"
    "agent.initial_drafts=$INITIAL_DRAFTS"
    "agent.seed=$SEED"
    "start_cpu_id=$START_CPU_ID"
    "cpu_number=$CPU_NUMBER"
    "scheduler.runtime_root=$scheduler_runtime"
  )

  if [[ "$mode" == "baseline" && "$DISABLE_SCHEDULER_BASELINE" -eq 1 ]]; then
    cmd+=("scheduler.enabled=false")
  fi

  cmd+=("${EXTRA_OVERRIDES[@]}")

  {
    echo "mode: $mode"
    echo "started_at: $(date -Iseconds)"
    echo -n "command: "
    quote_cmd "${cmd[@]}"
    echo
  } > "$mode_root/command.txt"

  echo
  echo "==> Running $mode"
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

generate_plots() {
  if [[ "$GENERATE_PLOTS" -eq 0 ]]; then
    return 0
  fi

  local -a plot_cmd=(
    python "$ROOT/utils/plot_hardware_awareness_comparison.py"
    --run-root "$RUN_ROOT"
    --output-dir "$PLOT_OUTPUT_DIR"
  )

  echo
  echo "==> Generating comparison graphs"
  echo "    output: $PLOT_OUTPUT_DIR"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    quote_cmd "${plot_cmd[@]}"
    echo
    return 0
  fi

  if ! "${plot_cmd[@]}"; then
    echo "Warning: comparison graph generation failed." >&2
  fi
}

MANIFEST="$RUN_ROOT/manifest.txt"
{
  echo "created_at: $(date -Iseconds)"
  echo "competition_id: $COMPETITION_ID"
  echo "dataset_root: $DATASET_ROOT"
  echo "public_dir: $PUBLIC_DIR"
  echo "description_file: $DESCRIPTION_FILE"
  echo "kaggle_config_dir: ${KAGGLE_CONFIG_DIR:-}"
  echo "config: $CONFIG_PATH"
  echo "steps: $STEPS"
  echo "initial_drafts: $INITIAL_DRAFTS"
  echo "seed: $SEED"
  echo "timeout_seconds: $TIMEOUT_SECONDS"
  echo "validation_server_port: $GRADING_SERVER_PORT"
  echo "extra_overrides: ${EXTRA_OVERRIDES[*]:-}"
  echo "run_root: $RUN_ROOT"
  echo "plots_enabled: $GENERATE_PLOTS"
  echo "plot_output_dir: $PLOT_OUTPUT_DIR"
} > "$MANIFEST"

prepare_dataset

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

MODE_ORDER=("baseline" "hardware_aware")
if [[ "$HARDWARE_FIRST" -eq 1 ]]; then
  MODE_ORDER=("hardware_aware" "baseline")
fi

OVERALL_EXIT=0
for mode in "${MODE_ORDER[@]}"; do
  if ! run_mode "$mode"; then
    OVERALL_EXIT=1
  fi
done

generate_plots

{
  echo
  echo "results:"
  for mode in baseline hardware_aware; do
    if [[ -f "$RUN_ROOT/$mode/exit_code.txt" ]]; then
      echo "  $mode: $(cat "$RUN_ROOT/$mode/exit_code.txt")"
    else
      echo "  $mode: not-run"
    fi
  done
  if [[ "$GENERATE_PLOTS" -eq 1 ]]; then
    echo "  plots: $PLOT_OUTPUT_DIR"
  fi
} >> "$MANIFEST"

echo
echo "Comparison written to: $RUN_ROOT"
echo "Manifest: $MANIFEST"
if [[ "$GENERATE_PLOTS" -eq 1 ]]; then
  echo "Graphs: $PLOT_OUTPUT_DIR"
fi
exit "$OVERALL_EXIT"

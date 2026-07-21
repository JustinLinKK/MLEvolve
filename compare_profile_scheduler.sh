#!/usr/bin/env bash
# Compare the current branch profile-based scheduler path against the same
# hardware-aware run with scheduler execution disabled.
#
# Example:
#   bash compare_profile_scheduler.sh histopathologic-cancer-detection \
#     --dataset-root "$PWD/data/mle-bench" \
#     --config "$PWD/config.yaml" \
#     --skip-prepare \
#     --steps 30 \
#     --initial-drafts 3 \
#     --seed 42 \
#     --agent-time-limit 10800 \
#     --timeout-seconds 10800 \
#     --memory-index 0

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash compare_profile_scheduler.sh <competition-id> [options] [-- extra.omegaconf=overrides]

Runs the same current-branch hardware-aware task twice:
  scheduler_off: experiment.mode=hardware_aware with scheduler.enabled=false
  scheduler_on:  experiment.mode=hardware_aware with scheduler.enabled=true

Required:
  <competition-id>            Kaggle/MLE-bench competition slug.

Options:
  --dataset-root PATH         MLE-bench dataset root. Defaults to ./data/mle-bench.
  --kaggle-json PATH          Kaggle credentials. Defaults to ./kaggle.json, then ~/.kaggle/kaggle.json.
  --config PATH               MLEvolve config. Defaults to config.yaml, then config.example.yaml.
  --run-root PATH             Output root. Defaults to runs/profile_scheduler_compare_<competition>_<timestamp>.
  --server-id N               Validation server id. Port is 5005 + N. Defaults to 111.
  --steps N                   agent.steps override. Defaults to 30.
  --initial-drafts N          agent.initial_drafts override. Defaults to 3.
  --seed N                    agent.seed override. Defaults to 42.
  --agent-time-limit N        agent.time_limit override in seconds. Defaults to 10800.
  --timeout-seconds N         Per-mode shell timeout. 0 disables timeout. Defaults to 10800.
  --memory-index N            CUDA_VISIBLE_DEVICES value. Defaults to 0.
  --skip-prepare              Reuse an existing prepared dataset.
  --keep-raw                  Keep raw MLE-bench download files after prepare.
  --verify-checksums          Let mlebench verify checksums. Default skips verification.
  --no-validation-server      Do not start engine.validation.format_server.
  --plot-output-dir PATH      Graph output directory. Defaults to <run-root>/comparison_plots.
  --skip-plots, --no-plots    Do not generate comparison graph images after both runs.
  --scheduler-on-only         Run only scheduler_on with hardware knowledge enabled.
  --disable-max-packing-limit Set max_packed_jobs_per_gpu=0 and use VRAM-targeted auto-pack.
  --dry-run                   Print commands without preparing or running.
  -h, --help                  Show this help.

Extra overrides after -- are appended to every run.py invocation exactly as provided.
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
STEPS=30
INITIAL_DRAFTS=3
SEED=42
AGENT_TIME_LIMIT=10800
TIMEOUT_SECONDS=10800
MEMORY_INDEX=0
SKIP_PREPARE=0
KEEP_RAW=0
VERIFY_CHECKSUMS=0
START_VALIDATION_SERVER=1
GENERATE_PLOTS=1
PLOT_OUTPUT_DIR=""
SCHEDULER_ON_ONLY=0
DISABLE_MAX_PACKING_LIMIT=0
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
    --agent-time-limit)
      AGENT_TIME_LIMIT="${2:?--agent-time-limit requires a value}"
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
    --scheduler-on-only|--only-scheduler-on)
      SCHEDULER_ON_ONLY=1
      shift
      ;;
    --disable-max-packing-limit|--no-max-packing-limit)
      DISABLE_MAX_PACKING_LIMIT=1
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

dataset_is_prepared() {
  [[ -d "$PUBLIC_DIR" && -f "$DESCRIPTION_FILE" ]]
}

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
elif [[ "$SKIP_PREPARE" -eq 0 ]] && ! dataset_is_prepared; then
  echo "No kaggle.json found. Pass --kaggle-json PATH or set up ~/.kaggle/kaggle.json." >&2
  exit 2
fi

if [[ -z "$RUN_ROOT" ]]; then
  RUN_ROOT="$ROOT/runs/profile_scheduler_compare_${COMPETITION_ID}_$(date +%Y%m%d_%H%M%S)"
else
  RUN_ROOT="$(realpath -m "$RUN_ROOT")"
fi
mkdir -p "$RUN_ROOT"

if [[ -z "$PLOT_OUTPUT_DIR" ]]; then
  PLOT_OUTPUT_DIR="$RUN_ROOT/comparison_plots"
else
  PLOT_OUTPUT_DIR="$(realpath -m "$PLOT_OUTPUT_DIR")"
fi
EVIDENCE_DIR="$RUN_ROOT/evidence"
MODE_LABELS=()
if [[ "$SCHEDULER_ON_ONLY" -eq 0 ]]; then
  MODE_LABELS+=(scheduler_off)
fi
MODE_LABELS+=(scheduler_on)

quote_cmd() {
  printf '%q ' "$@"
}

collect_evidence() {
  local phase="$1"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Would collect profile scheduler evidence ($phase) into $EVIDENCE_DIR"
    return 0
  fi
  python "$ROOT/utils/collect_profile_scheduler_evidence.py" \
    --run-root "$RUN_ROOT" \
    --output-dir "$EVIDENCE_DIR" \
    --config "$CONFIG_PATH" \
    --phase "$phase" || echo "Warning: evidence collection failed for phase $phase." >&2
}

prepare_dataset() {
  if dataset_is_prepared; then
    echo "==> Prepared dataset already exists; skipping mlebench prepare"
    echo "    public_dir: $PUBLIC_DIR"
    return 0
  fi

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
CLEANED_UP=0

signal_scheduler_workers_for_run_root() {
  local sig="$1"
  local pid pgid cmd
  local found=1
  while read -r pid pgid cmd; do
    [[ -n "${pid:-}" ]] || continue
    [[ "${cmd:-}" == *"localml_scheduler.execution.worker_entry --runtime-root ${RUN_ROOT}/"* ]] || continue
    found=0
    if [[ -n "${pgid:-}" ]]; then
      kill -s "$sig" -- "-$pgid" 2>/dev/null || kill -s "$sig" -- "$pid" 2>/dev/null || true
    else
      kill -s "$sig" -- "$pid" 2>/dev/null || true
    fi
  done < <(ps -eo pid=,pgid=,args=)
  return "$found"
}

cleanup() {
  local status=$?
  if [[ "$CLEANED_UP" -eq 1 ]]; then
    return "$status"
  fi
  CLEANED_UP=1
  if signal_scheduler_workers_for_run_root TERM; then
    sleep 1
    signal_scheduler_workers_for_run_root KILL || true
  fi
  if [[ -n "$GRADING_SERVER_PID" ]] && kill -0 "$GRADING_SERVER_PID" 2>/dev/null; then
    kill -TERM "$GRADING_SERVER_PID" 2>/dev/null || true
    wait "$GRADING_SERVER_PID" 2>/dev/null || true
  fi
  return "$status"
}

on_interrupt() {
  exit 130
}

on_terminate() {
  exit 143
}

trap cleanup EXIT
trap on_interrupt INT
trap on_terminate TERM

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
  local label="$1"
  local scheduler_enabled="$2"
  local mode_root="$RUN_ROOT/$label"
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
    "exp_name=${COMPETITION_ID}_${label}"
    "log_dir=$mode_root/runs"
    "workspace_dir=$mode_root/runs"
    "experiment.mode=hardware_aware"
    "hardware_knowledge.enabled=true"
    "hardware_knowledge.include_profile_evidence=true"
    "scheduler.enabled=$scheduler_enabled"
    "agent.steps=$STEPS"
    "agent.initial_drafts=$INITIAL_DRAFTS"
    "agent.seed=$SEED"
    "agent.time_limit=$AGENT_TIME_LIMIT"
  )

  if [[ "$scheduler_enabled" == "true" ]]; then
    cmd+=("scheduler.runtime_root=$scheduler_runtime")
    if [[ "$DISABLE_MAX_PACKING_LIMIT" -eq 1 ]]; then
      cmd+=(
        "scheduler.settings.gpu_scheduler.mode=parallel_auto_pack"
        "scheduler.settings.gpu_scheduler.max_packed_jobs_per_gpu=0"
        "scheduler.settings.gpu_scheduler.auto_pack.target_metric=vram"
      )
    else
      cmd+=(
        "scheduler.settings.gpu_scheduler.candidate_window_size=2"
        "scheduler.settings.gpu_scheduler.max_packed_jobs_per_gpu=2"
      )
    fi
    cmd+=(
      "scheduler.settings.gpu_scheduler.batch_probe_max_search_rounds=4"
      "scheduler.settings.gpu_scheduler.profiling.warmup_steps=5"
      "scheduler.settings.gpu_scheduler.profiling.solo_probe_steps=10"
      "scheduler.settings.gpu_scheduler.parallel_optimizer.power_of_two_range_up=2"
      "scheduler.settings.gpu_scheduler.submission_defaults.batch_probe_max_multiplier=4"
      "scheduler.settings.gpu_scheduler.submission_defaults.batch_probe_probe_timeout_seconds=20"
    )
  fi

  cmd+=("${EXTRA_OVERRIDES[@]}")

  {
    echo "label: $label"
    echo "experiment_mode: hardware_aware"
    echo "hardware_knowledge_enabled: true"
    echo "hardware_knowledge_include_profile_evidence: true"
    echo "scheduler_enabled: $scheduler_enabled"
    echo "config: $CONFIG_PATH"
    echo "started_at: $(date -Iseconds)"
    echo "environment: MLEVOLVE_COMMAND_LABEL=$label MLEVOLVE_CONFIG=$CONFIG_PATH CUDA_VISIBLE_DEVICES=$MEMORY_INDEX"
    echo -n "command: "
    quote_cmd "${cmd[@]}"
    echo
  } > "$mode_root/command.txt"

  echo
  echo "==> Running $label"
  echo "    scheduler_enabled: $scheduler_enabled"
  echo "    output: $stdout_log"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    MLEVOLVE_COMMAND_LABEL="$label" quote_cmd "${cmd[@]}"
    echo
    echo "dry-run" > "$exit_file"
    return 0
  fi

  set +e
  if [[ "$TIMEOUT_SECONDS" -gt 0 ]]; then
    MLEVOLVE_COMMAND_LABEL="$label" CUDA_VISIBLE_DEVICES="$MEMORY_INDEX" timeout --foreground --signal=TERM --kill-after=10s "${TIMEOUT_SECONDS}s" "${cmd[@]}" >"$stdout_log" 2>&1
  else
    MLEVOLVE_COMMAND_LABEL="$label" CUDA_VISIBLE_DEVICES="$MEMORY_INDEX" "${cmd[@]}" >"$stdout_log" 2>&1
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
    --modes "${MODE_LABELS[@]}"
  )

  echo
  echo "==> Generating scheduler comparison graphs"
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

run_and_record_mode() {
  local label="$1"
  local scheduler_enabled="$2"
  local mode_exit=0
  if run_mode "$label" "$scheduler_enabled"; then
    return 0
  else
    mode_exit=$?
  fi
  if [[ "$mode_exit" -eq 130 || "$mode_exit" -eq 143 ]]; then
    exit "$mode_exit"
  fi
  OVERALL_EXIT=1
  return 0
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
  echo "agent_time_limit: $AGENT_TIME_LIMIT"
  echo "timeout_seconds: $TIMEOUT_SECONDS"
  echo "memory_index: $MEMORY_INDEX"
  echo "scheduler_on_only: $SCHEDULER_ON_ONLY"
  echo "disable_max_packing_limit: $DISABLE_MAX_PACKING_LIMIT"
  echo "validation_server_port: $GRADING_SERVER_PORT"
  echo "extra_overrides: ${EXTRA_OVERRIDES[*]:-}"
  echo "run_root: $RUN_ROOT"
  echo "plots_enabled: $GENERATE_PLOTS"
  echo "plot_output_dir: $PLOT_OUTPUT_DIR"
} > "$MANIFEST"

collect_evidence preflight

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

OVERALL_EXIT=0
if [[ "$SCHEDULER_ON_ONLY" -eq 0 ]]; then
  run_and_record_mode scheduler_off false
fi
run_and_record_mode scheduler_on true

generate_plots

{
  echo
  echo "results:"
  for label in "${MODE_LABELS[@]}"; do
    if [[ -f "$RUN_ROOT/$label/exit_code.txt" ]]; then
      echo "  $label: $(cat "$RUN_ROOT/$label/exit_code.txt")"
    else
      echo "  $label: not-run"
    fi
  done
  if [[ "$GENERATE_PLOTS" -eq 1 ]]; then
    echo "  plots: $PLOT_OUTPUT_DIR"
  fi
  echo "  evidence: $EVIDENCE_DIR"
} >> "$MANIFEST"

collect_evidence postrun

echo
echo "Comparison written to: $RUN_ROOT"
echo "Manifest: $MANIFEST"
if [[ "$GENERATE_PLOTS" -eq 1 ]]; then
  echo "Graphs: $PLOT_OUTPUT_DIR"
fi
echo "Evidence: $EVIDENCE_DIR"
exit "$OVERALL_EXIT"

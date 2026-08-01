#!/usr/bin/env bash
# Replay the histopathologic-cancer-detection scheduler timeline without
# running the MLEvolve agent or making LLM calls.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SOURCE_RUN="$ROOT/runs/profile_scheduler_compare_histopathologic-cancer-detection_20260704_212842"
FULL_FIXTURE_DIR="$ROOT/scheduler_benchmark_test/fixtures/histopathologic-cancer-detection_20260704_212842"
FAST_FIXTURE_DIR="$ROOT/scheduler_benchmark_test/fixtures/histopathologic-cancer-detection_20260704_212842_clean_scripts_completed"
STRESS_FIXTURE_DIR="$ROOT/scheduler_benchmark_test/stress_test_data/histopathologic-cancer-detection_20260704_212842_scheduler_stress_2epoch"
ARCHIVE_ROOT="$STRESS_FIXTURE_DIR"
FIXTURE_DIR="$FULL_FIXTURE_DIR"
OUTPUT_ROOT="$ROOT/runs/scheduler_replay_histopathologic-cancer-detection_$(date +%Y%m%d_%H%M%S)"
PRESET="full"
RUNNER_MODE="real"
SPEEDUP="1"
UNTIL_SECONDS=""
INCLUDE_FINAL_CLEANUP=0
DRY_RUN=0
GENERATE_PLOTS=1
POST_ACTIONS_WAIT_SECONDS=60
NO_SLEEP=0
WAIT_FOR_ALL=0
CANCEL_POLICY="replay"
CLEAN_PROFILE_DB=0
EXTRA_ARGS=()

FIXTURE_DIR_SET=0
RUNNER_MODE_SET=0
SPEEDUP_SET=0
POST_ACTIONS_WAIT_SECONDS_SET=0
INCLUDE_FINAL_CLEANUP_SET=0
NO_SLEEP_SET=0
WAIT_FOR_ALL_SET=0
CANCEL_POLICY_SET=0
CLEAN_PROFILE_DB_SET=0

usage() {
  cat <<'EOF'
Usage:
  bash scheduler_benchmark_test/run_histopath_scheduler_replay.sh [options]

Options:
  --preset stress|quick|smoke|full   Replay preset. full preserves legacy defaults.
                                      stress builds/uses the 2-epoch archived-source fixture,
                                      submits all jobs immediately, ignores old cancels,
                                      waits for all jobs, and cleans scheduler/profile DBs.
                                      quick uses the original full fixture,
                                      real runner, speedup 100, ignores cancels,
                                      and waits for all jobs to finish.
                                      smoke uses noop, --no-sleep, and 5s final wait.
  --source-run PATH                  Original profile_scheduler_compare run root.
  --fixture-dir PATH                 Fixture directory. Created when missing.
  --output-root PATH                 Replay output root.
  --runner-mode real|noop            Real generated scripts by default; noop for smoke tests.
  --speedup N                        Divide timeline sleeps by N. Defaults to 1.
  --until-seconds N                  Replay actions up to this original offset.
  --include-final-cleanup-cancels    Include original timeout cleanup cancel burst.
  --post-actions-wait-seconds N      Wait for jobs after the final selected action. Defaults to 60.
                                      Ignored when --wait-for-all is active.
  --wait-for-all                     Wait for every submitted job to finish; do not cleanup-cancel.
  --no-wait-for-all                  Use post-action wait/cancel behavior even if a preset enables wait-for-all.
  --cancel-policy replay|ignore      Replay or ignore CANCEL actions. Defaults to replay.
  --clean-profile-db                 Remove replay scheduler/profile SQLite DBs before starting.
  --no-sleep                         Submit selected actions immediately. Intended for smoke checks only.
  --dry-run                          Build summary without starting the scheduler.
  --skip-plots, --no-plots           Skip comparison plot summary generation.
  --                                Remaining args are passed to replay_scheduler_timeline.
EOF
}

apply_preset_defaults() {
  case "$PRESET" in
    full)
      [[ "$FIXTURE_DIR_SET" -eq 0 ]] && FIXTURE_DIR="$FULL_FIXTURE_DIR"
      [[ "$RUNNER_MODE_SET" -eq 0 ]] && RUNNER_MODE="real"
      [[ "$SPEEDUP_SET" -eq 0 ]] && SPEEDUP="1"
      [[ "$POST_ACTIONS_WAIT_SECONDS_SET" -eq 0 ]] && POST_ACTIONS_WAIT_SECONDS=60
      [[ "$INCLUDE_FINAL_CLEANUP_SET" -eq 0 ]] && INCLUDE_FINAL_CLEANUP=0
      [[ "$NO_SLEEP_SET" -eq 0 ]] && NO_SLEEP=0
      [[ "$WAIT_FOR_ALL_SET" -eq 0 ]] && WAIT_FOR_ALL=0
      [[ "$CANCEL_POLICY_SET" -eq 0 ]] && CANCEL_POLICY="replay"
      ;;
    quick)
      [[ "$FIXTURE_DIR_SET" -eq 0 ]] && FIXTURE_DIR="$FULL_FIXTURE_DIR"
      [[ "$RUNNER_MODE_SET" -eq 0 ]] && RUNNER_MODE="real"
      [[ "$SPEEDUP_SET" -eq 0 ]] && SPEEDUP="100"
      [[ "$POST_ACTIONS_WAIT_SECONDS_SET" -eq 0 ]] && POST_ACTIONS_WAIT_SECONDS=60
      [[ "$INCLUDE_FINAL_CLEANUP_SET" -eq 0 ]] && INCLUDE_FINAL_CLEANUP=0
      [[ "$NO_SLEEP_SET" -eq 0 ]] && NO_SLEEP=0
      [[ "$WAIT_FOR_ALL_SET" -eq 0 ]] && WAIT_FOR_ALL=1
      [[ "$CANCEL_POLICY_SET" -eq 0 ]] && CANCEL_POLICY="ignore"
      ;;
    stress)
      [[ "$FIXTURE_DIR_SET" -eq 0 ]] && FIXTURE_DIR="$STRESS_FIXTURE_DIR"
      [[ "$RUNNER_MODE_SET" -eq 0 ]] && RUNNER_MODE="real"
      [[ "$SPEEDUP_SET" -eq 0 ]] && SPEEDUP="1"
      [[ "$POST_ACTIONS_WAIT_SECONDS_SET" -eq 0 ]] && POST_ACTIONS_WAIT_SECONDS=0
      [[ "$INCLUDE_FINAL_CLEANUP_SET" -eq 0 ]] && INCLUDE_FINAL_CLEANUP=0
      [[ "$NO_SLEEP_SET" -eq 0 ]] && NO_SLEEP=1
      [[ "$WAIT_FOR_ALL_SET" -eq 0 ]] && WAIT_FOR_ALL=1
      [[ "$CANCEL_POLICY_SET" -eq 0 ]] && CANCEL_POLICY="ignore"
      [[ "$CLEAN_PROFILE_DB_SET" -eq 0 ]] && CLEAN_PROFILE_DB=1
      ;;
    smoke)
      [[ "$FIXTURE_DIR_SET" -eq 0 ]] && FIXTURE_DIR="$FAST_FIXTURE_DIR"
      [[ "$RUNNER_MODE_SET" -eq 0 ]] && RUNNER_MODE="noop"
      [[ "$SPEEDUP_SET" -eq 0 ]] && SPEEDUP="1"
      [[ "$POST_ACTIONS_WAIT_SECONDS_SET" -eq 0 ]] && POST_ACTIONS_WAIT_SECONDS=5
      [[ "$INCLUDE_FINAL_CLEANUP_SET" -eq 0 ]] && INCLUDE_FINAL_CLEANUP=0
      [[ "$NO_SLEEP_SET" -eq 0 ]] && NO_SLEEP=1
      [[ "$WAIT_FOR_ALL_SET" -eq 0 ]] && WAIT_FOR_ALL=0
      [[ "$CANCEL_POLICY_SET" -eq 0 ]] && CANCEL_POLICY="replay"
      ;;
    *)
      echo "Unsupported preset: $PRESET" >&2
      usage >&2
      exit 2
      ;;
  esac
  return 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset)
      PRESET="${2:?--preset requires a value}"
      shift 2
      ;;
    --source-run)
      SOURCE_RUN="$(realpath -m "${2:?--source-run requires a value}")"
      shift 2
      ;;
    --fixture-dir)
      FIXTURE_DIR="$(realpath -m "${2:?--fixture-dir requires a value}")"
      FIXTURE_DIR_SET=1
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$(realpath -m "${2:?--output-root requires a value}")"
      shift 2
      ;;
    --runner-mode)
      RUNNER_MODE="${2:?--runner-mode requires a value}"
      RUNNER_MODE_SET=1
      shift 2
      ;;
    --speedup)
      SPEEDUP="${2:?--speedup requires a value}"
      SPEEDUP_SET=1
      shift 2
      ;;
    --until-seconds)
      UNTIL_SECONDS="${2:?--until-seconds requires a value}"
      shift 2
      ;;
    --include-final-cleanup-cancels)
      INCLUDE_FINAL_CLEANUP=1
      INCLUDE_FINAL_CLEANUP_SET=1
      shift
      ;;
    --post-actions-wait-seconds)
      POST_ACTIONS_WAIT_SECONDS="${2:?--post-actions-wait-seconds requires a value}"
      POST_ACTIONS_WAIT_SECONDS_SET=1
      shift 2
      ;;
    --wait-for-all)
      WAIT_FOR_ALL=1
      WAIT_FOR_ALL_SET=1
      shift
      ;;
    --no-wait-for-all)
      WAIT_FOR_ALL=0
      WAIT_FOR_ALL_SET=1
      shift
      ;;
    --cancel-policy)
      CANCEL_POLICY="${2:?--cancel-policy requires a value}"
      CANCEL_POLICY_SET=1
      shift 2
      ;;
    --clean-profile-db)
      CLEAN_PROFILE_DB=1
      CLEAN_PROFILE_DB_SET=1
      shift
      ;;
    --no-sleep)
      NO_SLEEP=1
      NO_SLEEP_SET=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --skip-plots|--no-plots)
      GENERATE_PLOTS=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

apply_preset_defaults

if [[ ! -f "$FIXTURE_DIR/timeline.json" || ! -f "$FIXTURE_DIR/jobs.jsonl" ]]; then
  if [[ "$PRESET" == "stress" ]]; then
    echo "==> Building 2-epoch scheduler stress fixture"
    python -m scheduler_benchmark_test.replay_model_sources build-stress-fixture \
      --source-fixture "$FAST_FIXTURE_DIR" \
      --output-fixture "$FIXTURE_DIR" \
      --archive-root "$ARCHIVE_ROOT" \
      --max-epochs 2
  else
    echo "==> Extracting scheduler replay fixture"
    python -m scheduler_benchmark_test.extract_scheduler_timeline \
      --source-run "$SOURCE_RUN" \
      --output-dir "$FIXTURE_DIR"
  fi
fi

echo "Preset: $PRESET"
echo "Fixture: $FIXTURE_DIR"
echo "Output root: $OUTPUT_ROOT"
echo "Runner mode: $RUNNER_MODE"
echo "Speedup: $SPEEDUP"
echo "Post-actions wait: $POST_ACTIONS_WAIT_SECONDS"
echo "Wait for all: $WAIT_FOR_ALL"
echo "Cancel policy: $CANCEL_POLICY"
echo "No sleep: $NO_SLEEP"
echo "Clean profile DB: $CLEAN_PROFILE_DB"

replay_cmd=(
  python -m scheduler_benchmark_test.replay_scheduler_timeline
  --fixture "$FIXTURE_DIR"
  --output-root "$OUTPUT_ROOT"
  --runner-mode "$RUNNER_MODE"
  --speedup "$SPEEDUP"
  --post-actions-wait-seconds "$POST_ACTIONS_WAIT_SECONDS"
)

if [[ -n "$UNTIL_SECONDS" ]]; then
  replay_cmd+=(--until-seconds "$UNTIL_SECONDS")
fi
if [[ "$INCLUDE_FINAL_CLEANUP" -eq 1 ]]; then
  replay_cmd+=(--include-final-cleanup-cancels)
fi
if [[ "$DRY_RUN" -eq 1 ]]; then
  replay_cmd+=(--dry-run)
fi
if [[ "$NO_SLEEP" -eq 1 ]]; then
  replay_cmd+=(--no-sleep)
fi
if [[ "$WAIT_FOR_ALL" -eq 1 ]]; then
  replay_cmd+=(--wait-for-all)
fi
if [[ "$CLEAN_PROFILE_DB" -eq 1 ]]; then
  replay_cmd+=(--clean-profile-db)
fi
replay_cmd+=(--cancel-policy "$CANCEL_POLICY")
replay_cmd+=("${EXTRA_ARGS[@]}")

echo "==> Replaying scheduler timeline"
"${replay_cmd[@]}"

if [[ "$GENERATE_PLOTS" -eq 1 && "$DRY_RUN" -eq 0 ]]; then
  echo "==> Generating replay plot summary"
  python "$ROOT/utils/plot_hardware_awareness_comparison.py" \
    --run-root "$OUTPUT_ROOT" \
    --output-dir "$OUTPUT_ROOT/comparison_plots" \
    --modes scheduler_replay || true
fi

echo "Replay output: $OUTPUT_ROOT"

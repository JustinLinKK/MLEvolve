# Baseline vs Hardware-Aware Metrics Test Instructions

This runbook describes how to generate comparable performance metrics when the
baseline run is produced on the `main` branch and the hardware-aware run is
produced after switching back to `hardware-awared`.

The important artifact contract is:

- Baseline metrics live under `<RUN_ROOT>/baseline/...`.
- Hardware-aware metrics live under `<RUN_ROOT>/hardware_aware/...`.
- Both runs must use the same dataset, GPU, seed, step count, draft count, and
  timeout settings if you want a fair comparison.

Do not use `compare_hardware_awareness.sh` for this specific cross-branch
workflow. That script runs its own baseline from the `hardware-awared` branch,
which is not the same thing as comparing against the `main` baseline artifact.

## 1. Pick The Test Inputs

Run these from the repository root. Use absolute paths for `RUN_ROOT` and
`DATASET_ROOT` so they survive branch switches.

```bash
export COMPETITION_ID="dogs-vs-cats-redux-kernels-edition"
export DATASET_ROOT="/path/to/mle-bench/data"
export RUN_ROOT="$(pwd)/runs/main_vs_hardware_${COMPETITION_ID}_$(date +%Y%m%d_%H%M%S)"
export SERVER_ID=111
export MEMORY_INDEX=0
export STEPS=20
export INITIAL_DRAFTS=3
export SEED=42
export TIMEOUT_SECONDS=0
```

For a full benchmark, increase `STEPS` to the planned experiment budget. For a
smoke test, keep `STEPS` small.

The prepared dataset must already exist at:

```bash
"$DATASET_ROOT/$COMPETITION_ID/prepared/public/description.md"
```

If it does not exist, prepare it with MLE-bench first:

```bash
mlebench prepare -c "$COMPETITION_ID" --data-dir "$DATASET_ROOT" --skip-verification
```

## 2. Generate The Baseline Metric On `main`

Switch to `main` and confirm the baseline measurement script exists.

```bash
git switch main
test -x ./measure_baseline_performance.sh
```

Optional dry run:

```bash
bash measure_baseline_performance.sh "$COMPETITION_ID" \
  --dataset-root "$DATASET_ROOT" \
  --run-root "$RUN_ROOT" \
  --server-id "$SERVER_ID" \
  --memory-index "$MEMORY_INDEX" \
  --steps "$STEPS" \
  --initial-drafts "$INITIAL_DRAFTS" \
  --seed "$SEED" \
  --timeout-seconds "$TIMEOUT_SECONDS" \
  --no-validation-server \
  --dry-run
```

Run the real baseline measurement:

```bash
bash measure_baseline_performance.sh "$COMPETITION_ID" \
  --dataset-root "$DATASET_ROOT" \
  --run-root "$RUN_ROOT" \
  --server-id "$SERVER_ID" \
  --memory-index "$MEMORY_INDEX" \
  --steps "$STEPS" \
  --initial-drafts "$INITIAL_DRAFTS" \
  --seed "$SEED" \
  --timeout-seconds "$TIMEOUT_SECONDS"
```

This command starts the validation server, runs `run.py`, and writes:

```text
$RUN_ROOT/baseline/command.txt
$RUN_ROOT/baseline/stdout.log
$RUN_ROOT/baseline/exit_code.txt
$RUN_ROOT/baseline/runs/<timestamp>_<competition>_baseline/logs/comparison_metrics.json
$RUN_ROOT/baseline/runs/<timestamp>_<competition>_baseline/logs/hardware_samples.csv
$RUN_ROOT/baseline/runs/<timestamp>_<competition>_baseline/logs/hardware_report.md
```

Verify the baseline metric file:

```bash
export BASELINE_METRICS="$(
  find "$RUN_ROOT/baseline" -path '*/logs/comparison_metrics.json' | sort | tail -n 1
)"

test -n "$BASELINE_METRICS"
python -m json.tool "$BASELINE_METRICS" | sed -n '1,120p'
```

The JSON should contain `"mode": "baseline"`.

## 3. Switch To `hardware-awared`

Keep the same shell session if possible so the exported variables remain
available.

```bash
git switch hardware-awared
```

If the branch uses a root config file, make sure it exists and contains the same
LLM/provider settings you used for the baseline run:

```bash
test -f config.yaml || cp config.example.yaml config.yaml
```

Edit `config.yaml` only if API keys, model providers, or scheduler settings need
to be adjusted for the machine.

## 4. Start The Validation Server For The Hardware-Aware Run

The hardware-aware run below calls `run.py` directly so it can reuse the same
`RUN_ROOT` layout. Start the validation server manually:

```bash
export GRADING_SERVER_PORT=$((5005 + SERVER_ID))
mkdir -p "$RUN_ROOT/validation_server_hardware_aware"

python -u -m engine.validation.format_server \
  dataset_dir="$DATASET_ROOT" \
  data_dir="none" \
  desc_file="none" \
  > "$RUN_ROOT/validation_server_hardware_aware/server_${SERVER_ID}.log" 2>&1 &

export VALIDATION_PID=$!
```

Wait for it to become healthy:

```bash
python - <<PY
import time
import urllib.request

url = "http://127.0.0.1:${GRADING_SERVER_PORT}/health"
for _ in range(30):
    try:
        urllib.request.urlopen(url, timeout=1).read()
        print("validation server ready:", url)
        break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("validation server did not become healthy")
PY
```

## 5. Generate The Hardware-Aware Metric

Run `hardware_aware` mode into the same shared `RUN_ROOT`:

```bash
CUDA_VISIBLE_DEVICES="$MEMORY_INDEX" python run.py \
  exp_id="$COMPETITION_ID" \
  dataset_dir="$DATASET_ROOT" \
  data_dir="$DATASET_ROOT/$COMPETITION_ID/prepared/public" \
  desc_file="$DATASET_ROOT/$COMPETITION_ID/prepared/public/description.md" \
  exp_name="${COMPETITION_ID}_hardware_aware" \
  log_dir="$RUN_ROOT/hardware_aware/runs" \
  workspace_dir="$RUN_ROOT/hardware_aware/runs" \
  experiment.mode=hardware_aware \
  scheduler.runtime_root="$RUN_ROOT/hardware_aware/scheduler_runtime" \
  agent.steps="$STEPS" \
  agent.initial_drafts="$INITIAL_DRAFTS" \
  agent.seed="$SEED"
```

If you want a timeout, wrap the command:

```bash
CUDA_VISIBLE_DEVICES="$MEMORY_INDEX" timeout --foreground --signal=TERM --kill-after=10s "${TIMEOUT_SECONDS}s" \
  python run.py ...
```

After the run finishes, stop the validation server:

```bash
kill "$VALIDATION_PID" 2>/dev/null || true
```

Verify the hardware-aware metric file:

```bash
export HARDWARE_METRICS="$(
  find "$RUN_ROOT/hardware_aware" -path '*/logs/comparison_metrics.json' | sort | tail -n 1
)"

test -n "$HARDWARE_METRICS"
python -m json.tool "$HARDWARE_METRICS" | sed -n '1,120p'
```

The JSON should contain `"mode": "hardware_aware"`.

## 6. Generate Comparison Tables And Plots

On `hardware-awared`, generate summary JSON, Markdown, and PNG plots:

```bash
python utils/plot_hardware_awareness_comparison.py \
  --run-root "$RUN_ROOT" \
  --modes baseline hardware_aware \
  --output-dir "$RUN_ROOT/comparison_plots"
```

Generate a direct baseline-vs-hardware Markdown table:

```bash
python utils/compare_experiment_runs.py \
  "$BASELINE_METRICS" \
  "$HARDWARE_METRICS" \
  --format markdown \
  --output "$RUN_ROOT/comparison_plots/baseline_vs_hardware_aware.md"
```

Expected comparison artifacts:

```text
$RUN_ROOT/comparison_plots/comparison_summary.json
$RUN_ROOT/comparison_plots/comparison_summary.md
$RUN_ROOT/comparison_plots/comparison_metrics.png
$RUN_ROOT/comparison_plots/utilization_timeseries.png
$RUN_ROOT/comparison_plots/baseline_vs_hardware_aware.md
```

## 7. Final Sanity Checks

Check both metric files exist:

```bash
test -f "$BASELINE_METRICS"
test -f "$HARDWARE_METRICS"
```

Check the modes:

```bash
python - <<PY
import json
from pathlib import Path

baseline = json.loads(Path("$BASELINE_METRICS").read_text())
hardware = json.loads(Path("$HARDWARE_METRICS").read_text())
assert baseline["mode"] == "baseline", baseline["mode"]
assert hardware["mode"] == "hardware_aware", hardware["mode"]
print("baseline run:", baseline.get("run_id"))
print("hardware-aware run:", hardware.get("run_id"))
print("baseline best metric:", baseline.get("best_metric"))
print("hardware-aware best metric:", hardware.get("best_metric"))
PY
```

Confirm the run root has exactly the comparison directories you expect:

```bash
find "$RUN_ROOT" -maxdepth 2 -type d | sort
```

## Troubleshooting

- If `comparison_metrics.json` is missing, inspect `stdout.log` and
  `MLEvolve.log`. Metrics are only written after `run.py` reaches the point
  where a journal exists.
- If the validation server does not become healthy, inspect
  `$RUN_ROOT/validation_server_hardware_aware/server_${SERVER_ID}.log`.
- If plots omit a mode, confirm the directory names are exactly `baseline` and
  `hardware_aware` under the same `RUN_ROOT`.
- If results are not comparable, verify both commands used the same
  `STEPS`, `INITIAL_DRAFTS`, `SEED`, `MEMORY_INDEX`, dataset path, and model
  provider configuration.
- If branch switching hides an uncommitted local file, commit or stash only the
  intended code changes first. Keep `RUN_ROOT` under `runs/` or another ignored
  path so generated artifacts are not part of the source diff.

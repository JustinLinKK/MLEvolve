# RTX 5090 pressure benchmark

Run the full calibrated comparison from the repository root:

```bash
bash scheduler_benchmark_test/run_rtx5090_pressure_benchmark.sh
```

The default order is `baseline,warm,cold`; each measured mode has its own
5,400-second cap. Calibration and GPU cooldown happen before the measured
timer. To resume an interrupted run, point `OUTPUT_ROOT` at its existing
artifact directory and run the same command. Completed phase markers are not
repeated.

A seconds-scale implementation check is available with:

```bash
MODE_TIMEOUT=180 bash scheduler_benchmark_test/run_rtx5090_pressure_benchmark.sh --smoke
```

The entry point always invokes final analysis. The output root contains raw
attempts, logical results, scheduler events, calibration/profile snapshots,
the hardware/software manifest, 0.5-second GPU telemetry, CSV/JSON summaries,
`REPORT.md`, and PNG/PDF three-panel Gantt charts.

## Model-quality audit

The pressure trace measures scheduling performance, not accuracy: its target is
a synthetic zero-regression loss and it has no validation labels. Run the
separate paired quality audit after the pressure benchmark:

```bash
bash scheduler_benchmark_test/run_model_quality_audit.sh
```

By default it attaches to the newest `rtx5090-*` artifact directory. Eight BF16
classification replicates run with identical initial checkpoints, data, sample
order, optimizer settings, and epoch counts under MP2, warm scheduling, and
cold scheduling. The audit requires both scheduler modes to reach at least
three concurrent streams on one host with distinct stream IDs. It compares
final validation accuracy within a predeclared ±0.5 percentage-point band,
validation predictions, final parameter hashes, and the complete validation
learning curve.

The result is written as `QUALITY_REPORT.md`, `quality-summary.json`,
`quality-results.csv`, `quality_accuracy_by_job_bar.{png,pdf}`, and
`quality_accuracy_comparison.{png,pdf}`. Use
`OUTPUT_ROOT=/path/to/output` to select another destination, or rerender an
existing audit with:

```bash
python -m scheduler_benchmark_test.model_quality_benchmark analyze \
  --output-root /path/to/output
```

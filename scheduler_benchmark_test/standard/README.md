# Standard Histopathology Scheduler Benchmark v1

This benchmark replays one deterministic 100-job Poisson arrival trace through a strictly sequential FIFO baseline and four uncapped scheduler arms. Every real job trains for 50 epochs over all 174,464 labeled 96×96 RGB images. The safety target is the NVIDIA A10 feature set and a 22,528 MiB scheduler/allocator ceiling; reports always record the physical GPU name.

The committed fixture is `scheduler_benchmark_test/fixtures/standard_histopath_v1/`. It contains 100 self-contained model sources, complete `TrainingJob` payloads, the timeline, replay settings, dataset identity, distributions, schema/generator versions, and SHA-256 checksums. Model architecture code is inline in every generated source; only the dataset/training loop is shared.

## Commands

Set the prepared dataset directory first:

```bash
export HISTOPATH_DATA_ROOT=/datasets/histopathologic-cancer-detection/prepared/public
```

Regenerate or check the versioned fixture:

```bash
python -m scheduler_benchmark_test.standard.generate_fixture --data-root "$HISTOPATH_DATA_ROOT"
python -m scheduler_benchmark_test.standard.generate_fixture --check --data-root "$HISTOPATH_DATA_ROOT"
```

Run the required sequential one-epoch validation. `--resume` reuses only successful results whose source checksum still matches:

```bash
python -m scheduler_benchmark_test.standard.validate \
  --data-root "$HISTOPATH_DATA_ROOT" \
  --output-root /results/standard-histopath-validation \
  --resume
```

Run the full five-arm, three-repetition matrix:

```bash
python -m scheduler_benchmark_test.standard.run_benchmark \
  --data-root "$HISTOPATH_DATA_ROOT" \
  --output-root /results/standard-histopath-v1 \
  --resume
```

Completed cases are skipped on resume. A failed or interrupted case gets a new numbered attempt directory, so prior evidence is never overwritten. Repetitions rotate arm order. FIFO is hard-coded to one process; scheduler arms use `adaptive`, a 16-job candidate window, and at most eight packed jobs. The branch arm uses v3 profile curves and the ML arm uses `ml_predictor`.

For fast plumbing coverage without training or performance claims:

```bash
python -m scheduler_benchmark_test.standard.run_benchmark \
  --output-root /tmp/standard-histopath-noop \
  --runner-mode noop --repetitions 1 --no-sleep
```

Rebuild the aggregate report from retained case summaries:

```bash
python -m scheduler_benchmark_test.standard.reporting \
  --results-root /results/standard-histopath-v1
```

`primary_cuda` and `primary_stream` are recognized future arms. They intentionally fail preflight while the PerfSeer adapter is unhealthy. Even after it becomes healthy, an ML-primary case is claim-eligible only with at least 95% ML selection coverage and zero adapter failures. ML-shadow cases with no usable ML predictions are labeled `plumbing-only` and excluded from ML performance claims.

## Acceptance

A real case is accepted only when all 100 jobs complete 50 epochs and report 872,320,000 processed samples in total, with no failure, cancellation, timeout, skip, or OOM. FIFO must have a maximum observed concurrency of exactly one. Validation requires finite loss, 174,464 samples per job, peak CUDA allocation/reservation under 22,528 MiB, and no more than 15% peak-allocation spread within each five-job profile bucket. Large checkpoints are disabled.

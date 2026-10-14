# Scheduler estimation repair and Qwen3.8 V100 measurement

## Pathfinder / PetFinder finding

- Source comparison image: `records/petfinder_v100_orig_vs_scheduler.png`.
- The scheduler-side estimated duration was `None` while the profile database had
  `BatchSizeObservation.metadata.seconds_per_epoch` values.
- Root cause: `ResourceEstimator.predicted_remaining_runtime_seconds` read only
  `RuntimeProfile`; graph knowledge and hardware context likewise omitted batch
  observations as runtime evidence.

## Scheduler changes

- `predicted_remaining_runtime_seconds` now computes remaining duration from the
  matching batch observation's seconds per epoch and remaining epochs when a
  runtime profile is unavailable.
- Graph runtime evidence now emits `seconds_per_epoch` and
  `estimated_total_runtime_seconds` from batch-size observations.
- Hardware-context compaction preserves that estimate and search nodes consume
  it.
- Replay and repeated benchmark drivers no longer configure `parallel_job_cap`.
  The scheduler admits work from measured profile VRAM, the 31 GiB memory bound,
  and incremental placement; no fixed Max Parallel Jobs is used.

## Regression checks

- Batch-observation-only remaining-runtime estimate, graph runtime estimate, and
  replay configuration with no parallel-job cap: 3 passed.

## Follow-up scheduler repair

- A runtime profile with `estimated_total_runtime_seconds = 0` previously
  overrode valid branch-observation epoch timing and made the scheduler treat
  remaining work as zero.
- Non-positive or invalid total-runtime profile values now fall back to the
  batch observation's `seconds_per_epoch` and remaining epoch count.
- Runtime scheduler paths no longer read `parallel_job_cap`; decision replay,
  benchmark configuration, and baseline replay use incremental branch-profile
  and live-telemetry admission instead.
- Validation: 109 scheduler and benchmark tests passed; one unrelated
  Unix-socket integration test was excluded because this sandbox rejects
  `AF_UNIX` socket binding.

## Nautilus V100 local model deployment

- SSH alias: `Nautilus-V100`. SSH aliases cannot contain spaces.
- GPUs used: Tesla V100-SXM2-32GB GPU 0 and GPU 1; GPU 2 and GPU 3 were idle.
- Model: `Qwen/Qwen3.8-27B`.
- Quantization: bitsandbytes dynamic INT8; model device map `balanced`, maximum
  30 GiB per selected GPU.
- Runtime: PyTorch 2.6.0 CUDA 11.8, which executed an SM70 CUDA tensor test on
  both V100 GPUs. PyTorch 2.11 CUDA 12.8 was rejected because its wheel lacks
  SM70 kernels.
- The remote home volume stalled while replacing its original virtual
  environment, so the verified runtime environment is located at
  `/tmp/qwen38-v100-int8-venv` on the Pod local disk. The service remains
  running at `127.0.0.1:8000` on Nautilus V100.

## Measurement

- One warm-up request followed by three sequential streamed requests.
- Prompt: `Explain in one sentence why profile-based GPU scheduling needs runtime estimates.`
- Maximum completion length: 128 tokens; observed completion length: 36 tokens
  per measured request.
- Median time to first token: 2.416650878 seconds.
- Median generation throughput: 1.839899817 tokens per second.
- Median total request time: 21.968061263 seconds.
- Raw measurements: `records/qwen38_v100_int8_benchmark.json`.
- Required combined Gantt and metric-node image:
  `records/qwen38_v100_int8_benchmark.png`.

## Three-V100 measurement

- GPUs used: Tesla V100-SXM2-32GB GPU 0, GPU 1, and GPU 2. GPU 3 remained
  allocated for other tasks and used 40 MiB at measurement completion.
- Selected-GPU memory at measurement completion: GPU 0 7214 MiB, GPU 1 9766
  MiB, GPU 2 14712 MiB. Each configured upper bound remained 30 GiB.
- Same warm-up, prompt, maximum completion length, and three streamed requests
  as the two-GPU measurement.
- Median time to first token: 2.565404023 seconds.
- Median generation throughput: 1.743962445 tokens per second.
- Median total request time: 23.200376069 seconds.
- Raw measurements: `records/qwen38_v100_int8_3gpu_benchmark.json`.
- Per-run image: `records/qwen38_v100_int8_3gpu_benchmark.png`.
- Combined two- and three-GPU Gantt and metric-node image:
  `records/qwen38_v100_int8_2v3gpu_comparison.png`.

## Comparison conclusion

- For this single-stream, 36-completion-token prompt, two V100 GPUs measured
  lower median time to first token and higher median generation throughput than
  three V100 GPUs.

# stress_bench — total-training-time scheduler optimization

Benchmark harness + a placement-policy optimization for `localml_scheduler` that reduces
**total training time** (Σ per-job training seconds) on compute-bound GPU workloads.

## The optimization
`run_bench.py` exposes `--placement-policy {pack, gated}`:

- **`pack`** (baseline): `parallel_default` mode + permissive slowdown threshold — co-locates jobs
  on memory-fit. On a compute-bound GPU this time-slices a saturated device, so each co-located job
  is stretched ~linearly with the co-location degree.
- **`gated`** (optimized): `parallel_time_aware` mode + `pack_reject_max_slowdown = 1.15`. The
  scheduler probes solo-vs-paired execution and **refuses any co-location whose measured per-job
  slowdown exceeds 1.15×**, so compute-bound jobs (which already saturate the GPU) run **exclusive**
  at solo speed. It still co-locates genuinely complementary jobs (measured slowdown < 1.15×).

Metric note: the gate is on **per-job slowdown** (`paired/solo`), which is the quantity total
training time is made of (`Σ solo·slowdown`), not throughput/makespan gain. The refactored scheduler
gates on measured slowdown rather than SM-utilization (a time-occupancy proxy, not compute intensity).

## Result (matched 16-job workload, NVIDIA A10, profile predictor on CPU)
| Policy | Σ training time | Placement | Makespan |
|---|---|---|---|
| pack (baseline) | 1858 s | all co-located | 627 s |
| **gated (optimized)** | **494 s** | all exclusive | **584 s** |

**Total training time −73%** with no makespan cost (validated on scheduler 8d6dee9: −71%, and on the
"corrected time packing" refactor 102a2ae: −73%). Full analysis, per-job data, Gantt, and the trace
are archived under `perfseer_test/results/scheduler_opt_totaltime/`.

## Files
- `run_bench.py` — driver: mp_only / scheduler conditions, NVML sampling, the `--placement-policy` knob
- `stress_runner.py` — in-process training runner (streamed synthetic batches = full-pass epochs)
- `mp_worker.py` — standalone multiprocess baseline worker

## Reproduce
```
python -m scheduler_benchmark_test.stress_bench.run_bench \
  --condition scheduler_profile --trace <trace.jsonl> --outdir <out> \
  --max-parallel 4 --vram-budget-gib 20 --placement-policy gated
```

# stress_bench — time-aware scheduler validation

This harness measures total training time and makespan on compute-bound GPU
workloads. Scheduler conditions always use the production
`parallel_time_aware` policy:

- one profiled job starts as the anchor;
- one newcomer is considered at a time;
- verified drain-time gain decides whether the newcomer stays;
- predicted and live VRAM are admission ceilings only.

Older results under `perfseer_test/results/scheduler_opt_totaltime/` compare a
removed fill-oriented implementation with an early slowdown gate. Treat those
files as historical evidence, not runnable scheduler configurations.

## Files

- `run_bench.py` — scheduler/standalone driver and GPU sampling
- `stress_runner.py` — scheduler-aware training runner
- `mp_worker.py` — standalone multiprocess control

## Reproduce

```bash
python -m scheduler_benchmark_test.stress_bench.run_bench \
  --condition scheduler_profile --trace <trace.jsonl> --outdir <out> \
  --gpu-vram-gib 20
```

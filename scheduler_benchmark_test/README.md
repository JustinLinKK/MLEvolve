# Scheduler benchmark suite

This directory validates the production scheduler policy:
`parallel_time_aware`.

The policy starts one profiled anchor, considers one newcomer at a time, and
retains the newcomer only when measured or predicted piecewise drain time
improves by at least `colocation.min_gain`. VRAM is a hard admission constraint;
it is never the benchmarked placement objective.

## Production replay

`replay_scheduler.py` accepts only `--mode parallel_time_aware`. A measured run
has two stages:

1. Run exclusive five-option calibration and write a hardware-scoped profile.
2. Replay the workload with that profile on a non-exclusive backend.

`repeat_time_aware_benchmark.py` automates calibration and repeated runs. Its
serial control is the same production policy with an exclusive backend; neither
control nor scheduler replay sets a maximum parallel-job cap.

```bash
python scheduler_benchmark_test/repeat_time_aware_benchmark.py \
  --results-dir results/scheduler_benchmark_test/time_aware \
  --data-root /path/to/cassava/prepared/public \
  --repetitions 3 \
  --time-aware-backend cuda_process
```

Use `check_benchmark_env.py` before a real-GPU replay and keep each repetition
in a fresh runtime directory.

## Deterministic policy comparison

The trace simulator may compare `parallel_time_aware` with serial FIFO, a
simulator-only historical fill policy, and a small exhaustive oracle:

```bash
python -m localml_scheduler.scheduler.trace_simulator
pytest -q scheduler_benchmark_test/test_trace_policies.py
```

Those comparison policies are models inside the simulator. They are not valid
`SchedulerSettings.gpu_scheduler.mode` values and cannot be dispatched by the
production scheduler.

## Stress validation

`stress_bench/run_bench.py` compares the scheduler with a standalone
multiprocess control while keeping the scheduler policy fixed to
`parallel_time_aware`.

Historical result files may still contain removed mode names so old experiments
remain interpretable. Do not copy those labels into current configuration.

# Backend-Aware Trial-Ranking Trace Replay

The deterministic CPU-only trace fixture `backend_aware_benchmark_fixture()`
contains two compute-leaning jobs and one memory-leaning job on an MPS backend.
The compute/compute trial has measured slowdown 3.0; either compute/memory trial
has measured slowdown 1.05. All trial work is included in simulated wall time,
and `backend_awared` additionally charges the trial wall interval when deciding
whether to retain an unknown placement.

Run:

```bash
python - <<'PY'
from localml_scheduler.scheduler.trace_simulator import (
    backend_aware_benchmark_fixture,
    compare_policies,
    markdown_table,
)
print(markdown_table(compare_policies(backend_aware_benchmark_fixture())))
PY
```

Results at commit base `6b5cb056a237b9569f08d823d3371b12e15e56d5`:

| Policy | Makespan (s) | Mean flow (s) | Trial/rejected epochs | Rejections | Violations |
|---|---:|---:|---:|---:|---:|
| serial FIFO | 180.00 | 116.67 | 0.0 / 0.0 | 0 | 0 |
| VRAM-fill reference | 250.00 | 193.33 | 0.0 / 0.0 | 0 | 0 |
| baseline `parallel_time_aware` | 134.70 | 105.13 | 4.0 / 2.0 | 1 | 0 |
| `backend_awared` | 123.00 | 86.33 | 2.0 / 0.0 | 0 | 0 |
| small-trace exhaustive pack oracle | 133.50 | 86.50 | 0.0 / 0.0 | 0 | 0 |

For this fixture, backend-aware ranking finds a beneficial placement on its
first trial (one trial-to-benefit), while baseline tries the harmful
compute/compute pair first and needs a second trial. Backend-aware makespan is
8.7% lower than baseline after trial charging, with no memory, backend,
release-time, compatibility, parallel-cap, or starvation violation.

This is a focused deterministic regression fixture, not a GPU performance
claim. Real first-trial success, makespan, and trial-cost results still depend
on source coverage, configured hardware throughput/bandwidth, and live backend
measurements.

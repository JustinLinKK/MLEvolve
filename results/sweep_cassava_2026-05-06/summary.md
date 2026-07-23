# Sweep Results — 12-Config on Real Cassava (Claude opus 4.7 trace)

## Setting
- Workload: cassava-leaf-disease-classification (12 GB)
- LLM: Claude opus 4.7 via claude-agent-sdk (claude CLI subprocess)
- Trace: 20 nodes recorded by trace_recorder.py

## Per-config wall-clock (sec)

| ID | Mode | Backend | Probe | Runner | Wall-clock | Speedup vs B1 |
|---|---|---|---|---|---|---|
| B1 | serial_basic | exclusive | off | scheduler | 766.0s | 1.00x |
| B2 | serial_batch_optimized | exclusive | binary | scheduler | 783.7s | 0.98x |
| B3 | serial_batch_optimized | exclusive | power_of_two | scheduler | 753.8s | 1.02x |
| T1 | parallel_default | mps | off | scheduler | 622.3s | 1.23x |
| T2 | parallel_default | stream | off | scheduler | 135.6s | 5.65x |
| T4 | parallel_batch_optimized | mps | binary | scheduler | 0.0s | 0.00x |
| T5 | parallel_batch_optimized | mps | power_of_two | scheduler | 0.0s | 0.00x |
| T6 | parallel_batch_optimized | stream | binary | scheduler | 0.0s | 0.00x |
| T7 | parallel_batch_optimized | stream | power_of_two | scheduler | 0.0s | 0.00x |
| T8 | parallel_batch_optimized | mps | binary | torch_mp | 493.8s | 1.55x |
| T9 | parallel_batch_optimized | mps | power_of_two | torch_mp | 498.0s | 1.54x |
| T10 | parallel_batch_optimized | stream | binary | torch_mp | 465.8s | 1.64x |
| T11 | parallel_batch_optimized | stream | power_of_two | torch_mp | 467.3s | 1.64x |

## Files
- /tmp/sweep_2026-05-06/plots/gpu_utility_overlay.png
- /tmp/sweep_2026-05-06/plots/wallclock_bars.png
- /tmp/sweep_2026-05-06/plots/speedup_heatmap.png
- /tmp/sweep_2026-05-06/plots/agent_status_per_config.png

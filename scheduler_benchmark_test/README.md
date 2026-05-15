# Scheduler Benchmark Test

Benchmark harness for comparing `localml_scheduler` replay modes on the cassava workload trace, updated for the current scheduler settings layout and the RTX 5090 target machine.

## Overview

- All benchmark scripts now live under `scheduler_benchmark_test/`.
- The scheduler replay path uses a structured runner:
  - `localml_scheduler.examples.benchmark_timm_runner`
- Repeated TIMM startpoints are generated once and reused across jobs, so the scheduler RAM cache is exercised through `context.load_baseline_object()`.
- Packed batch optimization uses explicit threshold windows:
  - `binary`: exact search window `requested - down` to `requested + up`
  - `power_of_two`: powers of two from `2^(n - down)` to `2^(n + up)`, where `n = floor(log2(requested))`

## Prerequisites

| Item | Requirement |
|---|---|
| GPU | NVIDIA RTX 5090 class GPU recommended |
| VRAM | 32 GB physical VRAM; main sweep defaults use a 30 GiB safe budget |
| CUDA | Driver/toolkit compatible with your installed PyTorch build |
| Python deps | `torch`, `torchvision`, `timm`, `pandas`, `pillow`, `matplotlib`, `psutil`, `hydra-core`, `omegaconf` |
| `nvidia-cuda-mps-control` | Required only for MPS configs |

## Quick Start

```bash
cd /workspaces/MLEvolve

python -m pip install -r scheduler_benchmark_test/requirements.txt
export CASSAVA_ROOT=/your/path/to/cassava-leaf-disease-classification/prepared/public

# smoke test
bash scheduler_benchmark_test/smoke_run.sh

# full sweep with MPS configs
bash scheduler_benchmark_test/sweep_run.sh

# no-MPS / windows-style smoke
bash scheduler_benchmark_test/smoke_run_windows.sh

# no-MPS / windows-style full sweep
bash scheduler_benchmark_test/sweep_run_windows.sh

# no-MPS sweep using RAM-budget-only preload selection
COMMON_CACHE_WARM_POLICY=budget_only bash scheduler_benchmark_test/sweep_run_windows.sh

# focused cuda_process pack-3 / pack-4 comparison
bash scheduler_benchmark_test/cuda_process_pack34.sh

# short sensitivity check for pair composition
bash scheduler_benchmark_test/arch_sensitivity_run.sh

# focused preload + threshold sweep
bash scheduler_benchmark_test/feature_sweep.sh

# plot main sweep
python scheduler_benchmark_test/plot_results.py

# plot no-MPS sweep
RESULTS_DIR=results/scheduler_benchmark_test/main_sweep_windows \
python scheduler_benchmark_test/plot_results.py
```

## Runtime Controls

The Linux benchmark wrappers now isolate each run under `scheduler_benchmark_test/runtime/` by default and support these overrides:

| Variable | Default | Purpose |
|---|---|---|
| `BENCH_RUN_ROOT` | `scheduler_benchmark_test/runtime` | Parent scratch directory for per-run runtime state, workdirs, and MPS sockets |
| `BENCH_GPU_SAMPLER` | `query` | GPU telemetry mode: `query`, `dmon`, or `none` |
| `BENCH_CLEANUP_GRACE_SECONDS` | `10` | Grace period before a tracked child process is force-killed |
| `BENCH_ENABLE_MPS` | `true` unless `NO_MPS=1` | Enables the MPS-specific matrix and per-run MPS socket directories |

`gpu_metrics.csv` is now the default telemetry artifact. `plot_results.py` accepts either the new query-format sampler output or the legacy `dmon.csv` format.

## Generated Assets

`gen_trace_W3.py` and `gen_smoke_trace.py` generate:

- replayable Python scripts in `replay_codes_*`
- shared half-precision TIMM startpoints in `startpoints/`
- trace entries with:
  - `startpoint_id`
  - `startpoint_path`
  - `max_bs`
  - `dataset_seed`
  - `data_root`
- per-run scratch state under `scheduler_benchmark_test/runtime/<script>-<timestamp>-<pid>/`

This means:

- `replay_scheduler.py` can preload and cache real shared startpoint checkpoints
- `replay_torch_mp.py` still replays standalone generated scripts as a control path

## Main Sweep Defaults

The main sweep in [sweep_run.sh](/workspaces/MLEvolve/scheduler_benchmark_test/sweep_run.sh) currently uses:

| Variable | Default |
|---|---|
| `CONFIG_TIMEOUT` | `2700` |
| `COMMON_VRAM_BUDGET_GIB` | `30.0` |
| `COMMON_CACHE_WARM_TOP_K` | `4` |
| `COMMON_CACHE_ENTRY_CAPACITY` | `8` |
| `COMMON_CACHE_MAX_RAM_PERCENT` | `0.20` |
| `COMMON_CACHE_MEMORY_BUDGET_GIB` | `30.0` |
| `COMMON_BINARY_RANGE_UP` | `16` |
| `COMMON_BINARY_RANGE_DOWN` | `8` |
| `COMMON_POWER_OF_TWO_RANGE_UP` | `4` |
| `COMMON_POWER_OF_TWO_RANGE_DOWN` | `1` |

These are intended as aggressive 5090-era defaults.

## Config ID Table

### Main sweep with MPS available

| Config ID | Runner | Mode | Backend | Probe/Search | Notes |
|---|---|---|---|---|---|
| `B1` | scheduler | `serial_basic` | `exclusive` | `off` | baseline |
| `B2` | scheduler | `serial_batch_optimized` | `exclusive` | `binary` | exclusive binary probe |
| `B3` | scheduler | `serial_batch_optimized` | `exclusive` | `power_of_two` | exclusive 2^n probe |
| `T1` | scheduler | `parallel_default` | `mps` | `off` | parallel MPS baseline |
| `T2` | scheduler | `parallel_default` | `stream` | `off` | parallel stream baseline |
| `T4` | scheduler | `parallel_batch_optimized` | `mps` | `binary` | packed MPS + binary search |
| `T5` | scheduler | `parallel_batch_optimized` | `mps` | `power_of_two` | packed MPS + 2^n search |
| `T6` | scheduler | `parallel_batch_optimized` | `stream` | `binary` | packed stream + binary search |
| `T7` | scheduler | `parallel_batch_optimized` | `stream` | `power_of_two` | packed stream + 2^n search |
| `T8` | `torch_mp` | `parallel_batch_optimized` | `mps` | `binary` | control path, MPS |
| `T9` | `torch_mp` | `parallel_batch_optimized` | `mps` | `power_of_two` | control path, MPS |
| `T10` | `torch_mp` | `parallel_batch_optimized` | `stream` | `binary` | control path, stream |
| `T11` | `torch_mp` | `parallel_batch_optimized` | `stream` | `power_of_two` | control path, stream |

### No-MPS / windows-style sweep

The no-MPS path is driven by:

- [sweep_run_windows.sh](/workspaces/MLEvolve/scheduler_benchmark_test/sweep_run_windows.sh)
- or `NO_MPS=1 bash scheduler_benchmark_test/sweep_run.sh`

In no-MPS mode, optimized runs intentionally use `power_of_two` only. Binary optimization cases are skipped.

| Config ID | Runner | Mode | Backend | Probe/Search | Notes |
|---|---|---|---|---|---|
| `B1` | scheduler | `serial_basic` | `exclusive` | `off` | baseline |
| `B3` | scheduler | `serial_batch_optimized` | `exclusive` | `power_of_two` | exclusive 2^n probe |
| `W1` | scheduler | `parallel_default` | `cuda_process` | `off` | no-MPS parallel baseline |
| `T2` | scheduler | `parallel_default` | `stream` | `off` | stream baseline |
| `W4` | scheduler | `parallel_batch_optimized` | `stream` | `power_of_two` | no-MPS packed 2^n, RAM cache OFF |
| `W5` | scheduler | `parallel_batch_optimized` | `stream` | `power_of_two` | no-MPS packed 2^n, RAM cache ON |
| `W6` | scheduler | `parallel_batch_optimized` | `stream` | `power_of_two` | no-MPS packed 2^n, allow up to 3 concurrent jobs |
| `W7` | scheduler | `parallel_batch_optimized` | `stream` | `power_of_two` | no-MPS packed 2^n, allow up to 4 concurrent jobs |
| `W8` | scheduler | `parallel_batch_optimized` | `cuda_process` | `power_of_two` | no-MPS packed 2^n, `cuda_process`, allow up to 3 concurrent jobs |
| `W9` | scheduler | `parallel_batch_optimized` | `cuda_process` | `power_of_two` | no-MPS packed 2^n, `cuda_process`, allow up to 4 concurrent jobs |
| `T11` | `torch_mp` | `parallel_batch_optimized` | `stream` | `power_of_two` | control path, stream |

`W4` and `W5` are the explicit no-MPS A/B comparison for the scheduler RAM model cache:

- `W4`: same packed `stream` config with cache disabled
- `W5`: same packed `stream` config with cache enabled

`W6` and `W7` keep the same no-MPS scheduler path but raise the pack-width cap:

- `W6`: allows up to 3 concurrent packed jobs on one GPU
- `W7`: allows up to 4 concurrent packed jobs on one GPU

`W8` and `W9` are the matching `cuda_process` width-expansion cases:

- `W8`: `cuda_process` backend with pack width up to 3
- `W9`: `cuda_process` backend with pack width up to 4

### Smoke-only IDs

`smoke_run.sh` and `smoke_run_windows.sh` also use a small smoke-only ID:

| Config ID | Meaning |
|---|---|
| `C1` | serial baseline with cache warming disabled (`warm_queue_top_k=0`, `entry_capacity=0`) |

## Feature Sweep

[feature_sweep.sh](/workspaces/MLEvolve/scheduler_benchmark_test/feature_sweep.sh) adds focused checks for:

- `PRELOAD_OFF` vs `PRELOAD_ON`
- `PAR_CACHE_OFF` vs `PAR_CACHE_ON`
- `PACK3_ON` and `PACK4_ON`
- tighter vs wider binary thresholds
- local vs wider `power_of_two` windows

It uses:

- `exclusive` for the serial preload on/off control
- `cuda_process` for the parallel RAM-cache and packed-threshold checks

This keeps it runnable in more container setups without requiring MPS.

### Feature sweep IDs

| Config ID | Mode | Backend | Probe/Search | Notes |
|---|---|---|---|---|
| `PRELOAD_OFF` | `serial_basic` | `exclusive` | `off` | serial baseline, RAM cache OFF |
| `PRELOAD_ON` | `serial_basic` | `exclusive` | `off` | serial baseline, RAM cache ON |
| `PAR_CACHE_OFF` | `parallel_batch_optimized` | `cuda_process` | `power_of_two` | packed parallel, RAM cache OFF |
| `PAR_CACHE_ON` | `parallel_batch_optimized` | `cuda_process` | `power_of_two` | packed parallel, RAM cache ON |
| `PACK3_ON` | `parallel_batch_optimized` | `cuda_process` | `power_of_two` | packed parallel, RAM cache ON, allow up to 3 jobs |
| `PACK4_ON` | `parallel_batch_optimized` | `cuda_process` | `power_of_two` | packed parallel, RAM cache ON, allow up to 4 jobs |
| `BINARY_TIGHT` | `parallel_batch_optimized` | `cuda_process` | `binary` | tighter binary search window |
| `BINARY_WIDE` | `parallel_batch_optimized` | `cuda_process` | `binary` | wider binary search window |
| `POW2_LOCAL` | `parallel_batch_optimized` | `cuda_process` | `power_of_two` | local 2^n search window |
| `POW2_WIDE` | `parallel_batch_optimized` | `cuda_process` | `power_of_two` | wider 2^n search window |

Results are written to:

```text
results/scheduler_benchmark_test/feature_sweep/
```

Each `summary.json` includes:

- `cache_policy`
- `cache_stats`
- `cache_event_counts`
- `packing_policy`
- `packed_group_size_counts`
- `optimizer_thresholds`
- `batch_probe_event_counts`

## Results Paths

Main sweep:

```text
results/scheduler_benchmark_test/main_sweep/
```

No-MPS sweep:

```text
results/scheduler_benchmark_test/main_sweep_windows/
```

Feature sweep:

```text
results/scheduler_benchmark_test/feature_sweep/
```

Arch sensitivity:

```text
results/scheduler_benchmark_test/arch_sensitivity/
```

Plots:

```text
results/scheduler_benchmark_test/main_sweep/plots/
```

## Arch Sensitivity Check

[arch_sensitivity_run.sh](/workspaces/MLEvolve/scheduler_benchmark_test/arch_sensitivity_run.sh) is a brand-new short test for pairwise packing sensitivity. It compares total time for three two-job compositions:

- same exact model together
- same architecture family but different models together
- different architecture families together

Each composition is run in:

- serial baseline: `serial_basic` + `exclusive`
- packed baseline: `parallel_default` + `stream`

The packed path intentionally disables batch optimization so the result stays focused on co-location sensitivity rather than threshold tuning.

### Sensitivity IDs

| Config ID | Composition | Mode | Meaning |
|---|---|---|---|
| `SM_S` | same exact model | serial | `convnext_base` + `convnext_base`, serial |
| `SA_S` | same arch, different models | serial | `convnext_base` + `efficientnet_b3`, serial |
| `DA_S` | different arch | serial | `convnext_base` + `vit_base_patch16_224`, serial |
| `SM_P` | same exact model | packed stream | `convnext_base` + `convnext_base`, packed |
| `SA_P` | same arch, different models | packed stream | `convnext_base` + `efficientnet_b3`, packed |
| `DA_P` | different arch | packed stream | `convnext_base` + `vit_base_patch16_224`, packed |

The script writes a compact markdown report to:

```text
results/scheduler_benchmark_test/arch_sensitivity/summary.md
```

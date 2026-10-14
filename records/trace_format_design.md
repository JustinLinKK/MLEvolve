# MLEvolve Trace Format Design

## Purpose

- Record real MLEvolve agent runs → replay through scheduler `trace_simulator.py`

- Trace captures job arrivals and characteristics; scheduler decides ordering/packing

## Trace File

- Format: JSONL (one JSON object per line)

- Path: `traces/<experiment_name>.jsonl`

## Per-Job Fields

### Required by `TraceJob` (scheduler input)

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `job_id` | str | generated | Unique identifier, e.g. `"task03_step07"` |
| `release_seconds` | float | measured | Wall-clock offset from experiment start when job arrives at scheduler queue |
| `priority` | int | config | Job priority (default 0; higher = more urgent) |
| `planned_epochs` | int | extracted from code | Total epochs the training script runs |
| `validation_metrics` | list[float\|null] | measured | Per-epoch validation metric for early stopping |
| `backend_allowlist` | list[str] | config | `["cuda_process"]` for V100 (MPS ineffective) |

### Required by `TraceBatchOption` (per batch-size variant)

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `batch_size` | int | extracted from code | Training batch size |
| `memory_mb` | float | estimated or measured | Predicted VRAM usage in MiB |
| `solo_seconds` | float | measured on V100 | Total training time when run alone |
| `actual_memory_mb` | float\|null | measured | Actual peak VRAM (null if not measured) |
| `actual_solo_seconds` | float\|null | measured | Actual total time (null if not measured) |

### Metadata (not consumed by simulator, useful for analysis)

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `task_name` | str | config | MLEBench-Lite task name |
| `step_idx` | int | agent | Search step within this task |
| `agent_used` | str | agent | `"draft"`, `"improve"`, `"debug"`, etc. |
| `architecture` | str | extracted from code | Model name, e.g. `"efficientnet_b0"` |
| `family` | str | inferred | `"cnn"`, `"transformer"`, `"tabular"` |
| `n_params_M` | float | measured | Model parameter count in millions |
| `code` | str | agent | Full training script |
| `code_path` | str | filesystem | Path to saved training script |
| `llm_plan_s` | float | measured | LLM planning call duration |
| `llm_code_s` | float | measured | LLM code generation call duration |
| `llm_review_s` | float | measured | LLM review call duration |
| `exec_submit_at` | float | measured | When code was submitted to GPU |
| `exec_complete_at` | float | measured | When GPU execution finished |
| `exec_duration_s` | float | measured | `exec_complete_at - exec_submit_at` |
| `peak_vram_mib` | float | measured | Actual peak VRAM from `torch.cuda.max_memory_allocated` |
| `metric_value` | float\|null | measured | Final validation metric |
| `is_buggy` | bool | measured | Whether execution raised an exception |

## Arrival Model

- Jobs arrive per Poisson process with $\lambda = 4$ jobs/min (per CLAUDE.md)

- `release_seconds` for job $i$: cumulative sum of $\text{Exp}(1/\lambda)$ inter-arrival times

- In real MLEvolve run: `release_seconds = exec_submit_at` (when agent submits code to GPU)

## Mapping MLEvolve → TraceJob

```
TraceJob(
    job_id      = f"{task_name}_{step_idx:03d}",
    release_seconds = exec_submit_at - experiment_start_time,
    priority    = 0,
    options     = (TraceBatchOption(
        batch_size   = extracted_batch_size,
        memory_mb    = estimated_vram_mb,
        solo_seconds = exec_duration_s,
        actual_memory_mb    = peak_vram_mib,
        actual_solo_seconds = exec_duration_s,
    ),),
    backend_allowlist = ("cuda_process",),
    validation_metrics = per_epoch_metrics,
    planned_epochs = extracted_epochs,
)
```

## TraceProblem Config

```
TraceProblem(
    jobs = tuple_of_all_trace_jobs,
    memory_budget_mb = 31000,    # 31GB per CLAUDE.md
    parallel_cap = 5,
    default_slowdown = 4.0,      # V100 CNN packing ~4x at N=2
    colocation_trial_epochs = 2,
    colocation_min_gain = 1.0,   # reject if gain < 1
    early_stopping_enabled = True,
    early_stopping_patience_epochs = 5,
    starvation_timeout_seconds = 1800,
)
```

## What Gets Tested

- Job ordering: shortest-remaining-time-first vs priority vs FIFO

- Early stopping: plateau detection saves epochs

- Packing rejection: on V100, colocation gain < 1 → correct serial execution

- Queue management: Poisson arrivals overlap with running jobs

- Starvation prevention: no job waits > 1800s

## Recording Flow

- Wrap MLEvolve executor to emit JSONL lines at `exec_submit_at` and `exec_complete_at`

- Extract model architecture, batch_size, epochs from generated code via regex

- Measure `peak_vram_mib` via `torch.cuda.max_memory_allocated` in subprocess

- Collect per-epoch validation metrics from stdout/metric.json

# MLEvolve V100 Trace Run

## Goal

- Run the original MLEvolve agent on Nautilus against an MLEBench-Lite task, record a trace, and replay that trace through the scheduler

- Single V100, branch-profile scheduling only, no PerfSeer predictor

## Running Original MLEvolve Without an API Key

- The original MLEvolve lives on Nautilus at `/root/downeyflyfan/perfseer_test/exp_run` and drives its agents through `llm/__init__.py`, which dispatched only to `llm/openai.py` and `llm/gemini.py`

- Both backends require an API key, and `config/config.yaml` still carries the `enter-your-api-key` placeholder, which is what previously blocked this step

- The `claude` CLI is installed and authenticated on Nautilus, and the earlier cassava trace run drove MLEvolve the same way (claude CLI as a subprocess), so a CLI backend was added rather than a new key introduced

- `llm/claude_cli.py` implements the existing backend contract:

| Entry point | Returns |
|---|---|
| `query(system_message, user_message, func_spec, cfg, **kwargs)` | `(output, req_time, in_tokens, out_tokens, info)` |
| `generate(prompt, cfg, ..., json_schema, max_retries)` | generated text |

- When `func_spec` is supplied, the CLI is asked for a single JSON object matching the function's `json_schema`, and the parsed dict is returned in place of text, matching how the OpenAI backend surfaces a tool call

- `_provider()` in `llm/__init__.py` now routes `provider: claude_cli` to the new module; `openai` and `gemini` routing is unchanged

> Backend verification on Nautilus

| Path | Result |
|---|---|
| plain text query | returned `PONG` in 4.7 s |
| function-calling query | returned a `dict` with keys `['plan', 'epochs']`, `epochs = 3`, in 8.8 s |

- Python dependencies installed to make `run.py` import: `backoff`, `jsonschema`, `dataclasses-json`, `omegaconf`, `rich`, `humanize`, `funcy`, `google-genai`, `openai`, `coolname`, `shutup`, `black`, `genson`

## Trace Recording

- `run_traced.py` wraps `engine.executor.Interpreter.run`, so one row is appended per code execution the agent performs

- The interpreter is pinned to a single execution slot, because per-job VRAM attribution by device sampling is only valid when one job holds the GPU

- Recorded per job:

| Field | Meaning |
|---|---|
| `job_id`, `step_idx`, `node_id` | identity and position in the agent's search |
| `release_seconds` | submission time relative to run start, the scheduler's arrival time |
| `exec_submit_at`, `exec_complete_at`, `exec_duration_s` | measured execution window |
| `peak_vram_mib`, `avg_vram_mib` | NVML device memory across the window |
| `avg_sm_util_percent`, `peak_sm_util_percent` | NVML SM utilization across the window |
| `architecture`, `family`, `batch_size`, `epochs` | parsed back out of the generated script |
| `is_buggy`, `exc_type` | whether the execution failed |
| `code` | the full generated script, so the job can be re-run |

- Fields parsed from source are best-effort; `None` means the value was not stated literally and must not be read as measured

- Per-epoch validation metrics are not captured, since agent scripts report progress in arbitrary formats, so early stopping is left disabled when replaying

## Configuration

- Task: `leaf-classification` (MLEBench-Lite), 891 training rows, 192 features, 99 classes

- `scheduler.enabled=false` and `coldstart.use_coldstart=false`, so the recorded trace reflects unscheduled agent behavior and no predictor is involved

- `agent.search.parallel_search_num=1`, `CUDA_VISIBLE_DEVICES=0`

- Model: `claude-sonnet` via `provider: claude_cli` for both the code and feedback stages

## Measured Jobs

> First four non-buggy executions

| Job | Family | Batch | Duration (s) | Peak VRAM (MiB) | Avg SM (%) | Peak SM (%) |
|---|---|---|---|---|---|---|
| mlevolve_0000 | mlp | 64 | 56.4 | 669.5 | 1.9 | 11.0 |
| mlevolve_0001 | cnn | 64 | 113.8 | 1359.5 | 4.4 | 14.0 |
| mlevolve_0002 | mlp | 32 | 23.7 | 665.5 | 1.9 | 7.0 |
| mlevolve_0003 | mlp | — | — | 669.5 | 4.7 | 11.0 |

## Finding: leaf-classification Does Not Fill V100 Compute

- Real agent jobs on this task sustain 1.9-4.7 % average SM and peak at 7-14 %

- Packing 4-5 of them would reach roughly 8-20 % SM, so this task does not satisfy the requirement that 4-5 concurrent jobs fill the device

- Average SM is taken across the whole execution window, which includes data loading and preprocessing, so it understates the training phase; the 7-14 % peaks bound the training phase and are still far below saturation

- The cause is dataset size, not model family. 891 training rows give too little work per epoch to occupy 80 SMs, regardless of what the agent builds

- This is consistent with the standalone sweep in [v100_tabular_packing.md](v100_tabular_packing.md): reaching the packable-and-filling regime needed roughly 23 % solo SM, which came from tabular-playground data of 900k-4M rows

- Within MLEBench-Lite, the tasks that meet the requirement are therefore the tabular-playground ones, not the small tasks

| Task | Train rows | Solo SM | 4-5 jobs fill device |
|---|---|---|---|
| tabular-playground-series-dec-2021 | 4,000,000 | 23 % (w256_d2_b1024) | yes, 99 % at N=4 |
| tabular-playground-series-may-2022 | 900,000 | same configuration | yes |
| leaf-classification | 891 | 1.9-4.7 % measured | no |

## Status

- MLEvolve is running on Nautilus and recording; the trace grows one row per agent execution

- Remaining: replay the completed trace through the scheduler, and measure N-way packing of the agent's actual generated scripts with `scheduler_benchmark_test/measure_script_packing.py`

## Reproduction

```bash
ssh Nautilus
```

```bash
cd /root/downeyflyfan/perfseer_test/exp_run && MLEVOLVE_TRACE_PATH=/root/downeyflyfan/mlevolve_runs/leaf_trace_v1.jsonl CUDA_VISIBLE_DEVICES=0 python3 -u run_traced.py data_dir=/root/downeyflyfan/mle-bench-data/leaf-classification/prepared/public desc_file=/root/downeyflyfan/mle-bench-data/leaf-classification/prepared/public/description.md exp_name=leaf_trace_v1 log_dir=/root/downeyflyfan/mlevolve_runs/leaf_trace_v1/logs workspace_dir=/root/downeyflyfan/mlevolve_runs/leaf_trace_v1/ws copy_data=True scheduler.enabled=false coldstart.use_coldstart=false agent.steps=14 agent.initial_drafts=3 agent.code.model=claude-sonnet agent.code.provider=claude_cli agent.feedback.model=claude-sonnet agent.feedback.provider=claude_cli agent.search.parallel_search_num=1
```

```bash
python3 -m scheduler_benchmark_test.load_mlevolve_trace traces/mlevolve_leaf_v100.jsonl
```

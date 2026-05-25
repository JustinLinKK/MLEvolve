# Baseline vs Hardware-Aware Mode Comparison

This report compares MLEvolve's `baseline` and `hardware_aware` experiment modes based on the current implementation. The main difference is not a separate static system prompt. Instead, hardware-aware behavior is assembled at runtime from configuration, scheduler attachment, hardware/profile context retrieval, and prompt sections injected into stage agents.

## Executive Summary

| Area | `baseline` mode | `hardware_aware` mode |
| --- | --- | --- |
| Experiment switch | `experiment.mode=baseline` | `experiment.mode=hardware_aware` |
| Hardware prompt context | Disabled | Enabled when scheduler client is attached |
| Training prompt additions | No hardware-specific additions | Adds hardware-aware data/model/training guidance |
| Scheduler evidence in prompts | Not injected | Injects graph evidence, vector/code evidence, recommendations, risk flags, and confidence |
| Agent behavior goal | Solve task from task text, data preview, memory, and execution feedback | Solve task while considering hardware limits, runtime risk, OOM risk, precision, batch sizing, and profile evidence |
| Scheduler execution | Enabled when config leaves `scheduler.enabled=true`; comparison helper keeps it enabled by default | Typically enabled |
| Best for | Clean comparison with original MLEvolve behavior | Hardware-constrained runs, GPU-sensitive training, profiling-informed optimization |

## Where The Mode Is Selected

The mode is selected through config or CLI overrides:

```bash
python run.py experiment.mode=baseline
python run.py experiment.mode=hardware_aware
```

The config schema defines `agent.hardware_context_enabled` with a default of `true` in `config/__init__.py`.

When `experiment.mode=baseline`, config preparation forcibly disables hardware context:

```python
cfg.experiment.mode = normalize_experiment_mode(cfg.experiment.mode)
if cfg.experiment.mode == EXPERIMENT_MODE_BASELINE:
    cfg.agent.hardware_context_enabled = False
```

Source: `config/__init__.py`.

## Prompt Construction Locations

The core training construction prompt is in:

```text
agents/coder/stepwise_coder.py
```

The relevant function is:

```python
create_default_step_agents(*, hardware_aware: bool = True)
```

It creates three step agents:

| Step agent | Responsibility |
| --- | --- |
| `data_processing_and_feature_engineering` | Load data, clean it, build features, create splits |
| `model_design` | Define model architecture, loss, optimizer |
| `training_evaluation` | Implement training loop, validation metric, best model saving, test inference, submission generation |

The `training_evaluation` base prompt exists in both modes. In hardware-aware mode, additional hardware-specific guidance is appended.

## Baseline Training Prompt Behavior

In baseline mode, the `training_evaluation` agent receives the normal training instructions:

- Use data/features/model/loss/optimizer from earlier steps.
- Do not redefine earlier components.
- Implement validation and test inference consistently.
- Prevent overfitting.
- Implement the exact metric from the task description.
- Save `submission.csv`.
- Print the final validation score in the expected parser format.

Baseline mode does not include:

- The `# Hardware/Profile Optimization Context` section.
- Hardware-aware data pipeline guidance.
- Hardware-aware model design guidance.
- Hardware-aware training guidance.
- Scheduler recommendations, risk flags, graph evidence, or vector evidence in the prompt.

The stepwise coder explicitly disables hardware reasoning in baseline mode:

```python
mode = str(getattr(experiment, "mode", "") or "").strip().lower().replace("-", "_")
if mode == "baseline":
    return False
```

Source: `agents/coder/stepwise_coder.py`.

## Hardware-Aware Training Prompt Behavior

In hardware-aware mode, the same base training prompt is used, but extra guidance is appended to the data, model, and training step agents.

For the data step, the agent is told to keep input size configurable and use hardware-compatible data loading choices such as pinned memory and non-blocking CUDA transfers when appropriate.

For the model step, the agent is told to consider model size, precision compatibility, and tensor-core-friendly PyTorch paths when evidence supports AMP, bf16, or fp16.

For the training step, the agent is told to use hardware/profile context to choose:

- Physical batch size
- Epoch budget
- AMP dtype
- Gradient accumulation
- Checkpoint cadence
- Dataloader settings
- OOM and timeout fallback behavior
- Runtime and peak CUDA memory logging

The hardware-aware additions are in `agents/coder/stepwise_coder.py` inside `create_default_step_agents(...)`.

## Dynamic Hardware/Profile Context

The dynamic prompt block is built in:

```text
agents/hardware_context.py
```

The prompt section heading is:

```markdown
# Hardware/Profile Optimization Context
```

This section can include:

| Prompt field | Meaning |
| --- | --- |
| Hardware summary | Current GPU/hardware profile and toolkit context |
| Scheduler limits | Safe VRAM budget, packing limits, backend mode |
| Diagnosis | Derived symptoms and optimization targets |
| Recommendations | Actionable hardware/profile suggestions |
| Risk flags | OOM, timeout, batch-size, backend, or profile-risk notes |
| Graph evidence | Runtime profiles, batch probes, solo profiles, packed profiles |
| Code knowledge | Retrieved recipes, docs, and API guidance from vector search |
| Evidence refs | Traceable references to graph/vector evidence |
| Confidence | Aggregate confidence score |

The core rule injected into hardware-aware prompts is:

```text
Treat recommendations as empirical evidence, not hard rules. Follow high-confidence hardware/profile guidance by default; if a scoring reason requires deviating, state why and include a fallback such as smaller physical batch size, gradient accumulation, AMP, reduced resolution, fewer epochs, or checkpointing.
```

Source: `agents/hardware_context.py`.

## How The Context Is Retrieved

The hardware context helper first checks whether hardware context is enabled. It returns an empty context if:

- The run is in `baseline` mode.
- `agent.hardware_context_enabled` is false.
- No scheduler client is attached.
- Context lookup fails.

When enabled, it builds a candidate from the current stage and code using script introspection, then calls:

```python
scheduler_client.get_optimization_context(candidate=candidate, limit=limit)
```

That context is compacted and formatted before being inserted into prompts.

## Scheduler Attachment

`run.py` attaches the scheduler client when:

```python
scheduler.enabled=true
```

This is separate from `experiment.mode`.

Important distinction:

- `baseline` mode disables hardware prompt/context injection.
- It does not disable the scheduler execution bridge if the config still has `scheduler.enabled=true`.
- The comparison script keeps scheduler execution enabled for baseline by default so both modes use the same round-level execution flow; use `--disable-scheduler-baseline` only for legacy subprocess-only comparisons.

Source: `run.py`.

## Other Stage Agents

Hardware context is not limited to the stepwise training prompt. In hardware-aware mode, several other agents can receive the same section:

| Agent file | Hardware-aware usage |
| --- | --- |
| `agents/improve_agent.py` | Adds hardware/profile context when improving a parent solution |
| `agents/fusion_agent.py` | Adds context for solution fusion |
| `agents/evolution_agent.py` | Adds context for trajectory/evolution reasoning |
| `agents/code_review_agent.py` | Adds hardware-critical review guidance |
| `agents/planner/base_planner.py` | Adds hardware/profile context to planning suffixes |
| `agents/planner/planner_with_memory.py` | Includes hardware section in memory-aware planning |

The code review path is intentionally conservative: it should flag concrete hardware-critical risks only when profile evidence is strong, and it should not redesign the model just to satisfy hardware guidance.

## Practical Behavioral Difference

### Baseline Mode

Baseline mode behaves like the original MLEvolve prompt flow:

1. Read task description.
2. Preview data.
3. Generate code from task, memory, and prior execution feedback.
4. Validate submission format and metric.
5. Improve through search, debugging, fusion, and memory.

It does not ask the agent to reason explicitly about:

- Physical batch size vs effective batch size
- Mixed precision choice
- Tensor Core usage
- CUDA memory budget
- Dataloader throughput
- Checkpoint cadence for recoverability
- OOM fallback paths
- Scheduler profile evidence

### Hardware-Aware Mode

Hardware-aware mode keeps the same search architecture but augments it:

1. Inspect generated or parent code for training hints.
2. Build a hardware candidate.
3. Query scheduler graph and vector/code knowledge.
4. Format a compact hardware/profile section.
5. Insert that section into stage prompts.
6. Ask the agent to treat evidence as guidance, not a hard constraint.
7. Store compact evidence on search nodes for later reasoning.

The expected result is not simply "smaller model" or "safer code." The intended behavior is more nuanced:

- Start from hardware-compatible defaults.
- Use AMP or bf16/fp16 when supported and beneficial.
- Prefer configurable batch size and accumulation.
- Avoid fixed oversized batch sizes.
- Include fallback paths for risky choices.
- Log enough runtime/memory evidence for future scheduler learning.
- Override hardware guidance when leaderboard score justifies it, but explain the tradeoff.

## How To Compare The Modes

Use the comparison helper:

```bash
bash compare_hardware_awareness.sh dogs-vs-cats-redux-kernels-edition \
  --dataset-root /mle-bench/data \
  --steps 5
```

For quick hardware-aware debugging, `dogs-vs-cats-redux-kernels-edition` is the preferred small image task. It is much lighter than `histopathologic-cancer-detection` while still exercising GPU training. Use `plant-pathology-2020-fgvc7` as the next-small backup, and keep `histopathologic-cancer-detection` for heavier acceptance runs.

It prepares the Kaggle/MLE-bench competition, starts validation, and runs:

```bash
experiment.mode=baseline
experiment.mode=hardware_aware
```

By default, the helper does not impose a whole-mode wall-clock timeout. If you need a hard cap for a cluster job, pass `--timeout-seconds N` explicitly or set `MLEVOLVE_TIMEOUT_SECONDS=N` when using `bootstrap.sh`.

By default it also generates matplotlib comparison artifacts under:

```text
<run-root>/comparison_plots/
```

Those artifacts include `comparison_metrics.png`, `utilization_timeseries.png`, `comparison_summary.json`, and `comparison_summary.md`.

The baseline run keeps `scheduler.enabled` unchanged by default so baseline prompts run without hardware context while still using scheduler-managed round execution. Use this flag only if you intentionally want the old subprocess-only baseline:

```bash
--disable-scheduler-baseline
```

## What To Inspect In Outputs

After both runs, compare:

| Artifact | What to look for |
| --- | --- |
| `logs/config.yaml` | Confirm `experiment.mode` and `agent.hardware_context_enabled` |
| `logs/journal.json` | Search nodes, plans, code, hardware evidence fields |
| `logs/best_solution.py` | Batch size, precision, dataloader, checkpoint, fallback differences |
| `stdout.log` | Scheduler attachment, runtime failures, OOM/timeout symptoms |
| `workspace/top_solution/` | Best candidates and submissions |
| `localml_scheduler/runtime` or per-run scheduler runtime | Profile/cache/evidence written by scheduler |
| `comparison_plots/comparison_metrics.png` | Side-by-side runtime, search, utilization, scheduler, and failure metrics |
| `comparison_plots/utilization_timeseries.png` | CPU, RAM, GPU, and GPU memory utilization over elapsed time |

## Summary

The two modes share the same core search engine and base coding prompts. The difference is that `hardware_aware` mode adds both static hardware-aware guidance and dynamic scheduler-derived context to the prompts. Baseline mode suppresses those prompt additions, giving a cleaner task/data/memory/execution-feedback run.

The most important implementation distinction is:

```text
baseline = no hardware prompt/context
hardware_aware = hardware prompt/context plus scheduler evidence when available
```

Scheduler execution is controlled separately by `scheduler.enabled`, so a baseline run can still use scheduler-managed execution unless the runner or CLI overrides it to false.

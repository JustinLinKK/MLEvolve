# Pipeline Stage Prompt Contract

MLEvolve now creates a compact pipeline decision before every code-producing
stage. The decision enforces this order:

```text
datatype -> model -> optimizer -> tuning
```

`datatype` means data modality and target shape, not floating-point precision.
Numeric precision such as fp32, fp16, bf16, and tf32 belongs under `tuning`
unless the task itself requires a specific numeric type.

## Decision Schema

Each generated node can store `SearchNode.pipeline_decision`:

```json
{
  "datatype": {
    "modality": "image|tabular|text|audio|graph|time_series|mixed|unknown",
    "target_type": "classification|regression|segmentation|reconstruction|ranking|sequence|unknown",
    "shape_constraints": ["short concrete constraints"],
    "reason": "why this datatype interpretation is correct"
  },
  "model": {
    "family": "chosen model family",
    "alternatives_considered": ["other reasonable families"],
    "reason": "why this model follows from the datatype and metric",
    "hardware_fit": "how hardware evidence affects the choice, or none"
  },
  "optimizer": {
    "loss": "chosen loss or metric proxy",
    "optimizer": "chosen optimizer",
    "scheduler": "optional scheduler",
    "reason": "why this optimizer/loss is suitable",
    "advanced_optimizer_used": false
  },
  "tuning": {
    "batch_size_policy": "fixed|scheduler_recommended|adaptive",
    "precision_policy": "fp32|tf32|fp16_amp|bf16_amp|disabled",
    "dataloader_policy": "num_workers/pin_memory/non_blocking choices",
    "fallbacks": ["OOM fallback", "timeout fallback"],
    "metrics_to_log": ["elapsed_seconds", "peak_vram_mb", "resolved_batch_size"]
  },
  "evidence": {
    "hardware_context_used": true,
    "evidence_refs": [],
    "confidence": 0.0,
    "missing_evidence": ["predictor/graph evidence not available"]
  }
}
```

## Prompt Integration

The contract is injected into:

- draft prompts
- improve prompts
- debug prompts
- evolution prompts
- fusion and multi-reference fusion prompts
- root-level aggregation prompts
- direct diff planners
- memory-enhanced planners
- stepwise data, model, and training sub-agent prompts

Stepwise agents consume different parts of the same trace:

- `data_processing_and_feature_engineering`: `datatype` and dataloader policy
- `model_design`: `datatype`, `model`, and optimizer/loss choices
- `training_evaluation`: `optimizer`, `tuning`, and `evidence`

## Evidence Rules

Hardware/profile context may influence tuning only when actual evidence is
available and compatible with the task. Missing predictor or graph evidence is
stored in `evidence.missing_evidence` and forces a fixed batch-size policy.
When no usable hardware evidence exists, precision fast paths are disabled too.

The normalizer strips hallucinated `evidence_refs`; only references present in
the actual hardware/profile context can be persisted.

Hardware tuning must not increase epochs, folds, model size, image resolution,
ensemble count, TTA, dataset size, or validation workload as a hardware-only
optimization.

## Configuration And Evaluation

The contract is enabled by default:

```yaml
agent:
  pipeline_decision_enabled: true
```

Set `agent.pipeline_decision_enabled=false` to restore the previous prompt flow
without the extra structured decision call. This switch supports controlled
A/B evaluation with the same branch, task, seed, model, and execution settings.

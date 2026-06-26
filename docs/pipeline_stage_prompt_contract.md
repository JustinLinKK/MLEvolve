# Pipeline Stage Prompt Contract

MLEvolve now creates a compact pipeline decision before every code-producing
stage. The hardware-aware stepwise workflow follows this order:

```text
model_design -> datatype_precision -> training_evaluation
```

The persisted decision stores exactly these top-level stage keys:
`model_design`, `datatype_precision`, `training_evaluation`, and `evidence`.
Older traces with `datatype`, `model`, `optimizer`, and `tuning` are normalized
into this three-stage view for compatibility, but new prompts and metadata must
not emit the old four-stage contract.

`model_design` owns data modality, target shape, preprocessing/features,
model family, loss, criterion, and output interface. Numeric precision such as
fp32, fp16, bf16, tf32, and Transformer Engine FP8/MXFP8/NVFP4 belongs to the
`datatype_precision` stage unless the task itself requires a specific numeric
type. The `datatype_precision` stage may also make narrow precision-required
model adaptations, such as TE-compatible layer wrappers, precision shape
padding/config hooks, autocast recipes, or higher-precision islands, while
preserving the Stage 1 model family, loss, data features, and output interface.

## Decision Schema

Each generated node can store `SearchNode.pipeline_decision`:

```json
{
  "model_design": {
    "modality": "image|tabular|text|audio|graph|time_series|mixed|unknown",
    "target_type": "classification|regression|segmentation|reconstruction|ranking|sequence|unknown",
    "shape_constraints": ["short concrete constraints"],
    "family": "chosen model family",
    "alternatives_considered": ["other reasonable families"],
    "loss": "chosen loss or metric proxy",
    "output_interface": "prediction/output shape required by the metric and submission",
    "reason": "why this model follows from the datatype and metric",
    "hardware_fit": "how hardware evidence affects the choice, or none"
  },
  "datatype_precision": {
    "precision_policy": "fp32|tf32|fp16_amp|bf16_amp|fp8_te|mxfp8_te|nvfp4_te|disabled",
    "precision_model_adaptation": "none|short description of precision-required adapter",
    "fallback_policy": "safe fallback if precision path is unsupported",
    "reason": "why the precision policy is suitable"
  },
  "training_evaluation": {
    "optimizer": "chosen optimizer",
    "scheduler": "optional scheduler",
    "batch_size_policy": "fixed|scheduler_recommended|adaptive",
    "dataloader_policy": "num_workers/pin_memory/non_blocking choices",
    "fallbacks": ["OOM fallback", "timeout fallback"],
    "metrics_to_log": ["elapsed_seconds", "peak_vram_mb", "resolved_batch_size"],
    "advanced_optimizer_used": false,
    "reason": "why this training plan is suitable"
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
- stepwise model-design, datatype/quantization, and training sub-agent prompts

Stepwise agents consume different parts of the same trace:

- `model_design`: `model_design.modality`, `target_type`,
  `shape_constraints`, `family`, `loss`, `output_interface`, and compatible
  data/feature implications
- `datatype_precision`: `datatype_precision.precision_policy`,
  `precision_model_adaptation`, `fallback_policy`, and precision evidence
- `training_evaluation`: `training_evaluation.optimizer`, `scheduler`,
  `batch_size_policy`, `dataloader_policy`, `fallbacks`, metrics, and
  `evidence`

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

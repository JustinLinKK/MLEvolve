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

The precision list is an architecture- and mode-dependent allowlist, not a
menu that is valid on every GPU. Normal mode stops at 16-bit training formats;
aggressive mode can additionally permit validated Transformer Engine formats.
Aggressive permission is opt-in and is not an automatic recommendation.

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

For datatype optimization, evidence must show native Tensor Core operations, a
supported forward/backward training path, and an MLEvolve implementation and
validation path. The resulting matrix is:

- Volta and Turing: FP16 AMP.
- Ampere: TF32, BF16 AMP, and FP16 AMP.
- Ada and Hopper: the normal formats above; aggressive mode may add Transformer
  Engine FP8 E4M3 or HYBRID.
- Blackwell: the normal formats above; aggressive mode may add FP8, MXFP8, and
  NVFP4 through Transformer Engine.

Ada FP8 support starts at SM 8.9. Pure E5M2 is not a policy; E5M2 may appear
only as the backward part of the documented HYBRID recipe. FP6 is explicitly
excluded because low-level PTX instruction support is not an end-to-end
training recipe. Generic FP4 is not NVFP4. Integer formats remain capability
indicators rather than general training recommendations, and FP64 remains
queryable but hidden from datatype-speed optimization.

The normalizer converts a format outside the hardware/mode allowlist to
`disabled` and records an FP32 fallback. Code Review produces a critical
`datatype_precision` issue for a disallowed path. A final deterministic guard
runs immediately before execution, including when review is disabled, an
initial solution skips review, or the reviewer is unavailable.

Large workflow agents receive role-filtered hardware context. Draft sees model
options and one comparable success; Improve sees the diagnosed bottleneck and
at most two short recipes; Debug sees the failure and repair evidence;
Evolution and Fusion see compatible alternatives; Aggregation sees at most four
successful branch summaries; Code Review sees only the exact precision/backend
constraints needed to classify code. Only Improve and Debug receive small code
examples. Full context remains attached to node telemetry, while each prompt is
bounded by `agent.hardware_context_max_prompt_chars`.

## Configuration And Evaluation

The contract is enabled by default:

```yaml
agent:
  pipeline_decision_enabled: true
  precision_optimization_mode: normal  # conservative | normal | aggressive
  hardware_context_mode: full         # full | compact
```

Set `agent.pipeline_decision_enabled=false` to restore the previous prompt flow
without the extra structured decision call. This switch supports controlled
A/B evaluation with the same branch, task, seed, model, and execution settings.

`agent.precision_optimization_mode=conservative` requires float32 candidate
models, floating inputs, optimizer state, training, validation and inference.
AMP, GradScaler, lower-precision/quantized model paths and explicit FP64 are
rejected. Integer labels and indices remain valid. TF32 is permitted only with
confirmed Ampere-or-newer hardware; V100/T4 and unknown hardware use FP32.
TF32 retains float32 tensors but uses reduced internal mantissa precision.
This controls generated candidates, not the precision of the agent's LLM.
The policy applies to origin/baseline modes too, independently of hardware
context, pipeline decisions, code review and preflight availability. A configured
local preflight target profile can establish hardware when the graph is absent.
The conservative static guard checks explicit Python precision operations and
declarations, including inherited lower-precision fallback code; it is not a
runtime proof of arbitrary third-party or dynamically generated code.

`agent.hardware_context_mode=compact` independently selects a short evidence
view (at most 1,000 characters, or the smaller configured prompt character
budget). It retains hardware/precision/backend constraints, then at most two
critical recommended patterns and two avoid patterns from the relevant stage,
within a shared 250-character pattern budget.
Patterns must describe failure prevention or explicit restrictions; feature-level
patterns also require verified evidence. Optimizer advice, generic speed recipes,
profile/symptom dumps, references and patterns conflicting with the precision
policy or a known target compute capability are omitted. Long patterns are
omitted rather than truncated, preserving
their conditions. Shared hardware instructions are shortened, and the feature
catalog and feature-selection LLM call are skipped.
The pipeline decision receives the rendered view instead of the full
hardware JSON, and merge receives one bounded view rather than three joined
stage views. Full evidence remains in node diagnostics. This setting does not
reduce task/code context, lesson profiles, CUDA-documentation context or alter
scheduler admission. The defaults remain `normal` and `full`.

For the implementation audit, measured prompt savings and experiment limits,
see [precision/context analysis](../records/2026-09-08_conservative_precision_context_analysis.md).

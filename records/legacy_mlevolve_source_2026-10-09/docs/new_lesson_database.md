The lesson should match the decision each agent is responsible for. All agents share the same measured evidence, but each receives a different compact interpretation.

| Agent | Lesson it needs | Example |
| --- | --- | --- |
| Draft | Known-good starting recipe | “For ResNet on A10 with 224×224 images, FP16 AMP and batch 32 completed successfully.” |
| Improve | Focused modification lesson | “Adding this CNN block improved the metric but increased VRAM by 1.4 GB.” |
| Debug | Failure-to-fix lesson | “The added block caused a channel mismatch; changing its input channels from 64 to 128 fixed it.” |
| Evolution | Branch trajectory lesson | “Increasing depth helped twice, but further depth increases stopped improving the metric.” |
| Fusion | Transfer lesson | “This attention block transferred from branch B, but requires its normalization and projection layers.” |
| Aggregation | Stable cross-branch conclusion | “Three strong branches used AMP and scheduler-selected batches; augmentation choices remained diverse.” |
| Code review | Correctness and safety lesson | “When adding this block, register its parameters with the optimizer and preserve the output shape.” |

## Draft agent

The draft agent needs a short family baseline—not detailed historical experiments.

Prepare lessons about:

- Recommended model-family starting structure.
- Precision known to work.
- Safe initial physical batch size.
- Gradient accumulation fallback.
- Input-size or sequence-length limits.
- Data-loader settings.
- Expected runtime and VRAM range.
- Common family-level failures.
- Required output and loss interface.

Example:

```yaml
lesson_type: family_baseline
agent_audiences: [draft]
lesson: >
  ResNet-family models on this A10 environment have completed successfully
  with 224x224 inputs, FP16 AMP, and physical batch 32.
recommended_start:
  precision: fp16_amp
  batch_size: 32
  num_workers: 4
fallbacks:
  - reduce batch size to 16
  - use gradient accumulation
confidence: provisional
```

The draft agent normally does not need individual layer patches, long branch histories, or unrelated debugging details.

## Improve agent

The improve agent needs lessons about a specific proposed change.

Prepare lessons containing:

- Parent structure and changed structure.
- Exact layer or training modification.
- Minimal successful patch.
- Where the change was inserted.
- Shape and dependency requirements.
- Metric difference.
- VRAM, runtime, and throughput difference.
- Whether batch size had to change.
- Confidence that the modification caused the result.

Example:

```yaml
lesson_type: modification
agent_audiences: [improve]
change: add_conv_block
location: backbone.stage3
observed_effect:
  metric: improved
  peak_vram_mb: increased
  step_time: increased
implementation_example:
  kind: minimal_patch
  code: |
    self.extra_block = nn.Sequential(
        nn.Conv2d(128, 128, 3, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
    )
warnings:
  - Re-check batch headroom.
  - Preserve the 128-channel output contract.
```

Only lessons matching the current delta should be included.

## Debug agent

The debug agent needs failure-to-fix pairs, not general success stories.

Prepare lessons containing:

- Normalized error signature.
- Layer or pipeline stage responsible.
- Root cause.
- Failed configuration.
- Verified repair.
- Minimal repair patch.
- Hardware-specific symptoms such as OOM or unsupported precision.
- Conditions under which the repair applies.
- Counterexamples that look similar but have a different cause.

Example:

```yaml
lesson_type: verified_fix
agent_audiences: [debug]
error_signature: conv2d_channel_mismatch
symptom: expected 64 channels but received 128
root_cause: >
  The extra block used the output width of stage 2 even though it was inserted
  after stage 3.
verified_fix: set in_channels to 128
warnings:
  - Do not apply this fix when the block is inserted after stage 2.
```

The debug agent should prioritize exact error and structure matches over semantic similarity.

## Evolution agent

The evolution agent reasons over the history of one branch. It needs lessons about sequences and trends.

Prepare lessons containing:

- Which modifications were attempted and in what order.
- Which changes produced repeated improvement.
- Which directions plateaued.
- Which experiments were reversed.
- Dependencies between changes.
- Resource growth across the branch.
- Previously attempted changes that should not be repeated.
- Confidence changes as evidence accumulated.

Example:

```yaml
lesson_type: branch_trajectory
agent_audiences: [evolution]
pattern: >
  Adding one CNN block improved the branch, but adding a second similar block
  increased runtime without further metric improvement.
recommended_next_direction:
  - improve regularization
  - tune the existing added block
avoid:
  - add another identical block
supporting_nodes: 3
```

This agent needs the branch story, but not detailed histories from unrelated branches.

## Fusion agent

The fusion agent needs lessons about whether a feature can be transferred from one branch to another.

Prepare lessons containing:

- Donor branch and recipient branch structures.
- What can be transferred independently.
- Required companion changes.
- Shape, precision, and framework compatibility.
- Resource cost of the transferred feature.
- Whether the idea previously transferred successfully.
- Conflicts between donor and recipient assumptions.
- Minimal transplant patch.

Example:

```yaml
lesson_type: transfer
agent_audiences: [fusion]
transferable_component: attention_pooling_head
donor_family: cnn_attention_hybrid
recipient_family: resnet
requirements:
  - recipient feature width must be 512
  - include the donor projection layer
  - preserve classifier output shape
observed_result: successful_once
confidence: low
```

A useful fusion lesson describes the entire dependency bundle. Copying only the visible layer is often insufficient.

## Aggregation agent

The aggregation agent creates a new branch from several successful branches. It needs stable conclusions, not node-level anecdotes.

Prepare lessons containing:

- Decisions shared across multiple strong branches.
- Choices supported by independent runs.
- Score-versus-resource trade-offs.
- Known compatible combinations.
- Important disagreements between branches.
- Which dimensions should remain diverse.
- Stable hardware requirements and limits.

Example:

```yaml
lesson_type: cross_branch_consensus
agent_audiences: [aggregation]
stable_findings:
  - FP16 AMP worked across four successful branches.
  - Scheduler-selected batches were safer than fixed large batches.
  - All strong branches preserved the same evaluation interface.
diversity_to_preserve:
  - model depth
  - augmentation strategy
  - classifier-head design
conflict:
  - two branches disagreed on whether the extra CNN block was worthwhile
```

Aggregation should receive only mature or strongly supported lessons.

## Code-review agent

The reviewer needs implementation contracts and known hazards.

Prepare lessons containing:

- Required input/output shapes.
- Imports and dependencies.
- Optimizer-registration requirements.
- Precision-sensitive operations.
- Data-leakage and metric rules.
- Checkpoint and inference requirements.
- Unsafe patterns previously rejected.
- Framework and library-version constraints.

Example:

```yaml
lesson_type: implementation_contract
agent_audiences: [review]
component: extra_conv_block
checks:
  - output channels remain 128
  - block parameters are trainable
  - optimizer is created after the block
  - AMP does not force BatchNorm state to an unsupported type
  - submission inference uses the modified model
severity_if_missing: critical
```

The reviewer usually does not need metric-improvement narratives unless they affect whether a proposed repair should be preserved.

## One observation can create several lessons

Suppose an improve node adds a CNN block, initially fails, and then succeeds after repair. The profile builder could derive:

- Draft: “One extra block fits this GPU, but it is not required for the baseline.”
- Improve: “The block improved the metric and added measurable VRAM cost.”
- Debug: “The original block failed because its input channel count was wrong.”
- Evolution: “One block helped; a second block later produced no improvement.”
- Fusion: “The block can transfer only when the recipient stage also outputs 128 channels.”
- Review: “Check shape compatibility and optimizer registration.”

These are different views of the same evidence, not six independent versions of the truth.

The key rule is:

> Store evidence once, prepare lessons by purpose, and retrieve only the lessons that help the current agent make its next decision.
# Petfinder diagnostic rerun

This change makes the next experiment explain the difference between search,
hardware advice and precision. It does not establish that BF16 improves Petfinder
RMSE; that requires the controlled comparison below.

## What changed

- Deferred search waits for pending branch results when no candidate is selectable.
  A genuinely exhausted tree opens one additional draft slot, then resumes the
  existing selection policy. It no longer expands the root budget to the entire
  remaining experiment budget. GPU admission has no new parallel-job cap.
- Ampere/A100 hardware prompts recommend BF16 AMP with FP32 parameters, optimizer
  state and sensitive reductions. FP16 remains allowed with gradient scaling;
  explicit precision choices for comparisons take precedence over this preference.
- New generated PyTorch scripts use `utils.training_diagnostics.TrainingDiagnostics`
  in both ordinary and scheduled execution. The training-contract reviewer requires
  the helper and its update/report calls. Existing scripts without measurements
  are reported as missing, never as zero skipped updates.

## Rerun configuration

Use a fresh run directory and this commit on both arms. Keep the same agent model,
agent seed, initial drafts, search limits, pretrained files, data split, training
budgets and scheduler settings. `agent.precision_optimization_mode=normal` controls
the precision allowlist; it does not enable hardware knowledge.

For hardware reasoning enabled, the relevant overrides are:

```text
experiment.mode=hardware_aware
agent.hardware_context_enabled=true
agent.precision_optimization_mode=normal
scheduler.enabled=true
```

For a comparison with hardware reasoning disabled but the same scheduler mode,
keep these overrides and set `agent.hardware_context_enabled=false`. This compares
the hardware-aware generation path as a whole, including its additional precision
stage. `origin` and `baseline` deliberately disable hardware context regardless
of the context flag, and can also differ in execution/profile policy; do not label
those runs as a retrieval-only A/B.

Do not resume the old all-draft journal for this comparison: its different parents
and candidates would remain. Preserve the 31 GiB bound and branch-profile method.
Do not set `parallel_job_cap` to obtain parity.

## Files to inspect

Every journal save refreshes `logs/node_diagnostics.jsonl`, with one JSON object
per non-root candidate, including pending and rejected candidates. Each record
contains:

- Node/parent/branch IDs, action and reason, parent score, score, runtime and errors.
- Hardware-context configuration, generation-stage injection evidence (when
  available), saved-prompt presence/hash and retrieved evidence references.
  Retrieval alone is not evidence that advice reached a prompt, and prompt
  presence does not prove the model followed the advice.
- Statically inferred training settings, separately from runtime observations.
- Observed training autocast dtypes, parameter/state dtypes, learning rates,
  script-reported physical/effective batch and epoch budgets, seed and GPU name.
- Attempted, completed and skipped optimizer updates, GradScaler scale, runtime
  session ID, epoch samples and completion/error status. Missing runtime evidence
  is `null`; malformed diagnostic lines are counted.
- `diagnostic_flags` highlights missing prompt/runtime evidence, skipped updates,
  missing update hooks, inferred-versus-observed precision differences and FP16
  without gradient scaling. TF32 configuration flags are also recorded.

`logs/pipeline.sqlite3` retains existing tables and adds events/payloads:

- `run_started`: source revision/dirty flag, initial draft settings, seed, node
  budget, precision mode and hardware-context configuration.
- `candidate_action_selected`: why a child was drafted, improved, debugged,
  evolved or fused; join by node ID to existing `tree_node_selected` events for
  UCT/Top-K selection details.
- `search_waiting_for_feedback` and `search_draft_budget_expanded`: distinguish a
  temporary lack of selectable work from genuine search exhaustion.

Full prompts and raw runtime samples remain in the existing journal, prompt
snapshots and execution output. Archive those files along with `config.yaml` and
the generated scripts. A source revision with `dirty=true` is not a pinned run.

## Precision isolation

Before another search comparison, reuse one frozen generated candidate in three
arms: FP32, BF16 autocast, and FP16 autocast with GradScaler. Keep initialization,
data order, physical/effective batch, LR schedule and attempted-update budget
identical, and keep loss/metric reductions in FP32. Disable TF32 in the strict
FP32 reference. Change the precision configuration only, without regenerating
the architecture. Do a short numerical check first, then compare equal training
budgets and multiple fixed seeds for any quality claim.

Compare `autocast_dtypes_observed`, `completed_updates`, `skipped_updates`, scale
history, validation curves and elapsed time. A finite final score alone cannot
rule out skipped updates. After each experiment, refresh the repository's combined
Gantt-above-metric-versus-node PNG with all comparison results.

The helper uses PyTorch optimizer post-step hooks and observes AMP overflow flags
for fused optimizers, without synchronizing CUDA every update. Reporting and
checkpoint boundaries collect scalar evidence. Instantiate it after model/optimizer
creation, call `after_update()` after every complete update attempt, and save/restore
its `state_dict()` with the training checkpoint. Use one instance per optimizer;
multiple folds/optimizers have distinct session IDs, so inspect `runtime_samples`
instead of treating the last stream as an aggregate. The helper observes autocast
at the model entry point; custom internal precision changes/kernels need their
own evidence. See the [PyTorch AMP recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
and [optimizer hook API](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.register_step_post_hook.html).

## Local verification

Focused tests include real CPU GradScaler overflows for ordinary and fused AdamW,
unchanged training updates, diagnostic checkpoint continuation, both subprocess
execution paths, source-versus-runtime precision differences, missing hardware
injection evidence and deferred branch feedback. These are mechanism tests; the
Petfinder A100 quality comparison remains to be run by the experiment owner.

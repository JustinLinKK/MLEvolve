"""Canonical scheduler-ready training-script instructions for code agents."""

from __future__ import annotations


def elastic_training_contract_guidelines() -> list[str]:
    """Return one shared contract for generation, merge, and review prompts."""

    return [
        "**Mandatory adaptive-scheduler script contract (P0; validate before returning code):**",
        "Scheduler-managed trainable candidates must use a checkpointable PyTorch model and optimizer that implement `state_dict()` and `load_state_dict()`. Do not choose sklearn, XGBoost, LightGBM, CatBoost, or another `.fit()`-only trainer for an elastic scheduler job; use an equivalent PyTorch model for tabular/text/image data.",
        "Define exactly one top-level `MODEL_BRANCH = \"canonical-architecture-name\"` string and exactly one top-level integer literal such as `batch_size = 32`, chosen from 1, 2, 4, 8, 16, 32, 64, .... Never read the authored batch from an environment variable and never overwrite it. Use `session.batch_size` everywhere the physical training batch is needed.",
        "Define `epochs` as a positive top-level integer literal so the scheduler can estimate the job. Scheduler/probe limits may shorten execution externally; do not read scheduler-owned batch or epoch overrides yourself.",
        "Import with `from localml_scheduler.elastic import ElasticTrainingSession` and create `session = ElasticTrainingSession.from_env()` inside the executable path.",
        "Build the training loader only with `session.make_dataloader(train_dataset, ...)`. Never pass `batch_size=` or a non-resumable custom sampler; the session owns both. Ordinary `DataLoader` is allowed only for validation and test loaders.",
        "After constructing the model, optimizer, LR scheduler, and GradScaler, call `session.register_training_state(model, optimizer, lr_scheduler=lr_scheduler, scaler=scaler, extra_state=..., extra_state_loader=...)`. Pass `None` for components that do not exist. Mutable extra state must use a callable plus a loader so resume updates the live object.",
        "Call `progress = session.restore_if_present()` after registration and before training. Initialize `global_step` from `progress['global_step']`, begin the epoch loop at `progress['epoch']`, and preserve resumed sampler/batch progress rather than restarting or reshuffling consumed samples.",
        "A safe point is legal only after a complete optimizer update: backward/accumulation -> optimizer or scaler step -> scaler update -> optional LR scheduler step -> increment `global_step` -> `session.optimizer_step_completed(...)`. Put the callback inside the same accumulation condition as the optimizer step, never once per microbatch.",
        "Use the exact runtime API names: `session.optimizer_step_completed(samples=samples_since_update, epoch=epoch, batch_index=batch_index, global_step=global_step, metrics={'loss': float(loss.item())})`. The keyword is `batch_index`, never `batch_idx`; do not invent, abbreviate, or rename API keywords.",
        "Pass the actual samples consumed by that optimizer update, the current epoch and batch index, the monotonic global step, and numeric metrics to `optimizer_step_completed`. Do not catch or suppress pause, cancellation, early-stop, or probe `SystemExit` signals from the session.",
        "Keep all executable work in `main()` guarded by `if __name__ == \"__main__\": main()` so probing starts from a clean process and DataLoader workers are safe.",
        "Before returning code, mentally run the lifecycle in this order: literal identity -> session -> elastic train loader -> model/optimizer state registration -> restore -> completed optimizer step -> safe-point callback -> validation/test inference -> submission.",
    ]


def elastic_contract_review_guidelines() -> list[str]:
    """Return repair-specific guidance for the code-review agent."""

    return [
        "Treat every scheduler-submission or elastic-contract validator finding as P0 and set `needs_revision=true`.",
        "Preserve the selected model, preprocessing, loss, and metric while repairing lifecycle wiring; scheduler-contract repair is not permission to redesign the solution.",
        "The revised code must itself pass the validator. Adding method names in comments, dead code, or unused helper functions does not satisfy the contract.",
        *elastic_training_contract_guidelines(),
    ]

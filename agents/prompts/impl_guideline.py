"""Implementation guideline."""

import time

import humanize


def get_impl_guideline_from_agent(agent):
    """Build implementation guideline from agent config."""
    tot_time_remaining = agent.acfg.time_limit - (time.time() - agent.start_time)
    configured_timeout = getattr(getattr(agent.cfg, "exec", None), "timeout", None)
    if configured_timeout is None:
        exec_timeout = int(max(0, tot_time_remaining))
    else:
        exec_timeout = int(max(0, min(float(configured_timeout), tot_time_remaining)))
    return get_impl_guideline(
        tot_time_remaining=tot_time_remaining,
        steps_remaining=agent.acfg.steps - agent.current_step,
        exec_timeout=exec_timeout,
        expose_prediction=getattr(agent.acfg, "expose_prediction", False),
        k_fold_validation=getattr(agent.acfg, "k_fold_validation", 0),
        pretrain_model_dir=getattr(agent.cfg, "pretrain_model_dir", ""),
    )


def _format_time(time_in_sec):
    """Format seconds for display."""
    return f"{int(time_in_sec) // 3600}h {(int(time_in_sec) % 3600) // 60}m {int(time_in_sec) % 60}s"


def get_impl_guideline(
    tot_time_remaining: float,
    steps_remaining: int,
    exec_timeout: int,
    expose_prediction: bool = False,
    k_fold_validation: int = 0,
    pretrain_model_dir: str = "",
) -> dict:
    """Build implementation guideline from time and config."""
    impl_guideline = [
        f"**Resource Budget**: Time left ≈ {_format_time(tot_time_remaining)} | Steps left = {steps_remaining} | Max execution time per run = {humanize.naturaldelta(exec_timeout)}",
        "",
        "**Note:** Code execution MUST complete within 9 hours (hard limit) — any solution exceeding this will be invalid. Within this constraint, prioritize performance and optimization.",
        "🎯 **CRITICAL REQUIREMENTS** (Non-Negotiable):",
        "",
        "**1. Model Inference for ALL Predictions**",
        "• EVERY prediction (validation & test) MUST come from trained model's forward pass",
        "• Process: Load data → Preprocess → model.predict()/model.forward() → Save predictions",
        "• ❌ FORBIDDEN: Constants, placeholders, dummy values, empty arrays, statistics, random numbers",
        "• ❌ FORBIDDEN: Fake/mock metric functions (must use real sklearn.metrics or correct manual implementation)",
        "• Why: Shortcuts create fake high validation scores but fail on test (CRITICAL SYSTEM FAILURE)",
        "",
        "**2. Generate submission.csv**",
        "• Path: `./submission/submission.csv` (NOT ./working/submission.csv)",
        "• Content: Model predictions on ALL test samples",
        "• Format: Follow task description exactly",
        "",
        "**3. Print Validation Metric**",
        "• MUST print: `print(f'Final Validation Score: {score}')`",
        "• Score MUST be computed on hold-out validation set using proper metric formula",
        "• CRITICAL CONSISTENCY REQUIREMENT: Ensure that validation and test inference use IDENTICAL processing logic. Any differences in how validation and test data are handled (such as post-processing, reconstruction, or formatting) can cause large performance gaps between validation and test sets. Maintain consistency across all data processing steps for both validation and test phases.",
        "",
        "**4. Scheduler Model Family Contract**",
        "• MUST define a top-level constant `MODEL_BRANCH = \"mother_model_name\"` near the top of the file, e.g. `MODEL_BRANCH = \"resnet50\"`.",
        "• `MODEL_FAMILY` is accepted only as a legacy alias; prefer `MODEL_BRANCH` for scheduler profile reuse.",
        "• Use a stable, architecture-specific name, for example `resnet50_224`, `swin_b_384`, or `lightgbm_tabular_v1`.",
        "• If you switch to a different mother model in an improvement/evolution, update `MODEL_BRANCH` to that canonical branch.",
        "• If there is no more specific model/backbone name in the script, set the model name/key variable to the same value as `MODEL_BRANCH`.",
        "",
        "📁 **Directories**: Input data is read-only under `./input/`, submissions go in `./submission/`, and all temp/extracted/cache files go in `./working/`.",
        "• Inspect the data preview and actual path type before use: `train.zip` is an archive, not `train/`; `train.csv` is a file, not a directory.",
        "• Never create, overwrite, or extract files inside `./input/`. If a split is zipped, use Python `zipfile` to extract it into `./working/<split_name>/`, then inspect the extracted layout; archives may store files directly at the root or inside a nested folder. Use `.exists()` plus `rglob`/fallback checks before assuming `./working/<split_name>/<split_name>/` or any fixed child directory.",
        "• Use `pathlib.Path` plus `.exists()`, `.is_file()`, and `.is_dir()` checks before `glob`, `iterdir`, `os.listdir`, or train/validation splitting.",
        "",
        f"📦 **Packages & Internet**: Prefer numpy, pandas, sklearn, torch, torchvision, transformers, timm, xgboost, lightgbm, OpenCV, Pillow, and the Python standard library. Optional packages may be missing; do not rely on `pip install` from the solution script. torch.hub.load(), HuggingFace, etc. are available during development when configured."
        + (f" Offline models at `{pretrain_model_dir}`" if pretrain_model_dir else ""),
        "",
        "⚠️ **API Compatibility**: LightGBM/XGBoost: ❌ `fit(..., early_stopping_rounds=...)` → ✅ LightGBM: `fit(..., callbacks=[lgb.early_stopping(...)])` ✅ XGBoost: set `early_stopping_rounds` on `XGBClassifier`/`XGBRegressor` construction and pass `eval_set` to `fit()`.",
        "• PyTorch CUDA capability: use `torch.cuda.get_device_capability()`, never `torch.cuda.get_ability()`.",
        "• AdamW: ❌ `from transformers import AdamW` (deprecated) → ✅ `from torch.optim import AdamW`",
        "• Low-precision metric/export boundary: BF16/FP16/FP8/MXFP8/NVFP4/MXFP4/Transformer Engine outputs may be used for forward/loss, but prediction/logit/probability tensors must use `tensor.detach().to(torch.float32).cpu().numpy()` before NumPy, sklearn, pandas, or submission CSV export. Labels/IDs may remain integer CPU arrays.",
        "",
        "🚫 **Execution Guidelines**:",
        "• NO tqdm (not installed), NO verbose=1",
        "• Print only 1 line per epoch (minimize logging)",
        '• On Windows or in unguarded scripts, use PyTorch DataLoader `num_workers=0`. Only use worker processes (`num_workers>=2`) when executable code is protected by `if __name__ == "__main__":`.',
        '• Prefer a `main()` function with an `if __name__ == "__main__": main()` guard for every complete runnable script.',
        "",
        "⚠️  **Self-Check Before Finalizing**:",
        "□ Did predictions pass through model's learned weights during inference? (If NO → INVALID)",
        "□ Did I generate submission.csv in correct path with ALL test predictions?",
        "□ Did I print validation metric as the last line?",
        "□ Did I use the COMPLETE training dataset (not a tiny subset)?",
    ]
    if expose_prediction:
        impl_guideline.append(
            "The implementation should include a predict() function, "
            "allowing users to seamlessly reuse the code to make predictions on new data. "
            "The prediction function should be well-documented, especially the function signature."
        )

    if k_fold_validation > 1:
        impl_guideline.append(
            f"The evaluation should be based on {k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
        )

    return {"Implementation guideline": impl_guideline}

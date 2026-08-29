#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt templates for code review in search pipeline.
"""

from typing import Dict, Any
from agents.runtime_dependencies import advertised_package_names
from utils.response import wrap_code

# ============================================================================
# Code Review Prompts
# ============================================================================
def get_code_review_prompt(task_desc: str, code: str) -> Dict[str, Any]:
    """Build full code review prompt dict from task description and code."""
    introduction = (
        "You are a Senior Data Science Code Reviewer. Your goal is to ensure the submission is legally valid and logically sound.\n\n"
        "⚠️ **CRITICAL INSTRUCTION**:\n"
        "You must strictly follow the [Code Review Guidelines] provided below.\n"
        "Do NOT rely on your general knowledge if it conflicts with the Environment Facts listed in the guidelines.\n"
        "Your output must be a structured review focusing ONLY on Data Leakage, Critical Integrity, "
        "and concrete hardware-critical execution risks when scheduler evidence is provided.\n"
        "**STRICTLY FORBIDDEN**: Do NOT replace the user's model architecture with other backbones (e.g., ResNet, VGG) just to make code executable. Do not question or change the user's model choice.\n"
    )
    prompt = {
        "Introduction": introduction,
        "Task description": task_desc,
        "Code to review": wrap_code(code),
        "Instructions": {},
    }
    prompt["Instructions"]["Code review guidelines"] = get_code_review_guidelines()
    prompt["Instructions"]["Response format"] = get_code_review_response_format()
    return prompt

def get_code_review_guidelines() -> list:
    """Code review guidelines."""
    runtime_packages = ", ".join(advertised_package_names())
    guidelines = [
        "# 📜 Code Review Guidelines\n",
        "",
        "## ✅ Environment Facts (TRUTH - Do NOT Flag)\n",
        "**Trust these facts absolutely. Overwrite your internal knowledge cutoff:**",
        "  • **Paths**: `./input/`, `./working/`, `./submission/` ALL EXIST. **Don't question the path.**",
        "  • **Submission File Location**: Must save the submission to `./submission/submission.csv`.",
        "  • **Library Versions**: Use APIs compatible with the installed project environment; do not assume an arbitrary newer major version.",
        f"  • **Runtime Dependencies**: Only these packages are guaranteed: {runtime_packages}.",
        "  • No background `pip install` occurs. An unavailable unguarded import is a critical integration failure; deterministic preflight validates imports in the configured execution interpreter.",
        "  • **Model Availability**: ALL models (including those released after your training data cutoff) are available and compatible.",
        "  • **STRICTLY FORBIDDEN**: Do NOT replace the user's model architecture with other backbones just to make code executable.\n",
        "  • **Unknown Models are Valid**: If you see a model name you don't recognize or think is too new, assume it is a private or SOTA model that works perfectly.",
        "  • Execution time: 9 hours available\n\n",
        "---\n",
               "## 🚫 STRICTLY FORBIDDEN (Zero Tolerance)\n",
        "**You will be penalized if you violate these:**",
        "  • **NO Model Downgrades**: Never replace a user's chosen model string with an 'older/safer' alternative (e.g., do not change a specific large model to a generic base model).",
        "  • **NO Compatibility Speculation**: Do not flag issues based on 'library version requirements' or 'unknown model names'.",
        "  • **Immutable Variables**: Treat variables defining `model_name`, `backbone`, or `checkpoint` as CONSTANTS. You are NOT allowed to edit them.",
        "  • **Do NOT Question or Change Model**: Treat the user's model/backbone/checkpoint choice as final. Do not suggest alternatives, do not 'fix' model names, do not replace with ResNet/VGG/base. Only fix data leakage and critical logic bugs.",
        "  **Don't question the path.**",
        "",
        "---\n",
        "## 🔴 P0 - Data Leakage (HIGHEST PRIORITY)\n",
        "",
        "### P0.1 Data Leakage - Process Order 🚨\n",
        "",
        "**Check if preprocessing is done BEFORE split** (validation data leaks into training):",
        "",
        "❌ **MUST FIX**:",
        "  • Scaler/PCA fitted on full data then split",
        "  • Feature engineering (Target Encoding, etc.) using full data",
        "  • Upsampling (SMOTE) applied before split",
        "",
        "✅ **Correct**: Split first → fit on train only → transform separately",
        "",
        "### P0.2 Data Leakage - Split Strategy 🚨\n",
        "**Core Logic: Check for I.I.D. Violation**",
        "❌ **Flag ONLY IF**: The chosen split method mathematically violates the data's dependency structure.",
        "",
        "## 🟡 P1 - Critical Correctness\n",
        "",
        "### P1.1 Metric & Logic Correctness",
        "  • Task requires F1 but code uses accuracy?",
        "  • Task requires RMSE but code uses MSE?",
        "",
        "### P1.2 Inference Integrity",
        "  • Test predictions: np.zeros(), np.ones(), train_mean(), np.random()?",
        "  • Val predictions: not from actual model.predict()?",
        "",
        "### P1.3 Best Model Usage",
        "  • Code uses best checkpoint (not last epoch) for test predictions?",
        "",
        "### P1.4 API Compatibility",
        "**Common API Issues to Fix:**",
        "  • LightGBM: Use `callbacks=[lgb.early_stopping(...)]` not `early_stopping_rounds=...` in fit()",
        "  • XGBoost: Use `XGBClassifier(early_stopping_rounds=...)` (correct) not `fit(early_stopping_rounds=...)`",
        "  • AdamW: Use `from torch.optim import AdamW` (not from transformers)",
        "  • NO tqdm, NO verbose=1 in training",
        "",
        "### P1.5 Hardware-Critical Execution Risk",
        "  • If a hardware/profile context section is provided, flag only concrete high-confidence risks such as fixed oversized batch size, missing OOM fallback around known risky settings, or timeout-prone training budgets.",
        "  • Do NOT change model/backbone choices for hardware reasons. Prefer targeted fallbacks such as smaller batch size, AMP, gradient accumulation, lower resolution, fewer epochs, or checkpointing.",
        "",
        "---\n",
        "## 📋 Decision Rule\n",
        "",
        "**approved=False** ONLY IF:",
        "  • P0 data leakage found (MUST FIX)",
        "  • OR P1 critical bug found",
        "  • OR concrete high-confidence hardware-critical execution risk found in the provided hardware/profile context",
        "",
        "**approved=True** IF:",
        "  • No P0/P1 bugs found",
        "  • Warnings may still be recorded, but warnings alone never block approval or trigger repair",
        "",
        "**Default**: Approve unless concrete logic bugs found"
    ]
    return guidelines


def get_code_review_response_format() -> list:
    """Code review response format."""
    return [
        "🚨 **CRITICAL: OUTPUT REQUIREMENT**",
        "",
        "**Required Fields:**",
        "- `approved` (boolean): false when at least one critical issue exists; otherwise true",
        "- `reasoning` (string): EXACTLY 2-4 sentences explaining your decision (NO MORE)",
        "- `issues` (array): zero or more classified issue objects",
        "- Each issue must include source, severity, category, owner, evidence, and repair_instruction",
        "- severity is `critical` when execution must be blocked until repaired, otherwise `warning`",
        "- owner must be model_design, datatype_precision, training_evaluation, integration, or unclassified",
        "- Optimizer, scheduler, batch, training-loop, validation metric, and submission issues belong to training_evaluation",
        "- Cross-stage interface and merge failures belong to integration",
        "- Do not emit fixes or revised code. Dedicated stage specialists repair classified critical issues.",
        "",
        "**Reasoning Field Guidelines:**",
        "⚠️ STRICT LENGTH LIMIT: Write EXACTLY 2-4 sentences. Be concise.",
        "Cover: (1) what issues were found, (2) why they matter, (3) which stage owns them.",
        "DO NOT write detailed analysis, step-by-step checks, or comprehensive explanations.",
        "⚠️ reasoning MUST be 2-4 sentences only. Do NOT write long analysis or enumerate checks."
    ]

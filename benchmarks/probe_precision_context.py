"""Bounded OpenRouter prompt ablation; source prompts, outputs and costs are retained.

This tests the decision + three-stage coder + merge on real sklearn digits data.
It is a CPU code-generation diagnostic, not a PetFinder/GPU search benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.coder.stepwise_coder import stepwise_plan_and_code_query
from agents.hardware_context import (
    HardwarePromptContext, build_stepwise_hardware_stage_sections,
    compact_optimization_context, filter_hardware_context_for_agent,
    format_hardware_prompt_section,
)
from agents.precision_validation import validate_training_precision
from agents.prompts.impl_guideline import get_impl_guideline_from_agent
from agents.prompts.pipeline_decision import build_pipeline_decision, format_pipeline_decision_prompt_section
from engine.script_introspection import introspect_training_script
from localml_scheduler.client import SchedulerClient
from utils.precision_policy import resolve_precision_policy


def save(path, payload):
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def plot(rows, output):
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), layout="constrained")
    labels = [f"{r['model'].split('/')[-1]} {r['precision']}/{r['context_mode']} s{r['seed']}" for r in rows]
    for i, row in enumerate(rows):
        axes[0].barh(i, row.get("generation_seconds", 0), color="#607d9c")
        axes[0].barh(i, row.get("execution_seconds", 0), left=row.get("generation_seconds", 0),
                     color="#2d936c" if row.get("success") else "#c44e52")
        score = row.get("validation_accuracy")
        if score is not None:
            axes[1].scatter(i + 1, score, color="#2d936c" if row.get("success") else "#c44e52")
        else:
            axes[1].annotate("API blocked" if row.get("api_blocked") else "failed", (i + 1, 0.05), ha="center", color="#c44e52", rotation=45)
    axes[0].set(yticks=range(len(rows)), yticklabels=labels, xlabel="Seconds from each independent candidate start",
                title="Candidate Gantt: blue = decision/code/merge API time; green/red = CPU execution")
    axes[0].invert_yaxis()
    axes[1].set(xlabel="Candidate node (independent runs; no search feedback)", ylabel="Validation accuracy (higher is better)",
                xticks=range(1, len(rows) + 1), ylim=(0, 1), title="Real digits data; 3 epochs; CPU diagnostic, not GPU/PetFinder quality")
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["qwen/qwen3.5-9b"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--max-cost-usd", type=float, default=3.0)
    args = parser.parse_args()
    if not 0 < args.max_cost_usd <= 20:
        parser.error("The authorized total ceiling is $20.")
    key = os.environ["OPEN_ROUTER_API_KEY"]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    ledger = {"ceiling_usd": args.max_cost_usd, "reserved_usd": 0.0, "reported_cost_usd": 0.0, "requests": []}
    save(output / "budget.json", ledger)
    rows = []
    cells = [(model, seed, precision, context_mode) for model in args.models for seed in args.seeds
             for precision in ("normal", "conservative") for context_mode in ("full", "compact")]
    random.Random(42).shuffle(cells)
    for cell_index, (model, seed, precision, context_mode) in enumerate(cells):
        case = output / f"{cell_index:02d}_{model.split('/')[-1]}_{precision}_{context_mode}_{seed}"
        case.mkdir()
        (case / "input").mkdir()
        (case / "submission").mkdir()
        (case / "working").mkdir()
        x, y = load_digits(return_X_y=True)
        x = (x / 16).astype(np.float32)
        train, test = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed, stratify=y)
        train, valid = train_test_split(train, test_size=0.2, random_state=seed, stratify=y[train])
        np.savez(case / "input" / "digits.npz", X_train=x[train], y_train=y[train], X_valid=x[valid],
                 y_valid=y[valid], X_test=x[test], test_ids=test)
        save(case / "data_manifest.json", {"source": "sklearn.datasets.load_digits", "seed": seed,
             "train_indices": train.tolist(), "validation_indices": valid.tolist(), "test_indices": test.tolist(),
             "sha256": hashlib.sha256((case / "input" / "digits.npz").read_bytes()).hexdigest()})
        task = (
            "Classify handwritten digits using a small PyTorch MLP. Local ./input/digits.npz contains "
            "X_train [1149,64], y_train [1149], X_valid [288,64], y_valid [288], X_test [360,64], test_ids [360]. "
            "Features are float32 scaled to [0,1]; labels int64 in [0,9]. Use these fixed splits unchanged. "
            "Train the complete training split for exactly 3 epochs with seed " + str(seed) + ". "
            "No ensembling, tuning loops, downloads or external data. Use AdamW. CUDA if available, otherwise CPU. "
            "Evaluation: holdout classification accuracy, higher is better. Save ./submission/submission.csv "
            "with Id,Label columns for every test row in its supplied order. Print Final Validation Score: <accuracy>. "
            "Keep runtime under 90 seconds. CandidateAdapter must use the real MLP and cross-entropy, CPU safe."
        )
        agent_cfg = SimpleNamespace(precision_optimization_mode=precision, hardware_context_mode=context_mode,
            hardware_context_enabled=True, pipeline_decision_enabled=True, time_limit=3600, steps=1,
            code=SimpleNamespace(model=model, temp=0.2))
        agent = SimpleNamespace(acfg=agent_cfg, cfg=SimpleNamespace(agent=agent_cfg, exec=SimpleNamespace(timeout=90),
            experiment=SimpleNamespace(mode="hardware_aware"), pretrain_model_dir="", exp_id="digits"),
            task_desc=task, current_step=0, start_time=time.time(), use_coldstart=False)
        policy = resolve_precision_policy({"architecture": "ampere", "compute_capability": "8.0"}, mode=precision)
        raw = {"hardware_context": {"found": True, "hardware": {"architecture": "ampere", "gpu_name": "A100 (prompt target; execution is CPU)"}},
               "stage_hardware_features": SchedulerClient._stage_feature_context_from_static_graph(
                   hardware_name="NVIDIA A100 SXM4 80GB", stages=["model_design", "datatype_precision", "training_evaluation"], limit=8, precision_mode=precision)}
        compact = compact_optimization_context(raw)
        compact["precision_policy"] = policy.to_dict()
        filtered = filter_hardware_context_for_agent(compact, stage="draft", precision_policy=policy)
        if context_mode == "compact":
            compact["hardware_context_mode"] = filtered["hardware_context_mode"] = "compact"
        context = HardwarePromptContext(compact_context=compact, filtered_context=filtered,
            prompt_section=format_hardware_prompt_section(filtered))
        save(case / "hardware_context.json", {"raw": raw, "compact": compact, "filtered": filtered})
        call_index = 0

        def generate(prompt, **kwargs):
            nonlocal call_index
            if row.get("api_blocked"):
                raise RuntimeError("OpenRouter access/billing blocked; further requests disabled")
            messages = ([{"role": role, "content": str(value)} for role, value in prompt.items()]
                        if isinstance(prompt, dict) else [{"role": "user", "content": prompt}])
            payload = {"model": model, "messages": messages, "temperature": 0.2, "seed": seed,
                "max_tokens": 8192, "reasoning": {"enabled": False},
                "provider": {"max_price": {"prompt": 0.5, "completion": 2.0}, "sort": "price"}}
            if kwargs.get("json_schema"):
                payload["response_format"] = {"type": "json_object"}
            # Worst-case UTF-8 bytes + framing at provider price ceilings. Never
            # refund reservations: a timeout may still have incurred a charge.
            reserved = ((len(json.dumps(payload).encode()) + 2048) * 0.5 + 8192 * 2) / 1e6
            if ledger["reserved_usd"] + reserved > args.max_cost_usd:
                raise RuntimeError("Experiment cost reservation ceiling reached")
            ledger["reserved_usd"] += reserved
            entry = {"case": case.name, "call": call_index, "reserved_usd": reserved}
            ledger["requests"].append(entry)
            save(output / "budget.json", ledger)
            stem = case / f"call_{call_index:02d}"
            call_index += 1
            save(stem.with_suffix(".request.json"), payload)
            start = time.monotonic()
            response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": "Bearer " + key}, json=payload, timeout=180)
            data = response.json()
            save(stem.with_suffix(".response.json"), data)
            entry.update(latency_seconds=time.monotonic() - start, http_status=response.status_code)
            if response.status_code in {401, 402, 403}:
                row["api_blocked"] = True
            save(output / "budget.json", ledger)
            response.raise_for_status()
            usage = data.get("usage") or {}
            entry.update(latency_seconds=time.monotonic() - start, usage=usage, provider=data.get("provider"))
            ledger["reported_cost_usd"] += float(usage.get("cost") or 0)
            save(output / "budget.json", ledger)
            choice = data["choices"][0]
            if choice.get("finish_reason") == "length":
                raise RuntimeError("Truncated completion; retained for diagnosis")
            return choice["message"]["content"]

        row = {"model": model, "seed": seed, "precision": precision, "context_mode": context_mode, "case": case.name}
        start = time.monotonic()
        try:
            with patch("agents.prompts.pipeline_decision.generate", generate), patch("agents.coder.stepwise_coder.generate", generate):
                decision = build_pipeline_decision(agent, stage="draft", data_preview=task, hardware_contexts=[context], max_retries=1)
                instructions = get_impl_guideline_from_agent(agent)
                base = {"Introduction": "You are an ML engineer.", "Task description": task, "Instructions": instructions}
                _, code, metadata = stepwise_plan_and_code_query(agent, base, task, {
                    "hardware_context": compact, "hardware_prompt_section": context.prompt_section,
                    "hardware_stage_sections": build_stepwise_hardware_stage_sections(design_context=context, execution_context=context),
                    "pipeline_decision": decision, "pipeline_decision_section": format_pipeline_decision_prompt_section(decision),
                }, return_metadata=True)
            row["generation_seconds"] = time.monotonic() - start
            (case / "candidate.py").write_text(code)
            save(case / "pipeline_decision.json", decision)
            save(case / "stepwise_metadata.json", metadata)
            row["precision_issues"] = [issue.to_dict() for issue in validate_training_precision(agent, code, context=context)]
            row["script_metadata"] = introspect_training_script(code)
            compile(code, str(case / "candidate.py"), "exec")
            if row["precision_issues"]:
                raise RuntimeError("Precision guard rejected candidate before execution")
            execution_start = time.monotonic()
            result = subprocess.run([sys.executable, "candidate.py"], cwd=case,
                env={"PATH": os.environ["PATH"], "PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": "",
                     "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"},
                capture_output=True, text=True, timeout=90)
            row["execution_seconds"] = time.monotonic() - execution_start
            (case / "execution.log").write_text(result.stdout + "\n" + result.stderr)
            matches = re.findall(r"Final Validation Score:\s*([0-9.eE+-]+)", result.stdout)
            row["validation_accuracy"] = float(matches[-1]) if matches else None
            import pandas as pd
            submission = pd.read_csv(case / "submission" / "submission.csv")
            row["success"] = bool(result.returncode == 0 and matches and list(submission.columns) == ["Id", "Label"]
                and np.array_equal(submission.Id, test) and submission.Label.isin(range(10)).all()
                and 0 <= row["validation_accuracy"] <= 1)
            row["returncode"] = result.returncode
        except Exception as exc:
            row.update(success=False, error=f"{type(exc).__name__}: {exc}")
        row.setdefault("generation_seconds", time.monotonic() - start)
        row["calls"] = call_index
        rows.append(row)
        save(output / "results.json", rows)
        plot(rows, output)
        print(json.dumps({k: row.get(k) for k in ("case", "success", "error", "validation_accuracy", "generation_seconds", "calls")}), flush=True)
        if row.get("api_blocked"):
            break
    print(json.dumps({"reported_cost_usd": ledger["reported_cost_usd"], "reserved_usd": ledger["reserved_usd"]}), flush=True)


if __name__ == "__main__":
    main()

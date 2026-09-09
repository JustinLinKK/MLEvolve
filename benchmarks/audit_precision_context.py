"""Replay prompt construction against archived PetFinder stage outputs, without LLM calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import time
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.coder.stepwise_coder import MetaAgent, StepwiseContext, create_default_step_agents
from agents.hardware_context import (
    HardwarePromptContext, build_stepwise_hardware_stage_sections, compact_optimization_context,
    filter_hardware_context_for_agent, format_hardware_prompt_section, hardware_context_instructions,
)
from agents.prompts.impl_guideline import get_impl_guideline_from_agent
from agents.prompts.pipeline_decision import (
    _build_decision_prompt, _collect_evidence_state, _fallback_decision, format_pipeline_decision_prompt_section,
)
from localml_scheduler.client import SchedulerClient
from utils.precision_policy import CONSERVATIVE_PRECISION_INSTRUCTION, resolve_precision_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with sqlite3.connect(f"file:{args.pipeline_db.resolve()}?mode=ro", uri=True) as db:
        node_id, archived = db.execute("SELECT node_id,prompt_text FROM prompt_snapshots WHERE stage='draft' ORDER BY prompt_id LIMIT 1").fetchone()
        saved_sizes = db.execute("SELECT stage,COUNT(*),MIN(prompt_chars),MAX(prompt_chars),AVG(prompt_chars) FROM prompt_snapshots GROUP BY stage").fetchall()
    parts = re.split(r"(?m)^# Used Prompt \d+: ", archived)[1:]
    task = re.search(r"# Task description\s*\n(.*?)\n# (?:Memory|Hardware)", parts[0], re.S).group(1).strip()
    matches = re.findall(r"### Step \d+: (\w+)\s+\*\*Plan:\*\*(.*?)\*\*Code:\*\*\s*```(?:python)?\s*\n(.*?)```", parts[-1], re.S)
    steps = [{"name": name, "plan": plan.strip(), "code": code} for name, plan, code in matches]
    if [step["name"] for step in steps] != ["model_design", "datatype_precision", "training_evaluation"]:
        raise ValueError("Archive does not contain the complete three-stage source outputs")
    (output / "frozen_steps.json").write_text(json.dumps(steps, indent=2))
    rows = []
    for precision in ("normal", "conservative"):
        for mode in ("full", "compact"):
            start = time.monotonic()
            case = output / f"{precision}_{mode}"
            case.mkdir()
            acfg = SimpleNamespace(code=SimpleNamespace(model="qwen-diagnostic", temp=0), precision_optimization_mode=precision,
                hardware_context_mode=mode, time_limit=3600, steps=1)
            agent = SimpleNamespace(acfg=acfg, cfg=SimpleNamespace(exec=SimpleNamespace(timeout=90), exp_id="petfinder-pawpularity-score",
                pretrain_model_dir=""), current_step=0, start_time=time.time(), task_desc=task, use_coldstart=False)
            policy = resolve_precision_policy({"architecture": "ampere", "compute_capability": "8.0"}, mode=precision)
            raw = {"hardware_context": {"found": True, "hardware": {"architecture": "ampere", "gpu_name": "A100 (static prompt target)"}},
                "stage_hardware_features": SchedulerClient._stage_feature_context_from_static_graph(
                    hardware_name="NVIDIA A100 SXM4 80GB", stages=["model_design", "datatype_precision", "training_evaluation"], limit=8, precision_mode=precision)}
            compact = compact_optimization_context(raw)
            compact["precision_policy"] = policy.to_dict()
            filtered = filter_hardware_context_for_agent(compact, stage="draft", precision_policy=policy)
            if mode == "compact":
                compact["hardware_context_mode"] = filtered["hardware_context_mode"] = mode
            hardware = HardwarePromptContext(compact_context=compact, filtered_context=filtered,
                prompt_section=format_hardware_prompt_section(filtered))
            evidence = _collect_evidence_state([hardware])
            decision_prompt = _build_decision_prompt(task_desc=task, data_preview="Archived PetFinder task; fixed stage source below.",
                stage="draft", evidence_state=evidence, parent_pipeline_decision=None, previous_code="", execution_output="",
                stage_context="", precision_policy=policy)
            if precision == "conservative":
                decision_prompt["system"] += " " + CONSERVATIVE_PRECISION_INSTRUCTION
            # This is a deterministic fallback contract, never an LLM response.
            decision = _fallback_decision(task_desc=task, data_preview="PetFinder images and tabular metadata", evidence_state=evidence)
            context = StepwiseContext(hardware_brief=hardware.prompt_section, hardware_context=compact,
                hardware_stage_sections=build_stepwise_hardware_stage_sections(design_context=hardware, execution_context=hardware),
                pipeline_decision=decision, pipeline_decision_section=format_pipeline_decision_prompt_section(decision))
            base = {"Introduction": "You are an ML engineer.", "Task description": task, "Instructions": get_impl_guideline_from_agent(agent)}
            base["Instructions"].update(hardware_context_instructions(hardware))
            prompts = {"decision": "\n".join(decision_prompt.values())}
            for index, stage in enumerate(create_default_step_agents()):
                prompts[stage.name] = stage._build_prompt(task, "Fixed archived PetFinder stage outputs.", steps[:index], base, agent, context)
            prompts["merge"] = MetaAgent()._build_merge_prompt(task, "Fixed archived PetFinder stage outputs.", steps, base, agent, context)
            for name, text in prompts.items():
                (case / f"{name}.txt").write_text(text)
            rows.append({"precision": precision, "context_mode": mode, "prompt_chars": {name: len(text) for name, text in prompts.items()},
                         "hardware_lookup": raw["stage_hardware_features"].get("hardware"),
                         "hardware_section_chars": {name: len(text) for name, text in context.hardware_stage_sections.items()},
                         "construction_seconds": time.monotonic() - start})
    report = {"source_database": str(args.pipeline_db), "source_node": node_id,
              "source_prompt_sha256": hashlib.sha256(archived.encode()).hexdigest(),
              "archive_prompt_aggregates": saved_sizes, "archive_first_draft_calls": {
                  part.split('\n', 1)[0]: len(part.split('\n', 1)[1]) for part in parts},
              "evidence_tier": "CPU prompt construction only; no model calls or training; identical archived code in every cell",
              "rows": rows}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout="constrained")
    names = [f"{r['precision']}/{r['context_mode']}" for r in rows]
    axes[0].barh(names, [r["construction_seconds"] for r in rows], color=["#607d9c", "#2d936c"] * 2)
    axes[0].set(title="Gantt: independent CPU prompt-construction jobs (not model generation/training)", xlabel="Seconds from each job start")
    stages = list(rows[0]["prompt_chars"])
    for row, name in zip(rows, names):
        axes[1].plot(range(1, len(stages) + 1), list(row["prompt_chars"].values()), marker="o", label=name)
    axes[1].set(xticks=range(1, len(stages) + 1), xticklabels=stages, ylabel="Prompt characters (not tokens)",
                title="Metric per workflow stage; fixed real PetFinder source; no quality scores measured")
    axes[1].legend()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

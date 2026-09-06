"""One reviewable record per candidate, keeping runtime and inferred evidence separate."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from engine.script_introspection import introspect_training_script
from utils.training_diagnostics import TRAINING_DIAGNOSTICS_MARKER


def source_provenance():
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        return {"revision": revision, "dirty": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def build_node_diagnostics(cfg, node):
    code = getattr(node, "code", "") or ""
    inferred = introspect_training_script(code)
    raw_output = getattr(node, "_term_out", None) or ""
    output = raw_output if isinstance(raw_output, str) else "".join(str(p) for p in raw_output)
    samples = []
    invalid_samples = 0
    for line in output.splitlines():
        if not line.startswith(TRAINING_DIAGNOSTICS_MARKER + " "):
            continue
        try:
            sample = json.loads(line.split(" ", 1)[1])
            if not isinstance(sample, dict) or sample.get("schema_version") != 1:
                raise ValueError("unsupported runtime diagnostics")
            for key in ("attempted_updates", "completed_updates", "skipped_updates", "unaccounted_optimizer_calls"):
                value = sample.get(key)
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError("invalid optimizer counter")
            dtypes = sample.get("autocast_dtypes_observed")
            if dtypes is not None and (not isinstance(dtypes, list) or not all(isinstance(v, str) for v in dtypes)):
                raise ValueError("invalid runtime dtype observations")
            samples.append(sample)
        except (ValueError, TypeError):
            invalid_samples += 1
    mode = str(getattr(getattr(cfg, "experiment", None), "mode", "unknown"))
    agent_cfg = getattr(cfg, "agent", None)
    configured = bool(getattr(agent_cfg, "hardware_context_enabled", True))
    enabled = configured and mode not in {"origin", "baseline", "unknown"}
    prompt = getattr(node, "prompt_input", None)
    headings = (
        "# Hardware/Profile Optimization Context",
        "# Hardware-Aware Model Design Brief",
        "# Hardware-Aware Stage 1 Candidate Construction Context",
        "# Hardware-Aware Datatype/Precision Context",
        "# Hardware-Aware Training Hyperparameter Context",
    )
    present = [heading for heading in headings if prompt and heading in prompt]
    stages = (getattr(node, "diagnostics", None) or {}).get("hardware_prompt_stages")
    injected = any(item["context_present"] for item in stages) if stages else bool(present) if prompt else None
    parent = getattr(node, "parent", None)
    latest = samples[-1] if samples else None
    flags = []
    if enabled and injected is not True:
        flags.append("hardware_prompt_injection_unconfirmed")
    if latest is None:
        flags.append("runtime_diagnostics_missing")
    else:
        if (latest.get("skipped_updates") or 0) > 0:
            flags.append("optimizer_updates_skipped")
        if (latest.get("unaccounted_optimizer_calls") or 0) > 0:
            flags.append("optimizer_calls_missing_after_update_hook")
        observed = latest.get("autocast_dtypes_observed") or []
        inferred_dtype = {"fp16": "torch.float16", "bf16": "torch.bfloat16"}.get(inferred.get("precision_mode"))
        if inferred_dtype and observed and inferred_dtype not in observed:
            flags.append("runtime_precision_differs_from_static_inference")
        if "torch.float16" in observed and latest.get("grad_scaler_enabled") is False:
            flags.append("fp16_without_gradient_scaling")
    if invalid_samples:
        flags.append("invalid_runtime_diagnostics")
    return {
        "schema_version": 1,
        "run_id": str(getattr(cfg, "exp_name", "")),
        "experiment_mode": mode,
        "node_id": node.id,
        "parent_id": getattr(parent, "id", None),
        "stage": node.stage,
        "branch_id": getattr(node, "branch_id", None),
        "selection": (getattr(node, "diagnostics", None) or {}).get("selection"),
        "model_family": getattr(node, "model_family", None) or inferred.get("model_family"),
        "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "hardware_knowledge": {
            "configured": configured,
            "enabled_for_mode": enabled,
            "context_present_in_saved_prompt": bool(present) if prompt else None,
            "injection_status": ("disabled_for_mode" if not enabled else
                                 "present" if injected else "absent" if injected is False else "unknown_no_prompt"),
            "injection_evidence_source": "generation_stage_metadata" if stages else "saved_prompt_text",
            "generation_stages": stages,
            "prompt_sections": present,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest() if prompt else None,
            "evidence_refs": list(getattr(node, "hardware_evidence_refs", None) or []),
            "confidence": getattr(node, "scheduler_confidence", None),
        },
        "precision_optimization_mode": getattr(agent_cfg, "precision_optimization_mode", None),
        "inferred_training_settings": inferred,
        "runtime_observation_status": "reported" if latest else "missing",
        "runtime": latest,
        "runtime_samples": samples,
        "invalid_runtime_samples": invalid_samples,
        "diagnostic_flags": flags,
        "metric": getattr(getattr(node, "metric", None), "value", None),
        "metric_maximize": getattr(getattr(node, "metric", None), "maximize", None),
        "parent_metric": getattr(getattr(parent, "metric", None), "value", None),
        "exec_time_seconds": getattr(node, "exec_time", None),
        "created_time": getattr(node, "created_time", None),
        "finished_time": getattr(node, "finish_time", None),
        "is_buggy": getattr(node, "is_buggy", None),
        "is_valid": getattr(node, "is_valid", None),
        "exception_type": getattr(node, "exc_type", None),
        "review_status": getattr(node, "review_status", None),
        "review_issues": getattr(node, "review_issues", None),
        "backend": getattr(node, "backend_name", None),
    }


def write_node_diagnostics(cfg, journal):
    path = Path(cfg.log_dir) / "node_diagnostics.jsonl"
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for node in journal.nodes:
            if node.stage != "root":
                stream.write(json.dumps(build_node_diagnostics(cfg, node), default=str) + "\n")
    temporary.replace(path)
    return path

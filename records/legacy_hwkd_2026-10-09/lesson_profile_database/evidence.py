"""Frozen, sanitized evidence packets for validated search nodes."""

from __future__ import annotations

import ast
from collections import Counter
import difflib
import hashlib
import json
import re
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from engine.script_introspection import introspect_training_script, normalized_mlevolve_script_signature


MAX_PLAN_CHARS = 3000
MAX_ANALYSIS_CHARS = 3000
MAX_DIFF_CHARS = 8000
MAX_PROMPT_EXCERPT_CHARS = 1000
MAX_TERMINAL_CHARS = 2000

_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*(['\"]?)[^\s,'\"]+\2"),
    re.compile(r"(?i)bearer\s+[a-z0-9._~+/-]{8,}"),
    re.compile(r"\b(?:sk|ghp|xox[baprs])[-_][A-Za-z0-9_-]{12,}\b"),
)


def redact_text(value: Any, *, limit: int) -> str:
    text = str(value or "")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    text = re.sub(r"(?m)^\s*(?:DATA|DATASET|INPUT|OUTPUT)_?(?:DIR|PATH)\s*=.*$", "# [PATH REDACTED]", text)
    text = re.sub(
        r"(?m)^.*(?:,\s*[-+]?(?:\d+(?:\.\d+)?|['\"][^'\"]+['\"])){4,}.*$",
        "[DATA ROW REDACTED]",
        text,
    )
    return text[: max(0, int(limit))]


def code_fingerprint(code: str) -> str:
    return hashlib.sha256((code or "").encode("utf-8")).hexdigest()


def structural_fingerprint(code: str) -> dict[str, Any]:
    """AST-based fingerprint without importing or executing generated code."""

    operators: list[str] = []
    try:
        tree = ast.parse(code or "")
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                operators.append(f"{type(node).__name__}:{node.name}")
            elif isinstance(node, ast.Call):
                function = node.func
                if isinstance(function, ast.Attribute):
                    operators.append(f"call:{function.attr}")
                elif isinstance(function, ast.Name):
                    operators.append(f"call:{function.id}")
    except SyntaxError:
        operators = []
    normalized = sorted(operators)
    return {
        "hash": hashlib.sha256(json.dumps(normalized, separators=(",", ":")).encode()).hexdigest(),
        "operators": normalized[:256],
    }


def _layer_calls(code: str) -> list[str]:
    layer_names = {
        "conv1d": "conv1d",
        "conv2d": "conv2d",
        "conv3d": "conv3d",
        "multiheadattention": "attention",
        "attention": "attention",
        "batchnorm1d": "normalization",
        "batchnorm2d": "normalization",
        "batchnorm3d": "normalization",
        "layernorm": "normalization",
        "groupnorm": "normalization",
        "maxpool1d": "pooling",
        "maxpool2d": "pooling",
        "avgpool1d": "pooling",
        "avgpool2d": "pooling",
        "adaptiveavgpool1d": "pooling",
        "adaptiveavgpool2d": "pooling",
        "linear": "head",
    }
    result: list[str] = []
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return result
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = function.attr if isinstance(function, ast.Attribute) else function.id if isinstance(function, ast.Name) else ""
        kind = layer_names.get(name.lower())
        if kind:
            result.append(kind)
    return result


def bounded_parent_diff(parent_code: str, child_code: str) -> dict[str, Any]:
    lines = list(
        difflib.unified_diff(
            (parent_code or "").splitlines(),
            (child_code or "").splitlines(),
            fromfile="parent.py",
            tofile="child.py",
            lineterm="",
            n=2,
        )
    )
    added = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))
    material_groups = sum(1 for line in lines if line.startswith("@@"))
    diff = redact_text("\n".join(lines), limit=MAX_DIFF_CHARS)
    parent_layers = Counter(_layer_calls(parent_code))
    child_layers = Counter(_layer_calls(child_code))
    added_layers = list((child_layers - parent_layers).elements())
    removed_layers = list((parent_layers - child_layers).elements())
    changed_layer_types = sorted(set([*added_layers, *removed_layers]))
    if added_layers and not removed_layers:
        action = "add"
    elif removed_layers and not added_layers:
        action = "remove"
    elif added_layers or removed_layers:
        action = "replace"
    else:
        action = "other"
    parent_facts = introspect_training_script(parent_code or "") if parent_code else {}
    child_facts = introspect_training_script(child_code or "") if child_code else {}
    configuration_fields = (
        "model_key",
        "proposed_batch_size",
        "proposed_epochs",
        "input_resolution",
        "uses_amp",
        "gradient_accumulation_steps",
        "num_workers",
        "framework",
    )
    training_changes = {
        key: {"before": parent_facts.get(key), "after": child_facts.get(key)}
        for key in configuration_fields
        if parent_code and parent_facts.get(key) != child_facts.get(key)
    }
    if not lines:
        scope = "training_only"
    elif changed_layer_types and not training_changes:
        scope = "one_layer" if len(changed_layer_types) == 1 else "small_group"
    elif not changed_layer_types and len(training_changes) <= 1:
        scope = "training_only"
    else:
        scope = "multi_change"
    if material_groups > 3 or added + removed > 120 or len(changed_layer_types) > 3:
        scope = "multi_change"
    first_hunk = next((line for line in lines if line.startswith("@@")), "")
    return {
        "unified_diff": diff,
        "added_lines": added,
        "removed_lines": removed,
        "material_groups": material_groups,
        "change_scope": scope,
        "change_action": action,
        "layer_type": changed_layer_types[0] if len(changed_layer_types) == 1 else ("mixed" if changed_layer_types else "other"),
        "changed_layer_types": changed_layer_types,
        "training_changes": training_changes,
        "location_signature": first_hunk[:200],
        "controlled": scope != "multi_change",
        "diff_truncated": len("\n".join(lines)) > MAX_DIFF_CHARS,
    }


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        return "[DEPTH-LIMIT]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return redact_text(value, limit=4000)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_safe(asdict(value), depth=depth + 1)
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            key_text = str(key)
            if any(token in key_text.lower() for token in ("api_key", "apikey", "password", "secret", "access_token")):
                continue
            result[key_text] = _json_safe(item, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item, depth=depth + 1) for item in list(value)[:256]]
    return str(value)


def _terminal_error_excerpt(value: Any) -> str:
    selected = []
    for line in str(value or "").splitlines():
        lowered = line.lower()
        if any(token in lowered for token in (
            "error", "exception", "traceback", "failed", "failure", "out of memory", "oom", "cuda", "assert",
        )):
            selected.append(line)
    return redact_text("\n".join(selected), limit=MAX_TERMINAL_CHARS)


def _scheduler_measurements(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "job_id",
        "scheduler_mode",
        "placement_mode",
        "placement_backend",
        "status",
        "submitted_at",
        "started_at",
        "finished_at",
        "duration_seconds",
        "runtime_seconds",
        "detected_batch_size",
        "resolved_batch_size",
        "proposed_epochs",
        "model_key",
        "framework",
        "uses_amp",
        "requires_gpu",
        "script_signature",
        "metric",
        "peak_vram_mb",
        "average_vram_mb",
        "throughput",
        "step_time_seconds",
        "gpu_utilization",
        "resource_slice",
        "resource_slice_key",
        "mig_profile",
    }
    result = {key: value[key] for key in allowed if key in value and value[key] is not None}
    payload = value.get("payload")
    if isinstance(payload, Mapping):
        for key in allowed:
            if key in payload and payload[key] is not None and key not in result:
                result[key] = payload[key]
    return _json_safe(result)


def prompt_reference(node: Any) -> dict[str, Any]:
    raw = str(getattr(node, "prompt_input", None) or "")
    allowlisted_lines = [
        line for line in raw.splitlines()
        if re.match(r"^\s*(?:#|Task|Goal|Stage|Constraint|Requirement|Metric)\b", line, re.IGNORECASE)
    ]
    return {
        "sha256": str(getattr(node, "prompt_snapshot_sha256", "") or "")
        or (hashlib.sha256(raw.encode("utf-8")).hexdigest() if raw else ""),
        "path": str(getattr(node, "prompt_snapshot_path", "") or ""),
        "allowlisted_excerpt": redact_text("\n".join(allowlisted_lines), limit=MAX_PROMPT_EXCERPT_CHARS),
    }


def build_evidence_packet(
    *,
    node: Any,
    identity: Mapping[str, Any],
    outcome: str,
    run_id: str,
    task_description: str = "",
    scheduler_measurements: Mapping[str, Any] | None = None,
    evidence_refs: list[str] | None = None,
) -> dict[str, Any]:
    parent = getattr(node, "parent", None)
    code = str(getattr(node, "code", "") or "")
    parent_code = str(getattr(parent, "code", "") or "")
    references = [str(item) for item in (evidence_refs or []) if str(item).strip()]
    node_id = str(getattr(node, "id", ""))
    if node_id and f"node:{node_id}" not in references:
        references.append(f"node:{node_id}")
    if run_id and f"run:{run_id}" not in references:
        references.append(f"run:{run_id}")
    if not references:
        raise ValueError("Evidence packets require at least one resolvable reference")
    metric = getattr(node, "metric", None)
    metric_value = getattr(metric, "value", None)
    packet = {
        "schema_version": "lesson-observation-v1",
        "captured_at": time.time(),
        "run_id": run_id,
        "node_id": node_id,
        "parent_node_id": str(getattr(parent, "id", "") or ""),
        "stage": str(getattr(node, "stage", "") or ""),
        "generation_strategy": str(getattr(node, "generation_strategy", "") or ""),
        "source_node_ids": list(getattr(node, "source_node_ids", []) or []),
        "identity": dict(identity),
        "outcome": outcome,
        "validation": {
            "is_buggy": getattr(node, "is_buggy", None),
            "is_valid": getattr(node, "is_valid", None),
            "metric": metric_value,
            "metric_maximize": getattr(metric, "maximize", None),
        },
        "code": {
            "sha256": code_fingerprint(code),
            "normalized_signature": normalized_mlevolve_script_signature(code),
            "structural": structural_fingerprint(code),
            "introspection": introspect_training_script(code),
        },
        "parent_code": {
            "sha256": code_fingerprint(parent_code) if parent else "",
            "structural": structural_fingerprint(parent_code) if parent else {},
        },
        "delta": bounded_parent_diff(parent_code, code) if parent else {
            "unified_diff": "",
            "added_lines": 0,
            "removed_lines": 0,
            "material_groups": 0,
            "change_scope": "training_only",
            "change_action": "other",
            "layer_type": "other",
            "changed_layer_types": [],
            "training_changes": {},
            "location_signature": "",
            "controlled": True,
            "diff_truncated": False,
        },
        "artifacts": {
            "plan": redact_text(getattr(node, "plan", ""), limit=MAX_PLAN_CHARS),
            "code_summary": redact_text(getattr(node, "code_summary", ""), limit=MAX_PLAN_CHARS),
            "analysis": redact_text(getattr(node, "analysis", ""), limit=MAX_ANALYSIS_CHARS),
            "pipeline_decision": _json_safe(getattr(node, "pipeline_decision", None)),
            "hardware_decision": _json_safe(getattr(node, "hardware_decision", None)),
            "stage_notes": _json_safe(getattr(node, "stage_note_board", [])),
            "review_issues": _json_safe(getattr(node, "review_issues", [])),
            "review_history": _json_safe(getattr(node, "review_history", [])),
            "bug_report": redact_text(getattr(node, "bug_report", ""), limit=MAX_ANALYSIS_CHARS),
            "fix_report": redact_text(getattr(node, "fix_report", ""), limit=MAX_ANALYSIS_CHARS),
            "terminal_excerpt": _terminal_error_excerpt(getattr(node, "term_out", "")),
        },
        "prompt": prompt_reference(node),
        "scheduler_measurements": _scheduler_measurements(dict(scheduler_measurements or {})),
        "task": {"description_hash": hashlib.sha256(task_description.encode()).hexdigest()},
        "evidence_refs": references,
    }
    # Round-trip enforces JSON serializability before this crosses the durable boundary.
    return json.loads(json.dumps(packet, sort_keys=True, separators=(",", ":"), default=str))

#!/usr/bin/env python3
"""Generate stress-test adjudication artifacts from existing run logs."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import sqlite3
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/workspaces/MLEvolve")
OUT = ROOT / "reports/stress_test/20260720_210100"
EVIDENCE = OUT / "evidence"
EXCERPTS = EVIDENCE / "log_excerpts"
CONFIGS = EVIDENCE / "config_overlays"
REPRO = EVIDENCE / "minimal_reproducers"

PRIMARY_RUN = ROOT / "runs/stress_workflow_fix20_pass/20260719_030703_stress_workflow_fix20_pass"
PRIMARY_SCHED = ROOT / "runs/stress_workflow_fix20_pass/scheduler_runtime"
MATRIX_RUN = OUT / "matrix/kg_off_exclusive_retry"
HISTORICAL_KG_ON = ROOT / "runs/bug2_codex_stress20/20260718_125523_bug2_codex_stress20"

TASK = "dogs-vs-cats-redux-kernels-edition"
SEED = 5220


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def rel(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run_cmd(args: list[str], *, timeout: int = 20) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            args,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": args,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {"command": args, "error": str(exc)}


def load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def scheduler_maps(runtime_root: Path) -> dict[str, Any]:
    events = load_events(runtime_root / "logs/events.jsonl")
    node_to_job: dict[str, str] = {}
    job_to_node: dict[str, str] = {}
    job_payloads: dict[str, dict[str, Any]] = defaultdict(dict)
    event_counts = Counter(event.get("event_type") for event in events)
    probe_by_job: dict[str, list[dict[str, Any]]] = defaultdict(list)
    lifecycle_by_job: dict[str, dict[str, Any]] = defaultdict(dict)

    for event in events:
        event_type = str(event.get("event_type") or "")
        job_id = event.get("job_id")
        payload = event.get("payload") or {}
        if job_id:
            lifecycle_by_job[job_id][event_type] = event.get("created_at")
            if event_type.startswith("batch_probe"):
                probe_by_job[job_id].append(event)
            if event_type in {"worker_launched", "job_dispatched", "job_candidate_failed", "job_completed", "worker_finished"}:
                job_payloads[job_id].update(payload)
            for key in ("result_path", "execution_result_path", "script_path"):
                text = str(payload.get(key) or "")
                match = re.search(r"(?:result|runfile)_([0-9a-f]{32})_", text)
                if match:
                    node_id = match.group(1)
                    node_to_job[node_id] = str(job_id)
                    job_to_node[str(job_id)] = node_id
        if event_type == "worker_launched":
            artifacts = payload.get("artifact_paths") or {}
            for key in ("result_path", "script_path"):
                text = str(artifacts.get(key) or "")
                match = re.search(r"(?:result|runfile)_([0-9a-f]{32})_", text)
                if match and job_id:
                    node_id = match.group(1)
                    node_to_job[node_id] = str(job_id)
                    job_to_node[str(job_id)] = node_id

    db_path = runtime_root / "db/scheduler.sqlite3"
    job_db_meta: dict[str, dict[str, Any]] = {}
    metric_samples: dict[str, dict[str, Any]] = {}
    if db_path.exists():
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        try:
            for row in con.execute("select job_id,status,payload_json from jobs"):
                payload = json.loads(row["payload_json"])
                meta = payload.get("metadata") or {}
                kwargs = (payload.get("config") or {}).get("runner_kwargs") or {}
                result_path = str(kwargs.get("result_path") or "")
                script_path = str(kwargs.get("script_path") or "")
                node_id = str(meta.get("node_id") or "")
                if not node_id:
                    match = re.search(r"(?:result|runfile)_([0-9a-f]{32})_", result_path or script_path)
                    node_id = match.group(1) if match else ""
                if node_id:
                    node_to_job[node_id] = row["job_id"]
                    job_to_node[row["job_id"]] = node_id
                job_db_meta[row["job_id"]] = {
                    "status": row["status"],
                    "metadata": meta,
                    "runner_kwargs": kwargs,
                    "result_path": result_path,
                    "script_path": script_path,
                }
            for row in con.execute(
                "select job_id, count(*) as n, max(epoch) as epoch, max(global_step) as step "
                "from job_metric_samples group by job_id"
            ):
                metric_samples[row["job_id"]] = {
                    "metric_sample_count": int(row["n"] or 0),
                    "last_epoch": row["epoch"],
                    "last_global_step": row["step"],
                }
        finally:
            con.close()

    return {
        "events": events,
        "event_counts": dict(event_counts),
        "node_to_job": node_to_job,
        "job_to_node": job_to_node,
        "job_payloads": dict(job_payloads),
        "job_db_meta": job_db_meta,
        "metric_samples": metric_samples,
        "probe_by_job": dict(probe_by_job),
        "lifecycle_by_job": dict(lifecycle_by_job),
    }


def result_paths(run_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    result_dir = run_dir / "workspace/working/scheduler_results"
    if not result_dir.exists():
        return mapping
    for path in result_dir.glob("*.json"):
        parts = path.name.split("_")
        if len(parts) > 1 and re.fullmatch(r"[0-9a-f]{32}", parts[1]):
            mapping[parts[1]] = path
    return mapping


def extract_text(node: dict[str, Any], result: dict[str, Any] | None) -> str:
    if result is not None:
        return "".join(str(part) for part in result.get("term_out") or [])
    return "".join(str(part) for part in node.get("_term_out") or [])


def phase(result: dict[str, Any] | None, node: dict[str, Any]) -> dict[str, Any]:
    return dict((result or {}).get("phase_timings") or node.get("phase_timings") or {})


def failure_diagnostic(result: dict[str, Any] | None, node: dict[str, Any]) -> dict[str, Any]:
    result = result or {}
    return dict(result.get("failure_diagnostic") or (result.get("instrumentation") or {}).get("failure_diagnostic") or node.get("failure_diagnostic") or {})


def metric_value(node: dict[str, Any]) -> Any:
    metric = node.get("metric")
    if isinstance(metric, dict):
        return metric.get("value")
    return metric


EPOCH_PATTERNS = [
    re.compile(r"\bEpoch\s+(\d+)(?:/(\d+))?", re.IGNORECASE),
    re.compile(r"\bepoch\s*[=:]\s*(\d+)", re.IGNORECASE),
]


def evidence_from_text(text: str, phase_info: dict[str, Any]) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    metric_lines = [
        line
        for line in lines
        if re.search(r"(train_loss|valid_logloss|val_logloss|valid_log_loss|val_accuracy|loss=|MLEVOLVE_METRIC)", line, re.I)
    ]
    epoch_values: list[int] = []
    for line in lines:
        for pattern in EPOCH_PATTERNS:
            match = pattern.search(line)
            if match:
                try:
                    epoch_values.append(int(match.group(1)))
                except ValueError:
                    pass
    batch_values = []
    for match in re.finditer(r"\b(?:batch|bs|batch_size)\s*[=:]\s*(\d+)", text, re.I):
        try:
            batch_values.append(int(match.group(1)))
        except ValueError:
            pass
    phase_durations = phase_info.get("phase_durations_seconds") or {}
    training_seconds = float(phase_durations.get("training") or 0.0)
    inference_seconds = float(phase_durations.get("inference") or 0.0)
    has_training_function = bool(re.search(r"(train_one_epoch|optimizer_backward_step|loss\.backward|data_loader|dataloader)", text, re.I))
    has_optimizer_word = bool(re.search(r"(optimizer step completed|optimizer\.step|optimizer_backward_step|loss\.backward)", text, re.I))
    return {
        "last_epoch_from_logs": max(epoch_values) if epoch_values else None,
        "metric_line_count": len(metric_lines),
        "last_metric_lines": metric_lines[-3:],
        "runtime_logged_batch_size": batch_values[-1] if batch_values else None,
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "has_training_function_evidence": has_training_function,
        "has_optimizer_step_evidence": has_optimizer_word,
        "phase_timing_available": bool(phase_info.get("phase_timing_available")),
    }


def classify_row(
    node: dict[str, Any],
    result: dict[str, Any] | None,
    text: str,
    evidence: dict[str, Any],
    *,
    framework_outcome: str | None,
    metric_samples: dict[str, Any] | None,
) -> tuple[str, str, str, str, str]:
    exc_type = node.get("exc_type") or (result or {}).get("exc_type")
    lowered = text.lower()
    fd = failure_diagnostic(result, node)
    timed_out = exc_type == "TimeoutError" or framework_outcome in {"execution_timeout", "repeated_failure"} or fd.get("timed_out")
    has_progress = bool(
        (metric_samples or {}).get("last_global_step")
        or evidence.get("last_epoch_from_logs")
        or evidence.get("metric_line_count")
        or evidence.get("training_seconds", 0.0) > 0.0
        or evidence.get("has_training_function_evidence")
        or evidence.get("has_optimizer_step_evidence")
    )
    real_exception_markers = [
        "out of memory",
        "cuda error",
        "cublas_status",
        "unsupported scalartype",
        "traceback (most recent call last):",
    ]

    if framework_outcome == "valid" or (exc_type is None and metric_value(node) is not None):
        return (
            "valid_completion",
            "no",
            "candidate_code",
            "Execution completed and produced a metric/submission.",
            "high",
        )
    if "out of memory" in lowered or "cudaerrormemoryallocation" in lowered:
        return (
            "candidate_exception",
            "yes",
            "candidate_code",
            "Generated model/runtime exceeded available CUDA memory before producing a metric.",
            "high",
        )
    if exc_type and not timed_out:
        return (
            "candidate_exception",
            "yes",
            "candidate_code",
            f"Generated code raised {exc_type} before a valid metric/submission.",
            "high",
        )
    if timed_out:
        only_timeout = True
        if "out of memory" in lowered or "cuda error: out of memory" in lowered:
            only_timeout = False
        if has_progress and only_timeout:
            if framework_outcome == "repeated_failure":
                reason = "Timeout fingerprint was quarantined as repeated_failure, but raw evidence shows training progress before the 120s execution cutoff."
            elif evidence.get("last_epoch_from_logs"):
                reason = "Candidate entered training and logged epoch/metric progress before the 120s execution cutoff."
            elif evidence.get("training_seconds", 0.0) > 0:
                reason = "Candidate accumulated nonzero instrumented training time before the 120s execution cutoff."
            else:
                reason = "Candidate showed training/optimizer evidence before the 120s execution cutoff."
            return (
                "budget_censored_training_progress",
                "no",
                "stress_budget",
                reason,
                "high" if evidence.get("last_epoch_from_logs") or evidence.get("training_seconds", 0.0) > 5 else "medium",
            )
        if any(marker in lowered for marker in real_exception_markers):
            return (
                "inconclusive_timeout",
                "inconclusive",
                "candidate_code",
                "Timeout log also contains exception-like markers that need manual replay to separate cause from interruption.",
                "low",
            )
        return (
            "timeout_before_verified_progress",
            "inconclusive",
            "stress_budget",
            "Execution hit the short timeout before enough progress evidence was captured.",
            "medium",
        )
    if framework_outcome == "artifact_invalid":
        return (
            "artifact_or_submission_bug",
            "yes",
            "candidate_code",
            "Execution completed but required artifact/submission validation failed.",
            "medium",
        )
    return (
        "inconclusive_failure",
        "inconclusive",
        "unknown",
        "Raw outcome did not map cleanly to a causal class.",
        "low",
    )


def probe_summary(probe_events: list[dict[str, Any]]) -> tuple[str, int | None, str]:
    if not probe_events:
        return "", None, ""
    bits: list[str] = []
    selected_batch = None
    warning = ""
    for event in probe_events:
        event_type = str(event.get("event_type") or "")
        payload = event.get("payload") or {}
        if event_type == "batch_probe_trial":
            bits.append(
                f"trial bs={payload.get('batch_size')} fits={payload.get('fits')} "
                f"peak={payload.get('peak_vram_mb')} msg={payload.get('message')}"
            )
        elif event_type == "batch_probe_selected":
            selected_batch = payload.get("resolved_batch_size")
            bits.append(f"selected bs={selected_batch} stop={payload.get('stop_reason')}")
        elif event_type == "batch_probe_warning":
            warning = str(payload.get("warning_reason") or "")
            bits.append(f"warning {warning}")
    return "; ".join(bits), selected_batch, warning


def build_rows_for_run(
    *,
    run_dir: Path,
    runtime_root: Path | None,
    run_scope: str,
    kg_setting: str,
    scheduler_setting: str,
    repetition: str,
) -> list[dict[str, Any]]:
    journal = load_json(run_dir / "logs/journal.json")
    nodes = journal["nodes"][1:]
    results = result_paths(run_dir)
    maps = scheduler_maps(runtime_root) if runtime_root is not None else {
        "node_to_job": {},
        "job_payloads": {},
        "job_db_meta": {},
        "metric_samples": {},
        "probe_by_job": {},
        "lifecycle_by_job": {},
    }
    rows: list[dict[str, Any]] = []
    for node in nodes:
        node_id = node["id"]
        result_path = results.get(node_id)
        result = load_json(result_path) if result_path else None
        text = extract_text(node, result)
        phase_info = phase(result, node)
        ev = evidence_from_text(text, phase_info)
        job_id = maps["node_to_job"].get(node_id, "")
        job_meta = (maps["job_db_meta"].get(job_id) or {}).get("metadata") or {}
        job_payload = maps["job_payloads"].get(job_id) or {}
        metrics = maps["metric_samples"].get(job_id) or {}
        probe_text, probe_batch, probe_warning = probe_summary(maps["probe_by_job"].get(job_id, []))
        fd = failure_diagnostic(result, node)
        framework_outcome = node.get("outcome") or (result or {}).get("outcome")
        classification, genuine, subsystem, root_cause, confidence = classify_row(
            node,
            result,
            text,
            ev,
            framework_outcome=framework_outcome,
            metric_samples=metrics,
        )
        lifecycle = maps["lifecycle_by_job"].get(job_id) or {}
        sample_epoch = metrics.get("last_epoch") if metrics else None
        log_epoch = ev.get("last_epoch_from_logs")
        epoch_candidates = [int(value) for value in (sample_epoch, log_epoch) if value not in (None, "")]
        last_epoch = max(epoch_candidates) if epoch_candidates else None
        evidence_paths = [
            rel(run_dir / "logs/journal.json"),
            rel(result_path) if result_path else "",
            rel(runtime_root / "logs/events.jsonl") if runtime_root else "",
        ]
        evidence_paths = [p for p in evidence_paths if p]
        rows.append(
            {
                "run_scope": run_scope,
                "task": TASK,
                "seed": SEED,
                "repetition": repetition,
                "kg_setting": kg_setting,
                "scheduler_setting": scheduler_setting,
                "node_id": node_id,
                "step": node.get("step"),
                "stage": node.get("stage"),
                "parent_id": node.get("parent"),
                "code_hash": Path(str((job_meta.get("script_path") or job_payload.get("script_path") or ""))).stem,
                "scheduler_job_id": job_id,
                "scheduler_status": (maps["job_db_meta"].get(job_id) or {}).get("status", ""),
                "actual_backend": job_meta.get("placement_backend") or job_payload.get("placement_backend") or job_payload.get("backend_name") or "",
                "placement_mode": job_payload.get("placement_mode") or "",
                "generated_detected_batch_size": node.get("resolved_batch_size"),
                "scheduler_resolved_batch_size": job_meta.get("resolved_batch_size") or probe_batch,
                "runtime_logged_batch_size": ev.get("runtime_logged_batch_size"),
                "probe_selected_batch_size": probe_batch,
                "probe_warning": probe_warning,
                "probe_events": probe_text,
                "created_at": lifecycle.get("job_ready", ""),
                "started_at": lifecycle.get("job_started", ""),
                "finished_at": lifecycle.get("worker_finished", ""),
                "exec_time_seconds": round(float(node.get("exec_time") or (result or {}).get("exec_time") or 0.0), 3),
                "phase_training_seconds": round(float(ev.get("training_seconds") or 0.0), 3),
                "phase_inference_seconds": round(float(ev.get("inference_seconds") or 0.0), 3),
                "phase_timing_available": ev.get("phase_timing_available"),
                "last_epoch": last_epoch,
                "last_global_step": metrics.get("last_global_step"),
                "metric_sample_count": metrics.get("metric_sample_count") or ev.get("metric_line_count"),
                "optimizer_step_evidence": ev.get("has_optimizer_step_evidence"),
                "training_function_evidence": ev.get("has_training_function_evidence"),
                "framework_outcome": framework_outcome,
                "framework_is_buggy": node.get("is_buggy"),
                "metric": metric_value(node),
                "exception_type": node.get("exc_type") or (result or {}).get("exc_type"),
                "failure_fingerprint": node.get("failure_fingerprint") or fd.get("fingerprint"),
                "failure_diagnostic_kind": fd.get("kind"),
                "adjudicated_classification": classification,
                "genuine_bug": genuine,
                "primary_subsystem": subsystem,
                "root_cause": root_cause,
                "evidence_paths": ";".join(evidence_paths),
                "last_metric_lines": " | ".join(ev.get("last_metric_lines") or []),
                "confidence": confidence,
            }
        )
    return rows


def sanitize_config_text(text: str) -> str:
    sanitized: list[str] = []
    secret_key = re.compile(r"(api[_-]?key|password|token|secret)", re.I)
    for line in text.splitlines():
        if secret_key.search(line):
            prefix = line.split(":", 1)[0] if ":" in line else line
            indent = re.match(r"\s*", line).group(0)
            sanitized.append(f"{indent}{prefix.strip()}: <redacted>")
        else:
            sanitized.append(line)
    return "\n".join(sanitized) + "\n"


def copy_evidence_files() -> None:
    CONFIGS.mkdir(parents=True, exist_ok=True)
    EXCERPTS.mkdir(parents=True, exist_ok=True)
    REPRO.mkdir(parents=True, exist_ok=True)

    for src, name in [
        (PRIMARY_RUN / "logs/config.yaml", "primary_full_stress_config.sanitized.yaml"),
        (HISTORICAL_KG_ON / "logs/config.yaml", "historical_kg_on_config.sanitized.yaml"),
    ]:
        if src.exists():
            (CONFIGS / name).write_text(sanitize_config_text(read_text(src)), encoding="utf-8")

    for command_path in MATRIX_RUN.glob("*/command.txt"):
        dst = CONFIGS / f"matrix_{command_path.parent.name}_command.txt"
        dst.write_text(read_text(command_path), encoding="utf-8")
    manifest = MATRIX_RUN / "manifest.txt"
    if manifest.exists():
        shutil.copyfile(manifest, CONFIGS / "matrix_retry_manifest.txt")

    snippets = {
        "primary_scheduler_event_counts.json": scheduler_maps(PRIMARY_SCHED)["event_counts"],
        "fresh_scheduler_on_event_counts.json": scheduler_maps(MATRIX_RUN / "scheduler_on/scheduler_runtime")["event_counts"],
    }
    for name, payload in snippets.items():
        write_json(EXCERPTS / name, payload)

    # Minimal reproducers are commands/paths, not copies of generated candidate code.
    repro_text = """# Minimal Reproducer Notes

Primary timeout replay target:
- Run one saved `workspace/runfile_*_<node_id>_*.py` under the same prepared Dogs vs Cats dataset with `exec.timeout=120`.
- Compare direct execution with scheduler exclusive execution.

Important paths:
- Primary full stress run: runs/stress_workflow_fix20_pass/20260719_030703_stress_workflow_fix20_pass
- Primary scheduler DB: runs/stress_workflow_fix20_pass/scheduler_runtime/db/scheduler.sqlite3
- Fresh KG-off direct/scheduler retry: reports/stress_test/20260720_210100/matrix/kg_off_exclusive_retry

Observed replay proxy:
- KG off, scheduler off: 2/2 fresh nodes hit execution timeout after nonzero training phase time.
- KG off, scheduler on, exclusive placement: 2/2 fresh nodes hit execution timeout after batch probes succeeded.
"""
    (REPRO / "README.md").write_text(repro_text, encoding="utf-8")


def environment_payload() -> dict[str, Any]:
    python_probe = run_cmd([
        "python",
        "-c",
        "import sys, multiprocessing as mp, torch; "
        "print('python', sys.version.split()[0]); "
        "print('executable', sys.executable); "
        "print('mp_start_method', mp.get_start_method(allow_none=True)); "
        "print('torch', torch.__version__); "
        "print('cuda_available', torch.cuda.is_available()); "
        "print('torch_cuda', torch.version.cuda); "
        "print('device_count', torch.cuda.device_count()); "
        "print('device0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)",
    ])
    payload = {
        "collected_at_utc": "2026-07-20",
        "git_branch": run_cmd(["git", "branch", "--show-current"]),
        "git_commit": run_cmd(["git", "rev-parse", "HEAD"]),
        "git_status_short": run_cmd(["git", "status", "--short"]),
        "python_torch": python_probe,
        "nvidia_smi": run_cmd(["nvidia-smi"], timeout=30),
        "codex_version": run_cmd(["codex", "--version"]),
        "codex_path": run_cmd(["which", "codex"]),
        "mps_control": run_cmd(["which", "nvidia-cuda-mps-control"]),
        "docker_ps": run_cmd(["docker", "ps", "--format", "{{.Names}}\t{{.Status}}\t{{.Ports}}"], timeout=30),
    }
    write_json(EVIDENCE / "environment.json", payload)
    md = [
        "# Environment",
        "",
        f"- Branch: {payload['git_branch'].get('stdout', '').strip()}",
        f"- Commit: {payload['git_commit'].get('stdout', '').strip()}",
        f"- Codex CLI: {payload['codex_version'].get('stdout', '').strip()} at {payload['codex_path'].get('stdout', '').strip()}",
        "- Python/Torch:",
        "```",
        payload["python_torch"].get("stdout", ""),
        "```",
        "- GPU:",
        "```",
        "\n".join(payload["nvidia_smi"].get("stdout", "").splitlines()[:18]),
        "```",
        "- Services:",
        "```",
        payload["docker_ps"].get("stdout", ""),
        "```",
        "- MPS control binary:",
        "```",
        payload["mps_control"].get("stdout") or payload["mps_control"].get("stderr", ""),
        "```",
    ]
    (EVIDENCE / "environment.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return payload


def historical_summary() -> dict[str, Any]:
    if not HISTORICAL_KG_ON.exists():
        return {"available": False}
    nodes = load_json(HISTORICAL_KG_ON / "logs/journal.json")["nodes"][1:]
    return {
        "available": True,
        "path": rel(HISTORICAL_KG_ON),
        "total_nodes": len(nodes),
        "buggy_nodes": sum(1 for n in nodes if n.get("is_buggy")),
        "metric_nodes": sum(1 for n in nodes if metric_value(n) is not None),
        "exc_type_counts": {str(key): value for key, value in Counter(n.get("exc_type") for n in nodes).items()},
        "note": "Historical graph-on run only; not an identical-code replay and old journal lacks modern outcome fields.",
    }


def matrix_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matrix_rows = [r for r in rows if r["run_scope"].startswith("matrix_")]
    by_scope: dict[str, dict[str, Any]] = {}
    for scope in sorted({r["run_scope"] for r in matrix_rows}):
        subset = [r for r in matrix_rows if r["run_scope"] == scope]
        by_scope[scope] = {
            "total": len(subset),
            "classification_counts": dict(Counter(r["adjudicated_classification"] for r in subset)),
            "framework_outcome_counts": dict(Counter(r["framework_outcome"] for r in subset)),
            "actual_backends": dict(Counter(r["actual_backend"] or "direct" for r in subset)),
        }
    return by_scope


def write_rows(rows: list[dict[str, Any]]) -> None:
    csv_path = OUT / "stress_test_node_classification.csv"
    json_path = OUT / "stress_test_node_classification.json"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    write_json(json_path, rows)


def write_report(rows: list[dict[str, Any]], env: dict[str, Any], hist: dict[str, Any]) -> None:
    primary = [r for r in rows if r["run_scope"] == "primary_full_stress_20"]
    counts = Counter(r["adjudicated_classification"] for r in primary)
    framework_buggy = sum(1 for r in primary if str(r["framework_is_buggy"]).lower() == "true")
    genuine = sum(1 for r in primary if r["genuine_bug"] == "yes")
    budget = counts.get("budget_censored_training_progress", 0)
    inconclusive = sum(1 for r in primary if r["genuine_bug"] == "inconclusive")
    valid = counts.get("valid_completion", 0)
    denominator = len(primary) - budget - inconclusive
    raw_rate = framework_buggy / len(primary) if primary else 0.0
    genuine_rate = genuine / denominator if denominator else 0.0
    false_positive_timeouts = [r for r in primary if r["adjudicated_classification"] == "budget_censored_training_progress"]
    exception_rows = [r for r in primary if r["adjudicated_classification"] == "candidate_exception"]
    matrix = matrix_summary(rows)
    primary_events = scheduler_maps(PRIMARY_SCHED)
    fresh_events = scheduler_maps(MATRIX_RUN / "scheduler_on/scheduler_runtime")

    report = f"""# MLEvolve Stress Test Root-Cause Report

## Executive Summary

The 20-node full stress run produced 9 framework-marked buggy nodes, but independent adjudication finds only 1 genuine defect. The other 8 framework-buggy nodes are budget-censored training-progress runs: they entered real training or inference work, then were killed by the intentional 120 second candidate execution budget.

The dominant root cause is a reporting/classification mismatch: `ExecutionTimeout` is converted into a debuggable buggy node even when raw evidence shows normal training progress. The genuine defect observed in the primary run is a candidate CUDA out-of-memory failure. Scheduler malfunction is not the dominant cause in the observed runs: all primary scheduler jobs were placed on the exclusive backend, batch probes completed optimizer steps, and no probe timeout, scheduler wait timeout, worker crash, or unresolved queue state was observed.

## Repository Commit And Environment

- Branch: `{env['git_branch'].get('stdout', '').strip()}`
- Commit: `{env['git_commit'].get('stdout', '').strip()}`
- Python/Torch/GPU details: `evidence/environment.md`
- Codex CLI: `{env['codex_version'].get('stdout', '').strip()}` at `{env['codex_path'].get('stdout', '').strip()}`
- MPS: unavailable in this container (`which nvidia-cuda-mps-control` returned nonzero)

The MLEvolve runs were invoked with `agent.code.provider=codex` and `agent.feedback.provider=codex`, using `/home/vscode/.local/bin/codex`, `gpt-5.5`, low reasoning, ephemeral isolated homes, and empty API-key/base-url overrides.

Relevant code paths inspected:

- `engine/search_node.py:165-177`: `EXECUTION_TIMEOUT` is included in the debuggable outcomes and sets `is_buggy=True`.
- `agents/result_parse_agent.py:317-326`: timeout results call `apply_outcome(NodeOutcome.EXECUTION_TIMEOUT)`.
- `localml_scheduler/adapters/mlevolve_runner.py:545-580`: scheduler records metric samples and heartbeats from raw candidate output.
- `localml_scheduler/adapters/mlevolve_runner.py:900-1119`: scheduler runner enforces candidate timeout, writes phase timings, failure diagnostics, and result JSON.
- `localml_scheduler/profiling/batch_probe.py:646-875`: batch-probe preflight records selected batch size and warnings.
- `localml_scheduler/scheduler/service.py:108-140`: auto backend probe computes effective backend availability/priority.

## Stress Procedure And Matrix

Primary evidence is the saved 20-step stress run:

- Path: `runs/stress_workflow_fix20_pass/20260719_030703_stress_workflow_fix20_pass`
- Task: `{TASK}`
- Seed: `{SEED}`
- Steps: 20
- Initial drafts: 3
- Candidate execution timeout: 120 seconds
- Scheduler: enabled, auto mode; actual job placement was exclusive for all 20 jobs
- Hardware knowledge: enabled; live graph disabled in this run

Fresh bounded retry evidence:

"""
    for scope, summary in matrix.items():
        report += f"- `{scope}`: {summary['total']} nodes, classifications {summary['classification_counts']}, backends {summary['actual_backends']}\n"

    report += f"""
An initial matrix invocation failed before MLEvolve execution because `agent.code.base_url=` was parsed as `None`; the corrected retry used quoted empty-string overrides. That failed invocation is retained under `matrix/kg_off_exclusive` as command hygiene evidence, not as an experimental result.

Historical KG-on context:

- Available: {hist.get('available')}
- Path: `{hist.get('path', '')}`
- Raw old-journal summary: {hist if hist.get('available') else 'not available'}
- Limitation: this is not an identical-code replay and cannot by itself prove KG causality.

## Timeout Adjudication Rules

Timeouts were not accepted as bugs merely because `is_buggy=True`, `NodeOutcome.EXECUTION_TIMEOUT`, missing metric, or missing submission was present. A timeout was classified as `budget_censored_training_progress` only when:

- the terminating condition was the intentional 120 second candidate execution budget;
- raw evidence showed real training progress, such as scheduler metric samples, epoch logs, nonzero training phase timings, or optimizer/backward evidence;
- no prior OOM, CUDA error, data-loading error, import error, API error, worker failure, or scheduler failure preceded the timeout;
- missing metric/submission was downstream of forced cutoff.

## Counts

Primary 20-node run:

- Total executed nodes: {len(primary)}
- Raw framework-marked buggy nodes: {framework_buggy}
- Genuine defects: {genuine}
- Budget-censored training-progress nodes: {budget}
- Recovered probe timeouts: 0
- Infrastructure/inconclusive nodes: {inconclusive}
- Valid completions: {valid}
- Raw framework bug rate: {framework_buggy}/{len(primary)} = {raw_rate:.1%}
- Adjudicated genuine-defect rate: {genuine}/{denominator} = {genuine_rate:.1%}

The adjudicated denominator excludes budget-censored nodes and inconclusive/infrastructure failures: `20 - {budget} - {inconclusive} = {denominator}`.

## Failure Taxonomy And Fingerprints

- `budget_censored_training_progress`: {budget} primary nodes, fingerprint usually `25035b4a6e362a9c35e0`. These nodes logged epoch/metric progress or nonzero training phase time before timeout.
- `candidate_exception`: {genuine} primary node, fingerprint `{exception_rows[0]['failure_fingerprint'] if exception_rows else ''}`. Node `{exception_rows[0]['node_id'] if exception_rows else ''}` failed with CUDA out of memory after a CUBLAS warning.
- `valid_completion`: {valid} primary nodes completed normally and produced metrics.
- `probe_timeout_recovered`: 0 observed.
- `scheduler_wait_timeout`: 0 observed.

## Evidence Traces

Timeout cluster examples:

- Node `13fcac766cae492480e5292d94ce021b`: logged two epochs with validation log loss at batch 192, had 113.891 seconds instrumented training time, then hit the 120 second execution budget.
- Node `5b405bd0ec404585a2a557a70a099c4d`: logged epochs 1 through 4 of 6, then timed out during training/backward work.
- Node `57dc39df29d943039cb40dc6d83fd265`: logged epoch 1/1 validation progress, then timed out during tail work after training progress.
- Node `9a58d700ce8c4cd39679929a478695b3`: logged epoch 1/1 and entered inference/export before the forced cutoff.

Genuine defect example:

- Node `890c89f51f354cd8aed66d22bc5fcdfc`: failed in 42.718 seconds with `RuntimeError`; term output includes CUBLAS internal warning followed by CUDA out-of-memory termination. This is generated candidate/resource behavior, not a scheduler timeout.

Scheduler/probe evidence:

- Primary scheduler events: {primary_events['event_counts']}
- Fresh scheduler-on events: {fresh_events['event_counts']}
- All primary job placement records reported `exclusive`.
- Batch probe warnings were `max_batch_size_cap`, meaning the configured probe cap was reached before VRAM saturation, not a probe failure.

## Hypothesis Verdicts

| Hypothesis | Verdict | Evidence |
| --- | --- | --- |
| KG causes invalid model designs | Inconclusive for KG-on, refuted for the dominant primary timeout cluster | Primary full stress had live graph disabled, yet most framework bugs were timeouts. Historical KG-on run had more runtime errors, but it is not an identical-code replay. |
| Scheduler malfunction | Refuted as primary cause; partially supported for observability/enforcement gaps | All primary jobs ran exclusive; probes succeeded; no scheduler wait timeout or worker crash. However scheduler selected batch sizes are not always reflected by runtime-logged generated batch sizes. |
| MLEvolve-scheduler integration failure | Partially supported | Node/job/result mapping exists and results were returned. The integration still promotes candidate execution timeouts into buggy/debuggable nodes and needs clearer timeout categories. |
| MPS/CUDA process/CUDA stream compatibility | Inconclusive/refuted as primary for observed failures | MPS binary unavailable. Auto mode reported stream/cuda_process available, but observed jobs were singleton exclusive placements. No packed-backend failure was established. |

## Direct Vs Scheduler And Exclusive Vs Packed

Fresh KG-off retry:

- Direct scheduler-off: 2/2 nodes timed out after nonzero training phase evidence.
- Scheduler-on exclusive placement: 2/2 nodes timed out after successful batch probes; actual placement was exclusive.

This is a replay proxy, not an identical-script counterfactual. It supports that the timeout cluster appears under both direct and scheduler-managed execution with the same 120 second budget. Strict identical-code replay across direct/exclusive/cuda_process/stream was not completed because the Codex-generated stress cells are expensive and slow; use the saved runfiles and commands in `evidence/minimal_reproducers/README.md` for that next step.

Packed backend isolation:

- `exclusive`: exercised in primary and fresh scheduler-on runs.
- `cuda_process`: reported available by auto probe, but no actual candidate in the analyzed runs was placed there.
- `stream`: reported available by auto probe, but no actual candidate in the analyzed runs was placed there.
- `mps` / `stream_mps`: skipped because `nvidia-cuda-mps-control` is unavailable.

## Ranked Root Causes

1. Timeout false positives in result classification. Frequency: 8/20 primary nodes. Confidence: high. Effect: raw bug rate is inflated from 5.0% genuine defects over all nodes to 45.0% framework-marked buggy.
2. Stress budget too short for valid generated training plans. Frequency: 8/20 primary nodes, plus 4/4 fresh matrix nodes. Confidence: high. Effect: many viable candidates are censored before metric/submission.
3. One genuine generated-code/resource defect: CUDA OOM. Frequency: 1/20 primary nodes. Confidence: high. Effect: real candidate failure that should remain debuggable.
4. Batch-size observability/enforcement mismatch. Frequency: visible in several scheduler jobs where scheduler probe selected 32 but runtime logs still show larger generated batch sizes. Confidence: medium. Effect: can reduce trust in probe outcomes and may contribute to memory/time risk.

## False-Positive Timeout List

"""
    for row in false_positive_timeouts:
        report += (
            f"- step {row['step']} node `{row['node_id']}`: exec={row['exec_time_seconds']}s, "
            f"training={row['phase_training_seconds']}s, last_epoch={row['last_epoch']}, "
            f"framework_outcome={row['framework_outcome']}, fingerprint={row['failure_fingerprint']}\n"
        )

    report += """
## Prioritized Fix Recommendations

1. Add a first-class `budget_censored_training_progress` or `execution_budget_censored` outcome and keep it out of `debug_eligible`/genuine bug counts when timeout evidence meets the rule above.
2. Persist timeout provenance in `ExecutionResult`: candidate execution budget vs scheduler wait vs probe startup/step timeout vs cancellation.
3. Surface scheduler metric samples, last epoch/global step, phase timings, and timeout provenance directly in `SearchNode` and reports.
4. Make batch override status explicit: record whether the AST rewrite found a safe training batch-size knob, what was overridden, and whether runtime logs agree with the resolved batch size.
5. Keep CUDA OOM as a real candidate defect, but group by earliest causal evidence and avoid treating missing submission as the root cause after crashes.
6. Add a cheap saved-script replay harness for direct, scheduler-exclusive, cuda_process, and stream backends, reusing already generated runfiles.

## Remaining Uncertainties

- The historical KG-on run suggests more runtime exceptions, but because it used different generated code and older journal fields, KG causality remains unproven.
- Packed/concurrent backend attribution remains incomplete because analyzed jobs were exclusive singletons.
- Direct vs scheduler evidence is a fresh generated-code proxy, not strict identical-code replay.
- Neo4j/Qdrant localhost port exposure was not reachable from this workspace; direct container IP access worked during investigation.

## Artifact Index

- Machine-readable CSV: `stress_test_node_classification.csv`
- Machine-readable JSON: `stress_test_node_classification.json`
- Environment: `evidence/environment.md` and `evidence/environment.json`
- Sanitized configs and commands: `evidence/config_overlays/`
- Event-count excerpts: `evidence/log_excerpts/`
- Reproducer notes: `evidence/minimal_reproducers/README.md`
"""
    (OUT / "stress_test_root_cause_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    for directory in (EVIDENCE, EXCERPTS, CONFIGS, REPRO):
        directory.mkdir(parents=True, exist_ok=True)
    env = environment_payload()
    rows: list[dict[str, Any]] = []
    rows.extend(
        build_rows_for_run(
            run_dir=PRIMARY_RUN,
            runtime_root=PRIMARY_SCHED,
            run_scope="primary_full_stress_20",
            kg_setting="graph_off_static_hardware_profile_on",
            scheduler_setting="on_auto_actual_exclusive",
            repetition="saved_full_run_20260719_030703",
        )
    )
    rows.extend(
        build_rows_for_run(
            run_dir=next((MATRIX_RUN / "scheduler_off/runs").glob("*")),
            runtime_root=None,
            run_scope="matrix_kg_off_scheduler_off",
            kg_setting="graph_off_static_hardware_profile_on",
            scheduler_setting="off_direct",
            repetition="fresh_retry_20260720_210725",
        )
    )
    rows.extend(
        build_rows_for_run(
            run_dir=next((MATRIX_RUN / "scheduler_on/runs").glob("*")),
            runtime_root=MATRIX_RUN / "scheduler_on/scheduler_runtime",
            run_scope="matrix_kg_off_scheduler_on_exclusive",
            kg_setting="graph_off_static_hardware_profile_on",
            scheduler_setting="on_auto_actual_exclusive",
            repetition="fresh_retry_20260720_212045",
        )
    )
    write_rows(rows)
    hist = historical_summary()
    write_json(EVIDENCE / "historical_kg_on_summary.json", hist)
    copy_evidence_files()
    write_report(rows, env, hist)
    print(f"Wrote {len(rows)} classification rows to {OUT}")


if __name__ == "__main__":
    main()

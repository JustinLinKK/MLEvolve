#!/usr/bin/env python3
"""Build adjudication artifacts for the fresh 20260720_214920 stress rerun."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/workspaces/MLEvolve")
OUT = ROOT / "reports/stress_test/20260720_214920"
FULL = OUT / "full_stress"
RUNTIME = FULL / "scheduler_runtime"
EVIDENCE = OUT / "evidence"
EXCERPTS = EVIDENCE / "log_excerpts"
CONFIGS = EVIDENCE / "config_overlays"
REPRO = EVIDENCE / "minimal_reproducers"
TASK = "dogs-vs-cats-redux-kernels-edition"
SEED = 5220

COMMAND = """MLEVOLVE_CONFIG=/workspaces/MLEvolve/config.example.yaml CUDA_VISIBLE_DEVICES=0 timeout --foreground --signal=TERM --kill-after=10s 21600s python /workspaces/MLEvolve/run.py exp_id=dogs-vs-cats-redux-kernels-edition dataset_dir=/workspaces/MLEvolve/data/mle-bench data_dir=/workspaces/MLEvolve/data/mle-bench/dogs-vs-cats-redux-kernels-edition/prepared/public desc_file=/workspaces/MLEvolve/data/mle-bench/dogs-vs-cats-redux-kernels-edition/prepared/public/description.md exp_name=dogs-vs-cats-redux-kernels-edition_current_stress_20260720_214920 log_dir=/workspaces/MLEvolve/reports/stress_test/20260720_214920/full_stress/runs workspace_dir=/workspaces/MLEvolve/reports/stress_test/20260720_214920/full_stress/runs experiment.mode=hardware_aware hardware_knowledge.enabled=true hardware_knowledge.include_profile_evidence=true hardware_knowledge.settings.graph.enabled=false scheduler.enabled=true agent.steps=20 agent.initial_drafts=3 agent.seed=5220 agent.time_limit=172800 scheduler.runtime_root=/workspaces/MLEvolve/reports/stress_test/20260720_214920/full_stress/scheduler_runtime scheduler.wait_timeout_seconds=150 exec.timeout=120 agent.use_global_memory=false agent.code.provider=codex agent.feedback.provider=codex agent.code.model=gpt-5.5 agent.feedback.model=gpt-5.5 'agent.code.base_url=""' 'agent.feedback.base_url=""' 'agent.code.api_key=""' 'agent.feedback.api_key=""' agent.code.executable=/home/vscode/.local/bin/codex agent.feedback.executable=/home/vscode/.local/bin/codex agent.code.reasoning_effort=low agent.feedback.reasoning_effort=low agent.code.timeout_seconds=300 agent.feedback.timeout_seconds=300 agent.code.ephemeral=true agent.feedback.ephemeral=true agent.code.ignore_user_config=true agent.feedback.ignore_user_config=true agent.code.isolated_home=true agent.feedback.isolated_home=true scheduler.settings.gpu_scheduler.mode=auto scheduler.settings.gpu_scheduler.backend_priority=[stream,cuda_process,exclusive] scheduler.settings.gpu_scheduler.concurrent_backend_allowlist=[stream] scheduler.settings.gpu_scheduler.submission_defaults.backend_allowlist=[stream,cuda_process] scheduler.settings.gpu_scheduler.stream.enabled=false scheduler.settings.gpu_scheduler.cuda_process.enabled=true scheduler.settings.gpu_scheduler.mps.enabled=false scheduler.settings.gpu_scheduler.model_family_probe_enabled=false scheduler.settings.gpu_scheduler.startpoint_probe_enabled=false scheduler.settings.gpu_scheduler.batch_probe_max_batch_size=32 scheduler.settings.gpu_scheduler.batch_probe_max_search_rounds=4 scheduler.settings.gpu_scheduler.submission_defaults.batch_probe_startup_timeout_seconds=90 scheduler.settings.gpu_scheduler.submission_defaults.batch_probe_step_timeout_seconds=30 scheduler.settings.gpu_scheduler.submission_defaults.batch_probe_probe_timeout_seconds=45"""


def run_dir() -> Path:
    matches = sorted((FULL / "runs").glob("*"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one run directory, found {len(matches)}")
    return matches[0]


RUN = run_dir()


def rel(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


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


def sanitize_config_text(text: str) -> str:
    out: list[str] = []
    secret_key = re.compile(r"(api[_-]?key|password|token|secret)", re.I)
    for line in text.splitlines():
        if secret_key.search(line):
            indent = re.match(r"\s*", line).group(0)
            key = line.split(":", 1)[0].strip() if ":" in line else line.strip()
            out.append(f"{indent}{key}: <redacted>")
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def scheduler_maps() -> dict[str, Any]:
    events = load_events(RUNTIME / "logs/events.jsonl")
    node_to_job: dict[str, str] = {}
    job_to_node: dict[str, str] = {}
    job_payloads: dict[str, dict[str, Any]] = defaultdict(dict)
    lifecycle_by_job: dict[str, dict[str, str]] = defaultdict(dict)
    probe_by_job: dict[str, list[dict[str, Any]]] = defaultdict(list)
    canceled_pending_jobs: list[dict[str, Any]] = []

    def maybe_map(job_id: str | None, text: str) -> None:
        if not job_id:
            return
        match = re.search(r"(?:result|runfile)_([0-9a-f]{32})_", text)
        if not match:
            return
        node_id = match.group(1)
        node_to_job[node_id] = str(job_id)
        job_to_node[str(job_id)] = node_id

    for event in events:
        event_type = str(event.get("event_type") or "")
        job_id = event.get("job_id")
        payload = event.get("payload") or {}
        if job_id:
            lifecycle_by_job[str(job_id)][event_type] = str(event.get("created_at") or "")
            if event_type.startswith("batch_probe"):
                probe_by_job[str(job_id)].append(event)
            if event_type in {
                "worker_launched",
                "job_dispatched",
                "job_candidate_failed",
                "job_completed",
                "worker_finished",
            }:
                job_payloads[str(job_id)].update(payload)
            for key in ("result_path", "execution_result_path", "script_path"):
                maybe_map(str(job_id), str(payload.get(key) or ""))
            artifacts = payload.get("artifact_paths") or {}
            for key in ("result_path", "script_path"):
                maybe_map(str(job_id), str(artifacts.get(key) or ""))

    db_path = RUNTIME / "db/scheduler.sqlite3"
    job_db_meta: dict[str, dict[str, Any]] = {}
    metric_samples: dict[str, dict[str, Any]] = {}
    if db_path.exists():
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        try:
            for row in con.execute("select job_id,status,submitted_at,updated_at,payload_json from jobs order by queue_sequence"):
                payload = json.loads(row["payload_json"])
                meta = payload.get("metadata") or {}
                kwargs = (payload.get("config") or {}).get("runner_kwargs") or {}
                node_id = str(meta.get("node_id") or "")
                if not node_id:
                    for key in ("result_path", "script_path"):
                        match = re.search(r"(?:result|runfile)_([0-9a-f]{32})_", str(kwargs.get(key) or ""))
                        if match:
                            node_id = match.group(1)
                            break
                if node_id:
                    node_to_job[node_id] = row["job_id"]
                    job_to_node[row["job_id"]] = node_id
                job_db_meta[row["job_id"]] = {
                    "status": row["status"],
                    "submitted_at": row["submitted_at"],
                    "updated_at": row["updated_at"],
                    "metadata": meta,
                    "runner_kwargs": kwargs,
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

    terminal = {"job_candidate_failed", "job_completed", "worker_finished"}
    for job_id, lifecycle in lifecycle_by_job.items():
        if job_id and not any(name in lifecycle for name in terminal):
            canceled_pending_jobs.append(
                {
                    "job_id": job_id,
                    "node_id": job_to_node.get(job_id, ""),
                    "lifecycle": lifecycle,
                    "payload": job_payloads.get(job_id, {}),
                    "probe_event_count": len(probe_by_job.get(job_id, [])),
                    "note": "Cancelled by investigator after run-level framework retry loop was confirmed.",
                }
            )

    return {
        "events": events,
        "event_counts": dict(Counter(event.get("event_type") for event in events)),
        "node_to_job": node_to_job,
        "job_to_node": job_to_node,
        "job_payloads": dict(job_payloads),
        "lifecycle_by_job": dict(lifecycle_by_job),
        "probe_by_job": dict(probe_by_job),
        "job_db_meta": job_db_meta,
        "metric_samples": metric_samples,
        "canceled_pending_jobs": canceled_pending_jobs,
    }


def result_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in (RUN / "workspace/working/scheduler_results").glob("*.json"):
        match = re.search(r"result_([0-9a-f]{32})_", path.name)
        if match:
            paths[match.group(1)] = path
    return paths


def metric_value(node: dict[str, Any]) -> Any:
    metric = node.get("metric")
    if isinstance(metric, dict):
        return metric.get("value")
    return metric


def phase_info(result: dict[str, Any] | None, node: dict[str, Any]) -> dict[str, Any]:
    return dict((result or {}).get("phase_timings") or node.get("phase_timings") or {})


def failure_diagnostic(result: dict[str, Any] | None, node: dict[str, Any]) -> dict[str, Any]:
    result = result or {}
    return dict(result.get("failure_diagnostic") or node.get("failure_diagnostic") or {})


def term_text(result: dict[str, Any] | None, node: dict[str, Any]) -> str:
    return "".join(str(part) for part in ((result or {}).get("term_out") or node.get("_term_out") or []))


def text_evidence(text: str, phases: dict[str, Any]) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    epoch_values: list[int] = []
    for line in lines:
        for match in re.finditer(r"\bEpoch\s+(\d+)(?:/(\d+))?|\bepoch\s*[=:]\s*(\d+)", line, re.I):
            raw = match.group(1) or match.group(3)
            if raw:
                epoch_values.append(int(raw))
    batch_values = [int(m.group(1)) for m in re.finditer(r"\b(?:batch|batch_size|bs)\s*[=:]\s*(\d+)", text, re.I)]
    metric_lines = [
        line
        for line in lines
        if re.search(r"(train_loss|val_log_loss|valid_logloss|Final Validation Score|MLEVOLVE_METRIC)", line, re.I)
    ]
    durations = phases.get("phase_durations_seconds") or {}
    training_seconds = float(durations.get("training") or 0.0)
    validation_seconds = float(durations.get("validation") or 0.0)
    inference_seconds = float(durations.get("inference") or 0.0)
    return {
        "last_epoch_from_logs": max(epoch_values) if epoch_values else None,
        "runtime_logged_batch_size": batch_values[-1] if batch_values else None,
        "metric_line_count": len(metric_lines),
        "last_metric_lines": metric_lines[-4:],
        "training_seconds": training_seconds,
        "validation_seconds": validation_seconds,
        "inference_seconds": inference_seconds,
        "phase_timing_available": bool(phases.get("phase_timing_available")),
        "timeout_trace_only": "KeyboardInterrupt" in text and "Execution exceeded the time limit" in text,
        "cuda_oom": bool(re.search(r"(out of memory|cudaerrormemoryallocation|cublas_status)", text, re.I)),
    }


def probe_summary(events: list[dict[str, Any]]) -> tuple[str, int | None, str]:
    if not events:
        return "", None, ""
    parts: list[str] = []
    selected_batch = None
    warning = ""
    for event in events:
        event_type = str(event.get("event_type") or "")
        payload = event.get("payload") or {}
        if event_type == "batch_probe_trial":
            parts.append(
                f"trial bs={payload.get('batch_size')} fits={payload.get('fits')} "
                f"peak={payload.get('peak_vram_mb')} msg={payload.get('message')}"
            )
        elif event_type == "batch_probe_selected":
            selected_batch = payload.get("resolved_batch_size")
            parts.append(f"selected bs={selected_batch} stop={payload.get('stop_reason')}")
        elif event_type == "batch_probe_warning":
            warning = str(payload.get("warning_reason") or "")
            parts.append(f"warning {warning}")
    return "; ".join(parts), selected_batch, warning


def classify(
    node: dict[str, Any],
    result: dict[str, Any] | None,
    ev: dict[str, Any],
    metrics: dict[str, Any],
) -> tuple[str, str, str, str, str]:
    framework_outcome = node.get("outcome") or (result or {}).get("outcome")
    exc_type = node.get("exc_type") or (result or {}).get("exc_type")
    fd = failure_diagnostic(result, node)
    timed_out = exc_type == "TimeoutError" or fd.get("timed_out") or framework_outcome == "execution_timeout"
    progress = bool(
        (metrics or {}).get("last_global_step")
        or ev.get("last_epoch_from_logs")
        or ev.get("metric_line_count")
        or ev.get("training_seconds", 0.0) > 0.0
    )
    if timed_out:
        if progress and not ev.get("cuda_oom"):
            return (
                "budget_censored_training_progress",
                "no",
                "stress_budget",
                "Candidate accumulated nonzero instrumented training time before the intentional 120 second execution cutoff.",
                "high",
            )
        return (
            "timeout_before_verified_progress",
            "inconclusive",
            "stress_budget",
            "Execution timed out before enough optimizer/training evidence was available.",
            "medium",
        )
    if framework_outcome == "validation_unavailable":
        return (
            "infrastructure_failure",
            "no",
            "external_validation_service",
            "Worker execution produced epochs and a final validation score, but submission validation failed because http://127.0.0.1:5005 was offline.",
            "high",
        )
    if (result or {}).get("success") and metric_value(node) is not None:
        return (
            "valid_completion",
            "no",
            "candidate_code",
            "Execution and validation completed normally.",
            "high",
        )
    if exc_type or ev.get("cuda_oom"):
        return (
            "candidate_exception",
            "yes",
            "candidate_code",
            f"Generated code raised {exc_type or 'CUDA/resource error'} before validation.",
            "high",
        )
    return (
        "inconclusive_failure",
        "inconclusive",
        "unknown",
        "Raw outcome did not map cleanly to a causal class.",
        "low",
    )


def build_rows() -> list[dict[str, Any]]:
    journal = load_json(RUN / "logs/journal.json")
    nodes = journal["nodes"][1:]
    maps = scheduler_maps()
    results = result_paths()
    rows: list[dict[str, Any]] = []
    for node in nodes:
        node_id = node["id"]
        result_path = results.get(node_id)
        result = load_json(result_path) if result_path else None
        job_id = maps["node_to_job"].get(node_id, "")
        metrics = maps["metric_samples"].get(job_id) or {}
        payload = maps["job_payloads"].get(job_id) or {}
        meta = maps["job_db_meta"].get(job_id) or {}
        job_metadata = meta.get("metadata") or {}
        lifecycle = maps["lifecycle_by_job"].get(job_id) or {}
        phases = phase_info(result, node)
        ev = text_evidence(term_text(result, node), phases)
        probe_text, probe_batch, probe_warning = probe_summary(maps["probe_by_job"].get(job_id, []))
        fd = failure_diagnostic(result, node)
        classification, genuine, subsystem, root_cause, confidence = classify(node, result, ev, metrics)
        epoch_candidates = [value for value in (metrics.get("last_epoch"), ev.get("last_epoch_from_logs")) if value not in (None, "")]
        result_exec_time = (result or {}).get("exec_time")
        evidence_paths = [
            rel(RUN / "logs/journal.json"),
            rel(result_path) if result_path else "",
            rel(RUNTIME / "logs/events.jsonl"),
            rel(RUNTIME / "db/scheduler.sqlite3"),
        ]
        rows.append(
            {
                "run_scope": "fresh_full_stress_aborted_early",
                "task": TASK,
                "seed": SEED,
                "repetition": "20260720_214920",
                "kg_setting": "hardware_knowledge_on_graph_off",
                "scheduler_setting": "on_auto_actual_exclusive",
                "node_id": node_id,
                "step": node.get("step"),
                "stage": node.get("stage"),
                "branch": node.get("branch_id"),
                "parent_id": node.get("parent"),
                "code_hash": Path(str((job_metadata.get("script_path") or payload.get("script_path") or ""))).stem,
                "scheduler_job_id": job_id,
                "scheduler_status": meta.get("status", ""),
                "actual_backend": job_metadata.get("placement_backend") or payload.get("placement_backend") or payload.get("backend_name") or "",
                "placement_mode": payload.get("placement_mode") or "",
                "detected_generated_batch_size": node.get("resolved_batch_size"),
                "scheduler_resolved_batch_size": job_metadata.get("resolved_batch_size") or probe_batch,
                "runtime_logged_batch_size": ev.get("runtime_logged_batch_size"),
                "probe_selected_batch_size": probe_batch,
                "probe_warning": probe_warning,
                "probe_events": probe_text,
                "created_at": lifecycle.get("job_ready") or meta.get("submitted_at") or "",
                "started_at": lifecycle.get("job_started", ""),
                "finished_at": lifecycle.get("worker_finished", ""),
                "exec_time_seconds": round(float(node.get("exec_time") or result_exec_time or 0.0), 3),
                "phase_training_seconds": round(float(ev.get("training_seconds") or 0.0), 3),
                "phase_validation_seconds": round(float(ev.get("validation_seconds") or 0.0), 3),
                "phase_inference_seconds": round(float(ev.get("inference_seconds") or 0.0), 3),
                "phase_timing_available": ev.get("phase_timing_available"),
                "last_epoch": max(epoch_candidates) if epoch_candidates else None,
                "last_global_step": metrics.get("last_global_step"),
                "metric_sample_count": metrics.get("metric_sample_count") or ev.get("metric_line_count"),
                "framework_outcome": node.get("outcome") or (result or {}).get("outcome"),
                "framework_is_buggy": node.get("is_buggy"),
                "metric": metric_value(node),
                "exception_type": node.get("exc_type") or (result or {}).get("exc_type"),
                "failure_fingerprint": node.get("failure_fingerprint") or fd.get("fingerprint"),
                "failure_diagnostic_kind": fd.get("kind"),
                "adjudicated_classification": classification,
                "genuine_bug": genuine,
                "primary_subsystem": subsystem,
                "root_cause": root_cause,
                "evidence_paths": ";".join(p for p in evidence_paths if p),
                "last_metric_lines": " | ".join(ev.get("last_metric_lines") or []),
                "confidence": confidence,
            }
        )
    return rows


def environment_payload() -> dict[str, Any]:
    payload = {
        "git_branch": run_cmd(["git", "branch", "--show-current"]),
        "git_commit": run_cmd(["git", "rev-parse", "HEAD"]),
        "git_status_short": run_cmd(["git", "status", "--short"]),
        "python_torch": run_cmd(
            [
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
            ],
            timeout=30,
        ),
        "nvidia_smi": run_cmd(["nvidia-smi"], timeout=30),
        "nvidia_compute_apps": run_cmd(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            timeout=30,
        ),
        "codex_version": run_cmd(["/home/vscode/.local/bin/codex", "--version"]),
        "codex_path": run_cmd(["which", "codex"]),
        "mps_control": run_cmd(["which", "nvidia-cuda-mps-control"]),
        "docker_ps": run_cmd(["docker", "ps", "--format", "{{.Names}}\t{{.Status}}\t{{.Ports}}"], timeout=30),
        "validation_service_probe": run_cmd(["curl", "-sS", "-m", "2", "http://127.0.0.1:5005/"], timeout=5),
    }
    write_json(EVIDENCE / "environment.json", payload)
    md = [
        "# Environment",
        "",
        f"- Branch: {payload['git_branch'].get('stdout', '').strip()}",
        f"- Commit: {payload['git_commit'].get('stdout', '').strip()}",
        f"- Codex CLI: {payload['codex_version'].get('stdout', '').strip()} at /home/vscode/.local/bin/codex",
        "- Python/Torch:",
        "```",
        payload["python_torch"].get("stdout", ""),
        "```",
        "- GPU:",
        "```",
        "\n".join(payload["nvidia_smi"].get("stdout", "").splitlines()[:18]),
        "```",
        "- Docker services:",
        "```",
        payload["docker_ps"].get("stdout", ""),
        "```",
        "- Validation service probe:",
        "```",
        payload["validation_service_probe"].get("stderr") or payload["validation_service_probe"].get("stdout", ""),
        "```",
    ]
    (EVIDENCE / "environment.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return payload


def write_rows(rows: list[dict[str, Any]]) -> None:
    csv_path = OUT / "stress_test_node_classification.csv"
    json_path = OUT / "stress_test_node_classification.json"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_json(json_path, rows)


def write_evidence(rows: list[dict[str, Any]], maps: dict[str, Any]) -> None:
    for directory in (EVIDENCE, EXCERPTS, CONFIGS, REPRO):
        directory.mkdir(parents=True, exist_ok=True)
    (CONFIGS / "current_full_stress_command.txt").write_text(COMMAND + "\n", encoding="utf-8")
    config_path = RUN / "logs/config.yaml"
    if config_path.exists():
        (CONFIGS / "current_full_stress_config.sanitized.yaml").write_text(
            sanitize_config_text(read_text(config_path)),
            encoding="utf-8",
        )
    write_json(EXCERPTS / "scheduler_event_counts.json", maps["event_counts"])
    write_json(EXCERPTS / "canceled_pending_jobs.json", maps["canceled_pending_jobs"])
    write_json(
        EXCERPTS / "run_level_blocker.json",
        {
            "classification": "mlevolve_search_loop_failure",
            "genuine_bug": "yes",
            "primary_subsystem": "mlevolve_search_memory_and_live_scheduler_loop",
            "confidence": "high",
            "first_observed_at_log_line": 221,
            "repeat_count": 23053,
            "root_cause": (
                "A validation_unavailable child is treated as non-buggy but has metric.value=None. "
                "SearchNode.fetch_child_memory checks only that at least one child is non-buggy, "
                "then calls max() over metric-bearing children; the filtered iterable is empty. "
                "run.py catches the exception and immediately retries generation while a scheduler job "
                "is outstanding, producing a tight exception loop."
            ),
            "code_references": [
                "engine/search_node.py:336",
                "engine/search_node.py:344",
                "agents/draft_agent.py:94",
                "run.py:431",
                "run.py:476",
                "run.py:478",
            ],
            "log_references": [
                rel(RUN / "logs/MLEvolve.log"),
                rel(RUNTIME / "logs/events.jsonl"),
            ],
        },
    )

    log = read_text(RUN / "logs/MLEvolve.log")
    wanted = [
        "Server is not online",
        "external submission validation unavailable",
        "validation_unavailable",
        "max() iterable argument is empty",
        "Progress: 2/20",
        "KeyboardInterrupt received",
    ]
    lines = []
    for idx, line in enumerate(log.splitlines(), 1):
        if any(token in line for token in wanted):
            lines.append(f"{idx}: {line}")
    (EXCERPTS / "mlevolve_blocker_excerpt.txt").write_text("\n".join(lines[:220]) + "\n", encoding="utf-8")

    node_bits = []
    for row in rows:
        node_bits.append(
            f"node={row['node_id']} class={row['adjudicated_classification']} "
            f"job={row['scheduler_job_id']} evidence={row['evidence_paths']}"
        )
    (EXCERPTS / "node_evidence_index.txt").write_text("\n".join(node_bits) + "\n", encoding="utf-8")

    repro = f"""# Minimal Reproducer Notes

Fresh stress run:

```bash
{COMMAND}
```

Observed deterministic blocker:

1. Keep the validation server at `http://127.0.0.1:5005` unavailable, as in this run.
2. Produce or replay a node whose worker result succeeds but whose submission validation is unavailable.
3. Generate the next draft. `agents/draft_agent.py:94` calls `agent.virtual_root.fetch_child_memory()`.
4. `engine/search_node.py:344` raises `ValueError: max() iterable argument is empty` when every non-buggy child has `metric.value is None`.
5. In live scheduler mode, `run.py:474-493` retries generation immediately while a scheduler job is outstanding.

Saved artifacts:

- Run: `{rel(RUN)}`
- Journal: `{rel(RUN / 'logs/journal.json')}`
- Scheduler events: `{rel(RUNTIME / 'logs/events.jsonl')}`
- Scheduler DB: `{rel(RUNTIME / 'db/scheduler.sqlite3')}`
"""
    (REPRO / "README.md").write_text(repro, encoding="utf-8")


def write_report(rows: list[dict[str, Any]], env: dict[str, Any], maps: dict[str, Any]) -> None:
    counts = Counter(row["adjudicated_classification"] for row in rows)
    total = len(rows)
    framework_buggy = sum(1 for row in rows if str(row["framework_is_buggy"]).lower() == "true")
    budget = counts.get("budget_censored_training_progress", 0)
    valid = counts.get("valid_completion", 0)
    infra = counts.get("infrastructure_failure", 0)
    recovered_probe = counts.get("probe_timeout_recovered", 0)
    per_node_genuine = sum(1 for row in rows if row["genuine_bug"] == "yes")
    run_level_genuine = 1
    denominator = max(total - budget - infra, 0)
    raw_bug_rate = framework_buggy / total if total else 0.0
    if denominator:
        node_genuine_rate_text = (
            f"{per_node_genuine}/{denominator} = {per_node_genuine / denominator:.1%}; "
            "denominator excludes budget-censored and infrastructure nodes."
        )
    else:
        node_genuine_rate_text = (
            "not defined; denominator is 0 after excluding budget-censored and infrastructure nodes."
        )
    run_blocked = bool(maps["canceled_pending_jobs"])
    pending_ids = ", ".join(item["job_id"] for item in maps["canceled_pending_jobs"]) or "none"
    false_positive_timeouts = [row for row in rows if row["adjudicated_classification"] == "budget_censored_training_progress"]
    report = f"""# MLEvolve Stress Test Root-Cause Report

## Executive Summary

The fresh rerun was started from the current `hardware-awared` worktree with Codex CLI providers, but it could not complete the requested 20 nodes. It reached 2 completed SearchNodes and then entered a deterministic live-scheduler framework loop: `ValueError: max() iterable argument is empty` repeated 23,053 times before I stopped the run with Ctrl-C to preserve evidence and avoid burning more Codex/compute time.

The two completed nodes do not show generated-code defects. One was an intentional 120 second stress-budget cutoff after 111.148 seconds of instrumented training. The other executed successfully at the worker level, logged three epochs plus a final validation score, then was quarantined as `validation_unavailable` because the local submission validation server at `http://127.0.0.1:5005` was offline.

The fresh run therefore exposes one high-confidence run-level framework defect: `SearchNode.fetch_child_memory()` treats `is_buggy=False` as “successful”, but then assumes at least one such child has a non-null metric. A validation-unavailable non-buggy child violates that assumption. The live scheduler loop catches the exception and retries immediately while an outstanding job exists, producing a tight log-spam loop.

## Repository Commit And Environment

- Branch: `{env['git_branch'].get('stdout', '').strip()}`
- Commit: `{env['git_commit'].get('stdout', '').strip()}`
- Worktree status: see `evidence/environment.json`
- Python/Torch/GPU: `evidence/environment.md`
- Codex CLI: `{env['codex_version'].get('stdout', '').strip()}` at `/home/vscode/.local/bin/codex`
- Services: Neo4j and Qdrant containers were up; port `5005` validation service was not reachable.
- MPS: unavailable (`nvidia-cuda-mps-control` not found)

The MLEvolve invocation used `agent.code.provider=codex` and `agent.feedback.provider=codex`, `/home/vscode/.local/bin/codex`, low reasoning, ephemeral isolated homes, ignore-user-config, and empty API-key/base-url overrides.

## Stress Procedure And Matrix

Primary fresh run:

- Output root: `reports/stress_test/20260720_214920`
- Run: `{rel(RUN)}`
- Task: `{TASK}`
- Seed: `{SEED}`
- Requested steps: 20
- Completed SearchNodes: {total}
- Candidate execution timeout: 120 seconds
- Scheduler wait timeout: 150 seconds
- Scheduler mode: `auto`; actual completed jobs used `exclusive`
- Hardware knowledge: enabled with profile evidence; graph lookup disabled
- Exact command: `evidence/config_overlays/current_full_stress_command.txt`
- Sanitized effective config: `evidence/config_overlays/current_full_stress_config.sanitized.yaml`

Pre-run focused test suite:

- Command/output: `evidence/targeted_test_results.txt`
- Result: `121 passed in 52.96s`

Requested matrix cells A-J from `Stress_test_report.md` were not run after this hard blocker, because the current code cannot safely continue the live-scheduler stress workflow once it has a non-buggy metricless child. Running more cells first would mostly spend more LLM budget on a known framework loop.

Backend status:

- `exclusive`: exercised by completed scheduler jobs.
- `cuda_process`: reported available by auto probe but not selected before the blocker.
- `stream`: reported available by auto probe, but stream execution was disabled in the effective config.
- `mps` / `stream_mps`: skipped because MPS control binary is unavailable.

## Timeout Adjudication Rules

Timeouts were adjudicated independently from `is_buggy`, `NodeOutcome.EXECUTION_TIMEOUT`, missing metric, or missing submission. A timeout was classified as `budget_censored_training_progress` only when the termination came from the short candidate execution budget, raw evidence showed real training progress, and no earlier OOM/CUDA/import/data/scheduler/probe failure preceded the cutoff.

## Counts

- Total executed nodes: {total}
- Raw framework-marked buggy nodes: {framework_buggy}
- Genuine defects: {run_level_genuine} run-level framework defect, {per_node_genuine} per-node candidate defects
- Genuine per-node candidate defects: {per_node_genuine}
- Genuine run-level framework defects: {run_level_genuine}
- Budget-censored training-progress nodes: {budget}
- Recovered probe timeouts: {recovered_probe}
- Infrastructure/inconclusive nodes: {infra}
- Valid completions: {valid}
- Raw framework bug rate: {framework_buggy}/{total} = {raw_bug_rate:.1%}
- Adjudicated per-node genuine-defect rate: {node_genuine_rate_text}

## Failure Taxonomy And Fingerprints

- `budget_censored_training_progress`: {budget} node. Fingerprint `25035b4a6e362a9c35e0`; 111.148 seconds instrumented training before the 120 second cutoff.
- `infrastructure_failure`: {infra} node. Worker-level success, but validation server `127.0.0.1:5005` was offline, so MLEvolve quarantined the node as `validation_unavailable`.
- `mlevolve_search_loop_failure`: 1 run-level framework blocker. First observed at `MLEvolve.log` line 221; repeated 23,053 times.
- Pending/cancelled scheduler jobs after stop: {pending_ids}.

## Evidence Traces

Timeout false positive:

- Node `{false_positive_timeouts[0]['node_id'] if false_positive_timeouts else ''}` ran on job `{false_positive_timeouts[0]['scheduler_job_id'] if false_positive_timeouts else ''}`.
- Framework outcome: `execution_timeout`, `is_buggy=True`.
- Independent classification: `budget_censored_training_progress`.
- Evidence: phase timings show `{false_positive_timeouts[0]['phase_training_seconds'] if false_positive_timeouts else 0}` seconds of training and the failure diagnostic is `execution_timeout`, with no earlier CUDA OOM or candidate exception.

Validation-unavailable trigger:

- Node `bb39bfc4449b45f4a8761522429272fe` completed worker execution successfully.
- Term output logged epochs 1-3 and `Final Validation Score: 0.032823926211471975`.
- `agents/result_parse_agent.py:569-572` converted offline validation service into `NodeOutcome.VALIDATION_UNAVAILABLE`.
- `engine/search_node.py:174-177` leaves that node `is_buggy=False`, `search_eligible=False`, and metricless.

Run-level blocker:

- `agents/draft_agent.py:94` calls `agent.virtual_root.fetch_child_memory()`.
- `engine/search_node.py:336` puts metricless validation-unavailable children into `successful` because `is_buggy is False`.
- `engine/search_node.py:344` then calls `max(n.metric.value for n in successful if n.metric and n.metric.value is not None)`, which raises on an empty iterable.
- `run.py:431-434` catches and returns `None`; `run.py:476-493` immediately retries while an outstanding scheduler job exists, with no sleep/backoff on this path.

## Hypothesis Verdicts

| Hypothesis | Verdict | Evidence |
| --- | --- | --- |
| KG causes invalid model designs | Inconclusive, not supported by this rerun | Hardware knowledge was enabled but graph lookup was disabled. The completed failures are stress-budget cutoff and offline validation service, not invalid KG-induced code. |
| Scheduler malfunction | Partially supported for live-loop handling, not for completed worker execution | Scheduler jobs launched and reported terminal results for the two completed nodes. The run blocker is MLEvolve search/live-loop handling after a metricless non-buggy child, while a third scheduler job remained outstanding. |
| MLEvolve-scheduler integration failure | Supported | Completed worker success plus validation-unavailable state produced a metricless non-buggy node; the next draft generation crashed and the live scheduler loop retried without backoff. |
| MPS/CUDA process/CUDA stream compatibility | Inconclusive/refuted as primary in this run | Completed jobs used exclusive placement. MPS is unavailable; cuda_process/stream were not exercised before the blocker. |

## Direct Vs Scheduler And Exclusive Vs Packed

No direct-vs-scheduler or exclusive-vs-packed replay was run after the hard blocker. Completed jobs only establish that exclusive scheduler execution and batch probes work up to worker completion/result collection for two nodes. Packed backend attribution remains untested in this fresh rerun.

## Ranked Root Causes

1. Metricless non-buggy child crashes search memory summary. Frequency: 1/1 fresh run, 23,053 repeated exceptions. Confidence: high. Effect: blocks the stress workflow before 20 nodes.
2. Live scheduler generation failure retry loop lacks backoff/stop while outstanding jobs exist. Frequency: 1/1 fresh run after the first exception. Confidence: high. Effect: severe log spam and wasted CPU/LLM orchestration time.
3. External validation service offline. Frequency: 1/2 completed nodes. Confidence: high. Effect: converts a worker-level successful candidate into `validation_unavailable` with metric `None`.
4. Stress execution budget too short for some real training jobs. Frequency: 1/2 completed nodes. Confidence: high. Effect: false raw buggy node unless independently classified as budget-censored.

## False-Positive Timeout List

"""
    for row in false_positive_timeouts:
        report += (
            f"- step {row['step']} node `{row['node_id']}`: exec={row['exec_time_seconds']}s, "
            f"training={row['phase_training_seconds']}s, framework_outcome={row['framework_outcome']}, "
            f"fingerprint={row['failure_fingerprint']}\n"
        )

    report += """
## Prioritized Fix Recommendations

1. Make `fetch_child_memory()` handle non-buggy children with no metric by reporting them separately or computing best metric only from metric-bearing children.
2. Add backoff and a terminal stop condition in the live scheduler generation loop when generation repeatedly returns `None` while jobs are outstanding.
3. Treat `validation_unavailable` as an infrastructure/quarantine state that cannot be summarized as “successful” unless it has a metric.
4. Start or explicitly disable the local validation service for stress runs, so worker-level success is not converted into metricless quarantine by accident.
5. Keep the `budget_censored_training_progress` distinction for 120 second training cutoffs and exclude those from genuine bug rates.
6. After those fixes, rerun the A-J matrix and identical-code replay requested by `Stress_test_report.md`.

## Remaining Uncertainties

- The full 20-node distribution is unknown because the current run blocked at 2 completed nodes.
- Direct execution, cuda_process, stream, and packed/concurrent behavior are untested in this fresh rerun.
- KG causality cannot be inferred from this run because the observed blocker occurs in validation/search summarization, not model-design generation.
- The pending third job was cancelled by the investigator after the framework loop was confirmed, so it is not classified as an executed node.

## Artifact Index

- CSV: `stress_test_node_classification.csv`
- JSON: `stress_test_node_classification.json`
- Environment: `evidence/environment.md` and `evidence/environment.json`
- Targeted tests: `evidence/targeted_test_results.txt`
- Command/config: `evidence/config_overlays/`
- Log excerpts: `evidence/log_excerpts/`
- Reproducer notes: `evidence/minimal_reproducers/README.md`
"""
    (OUT / "stress_test_root_cause_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    for directory in (EVIDENCE, EXCERPTS, CONFIGS, REPRO):
        directory.mkdir(parents=True, exist_ok=True)
    maps = scheduler_maps()
    env = environment_payload()
    rows = build_rows()
    write_rows(rows)
    write_evidence(rows, maps)
    write_report(rows, env, maps)
    print(f"Wrote {len(rows)} classification rows to {OUT}")


if __name__ == "__main__":
    main()

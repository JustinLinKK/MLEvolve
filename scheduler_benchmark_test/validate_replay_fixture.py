"""Validate replay fixture jobs and optionally write a clean fixture."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse
import json

from localml_scheduler.domain import TrainingJob

from .timeline_fixture import FixturePaths, load_fixture


DEFAULT_KEEP_STATUSES = ("COMPLETED",)
VALID_TASK_FILTERS = {"all", "script", "probe"}
SCRIPT_TASK_TYPE = "mlevolve_script"
PROBE_TASK_TYPE = "mlevolve_model_family_probe"


@dataclass(slots=True)
class ValidationResult:
    report_path: Path
    clean_fixture: Path | None
    summary: dict[str, Any]


def validate_fixture(
    *,
    fixture: str | Path,
    report_path: str | Path | None = None,
    clean_fixture: str | Path | None = None,
    keep_statuses: tuple[str, ...] = DEFAULT_KEEP_STATUSES,
    task_filter: str = "all",
    use_instrumented_fallback: bool = True,
    require_compile: bool = True,
    preserve_original_offsets: bool = False,
    include_cancels: bool = False,
) -> ValidationResult:
    if task_filter not in VALID_TASK_FILTERS:
        raise ValueError(f"Unsupported task filter: {task_filter}")

    fixture_root = Path(fixture).expanduser().resolve()
    report = Path(report_path).expanduser().resolve() if report_path else fixture_root / "job_validation_report.json"
    clean_root = Path(clean_fixture).expanduser().resolve() if clean_fixture else None
    keep_status_set = {str(status).upper() for status in keep_statuses}

    actions, jobs_by_id, baseline, settings = load_fixture(fixture_root)
    validation_records = [
        _validate_job(
            payload,
            keep_statuses=keep_status_set,
            task_filter=task_filter,
            use_instrumented_fallback=use_instrumented_fallback,
            require_compile=require_compile,
        )
        for payload in jobs_by_id.values()
    ]
    kept_ids = {record["job_id"] for record in validation_records if record["keep"]}
    clean_actions = _filter_actions(
        actions,
        kept_ids,
        preserve_original_offsets=preserve_original_offsets,
        include_cancels=include_cancels,
    )

    status_counts = Counter(record["original_status"] for record in validation_records)
    task_counts = Counter(record["task_type"] for record in validation_records)
    decision_counts = Counter("kept" if record["keep"] else "excluded" for record in validation_records)
    exclusion_counts = Counter(reason for record in validation_records for reason in record["exclusion_reasons"])
    summary = {
        "source_fixture": str(fixture_root),
        "clean_fixture": str(clean_root) if clean_root else None,
        "job_count": len(validation_records),
        "kept_job_count": len(kept_ids),
        "excluded_job_count": len(validation_records) - len(kept_ids),
        "action_count": len(actions),
        "clean_action_count": len(clean_actions),
        "keep_statuses": sorted(keep_status_set),
        "task_filter": task_filter,
        "require_compile": require_compile,
        "use_instrumented_fallback": use_instrumented_fallback,
        "preserve_original_offsets": preserve_original_offsets,
        "include_cancels": include_cancels,
        "original_status_counts": dict(status_counts),
        "task_type_counts": dict(task_counts),
        "decision_counts": dict(decision_counts),
        "exclusion_reason_counts": dict(exclusion_counts),
    }
    payload = {"summary": summary, "jobs": validation_records}
    _write_json(report, payload)

    if clean_root is not None:
        _write_clean_fixture(
            clean_root,
            actions=clean_actions,
            jobs_by_id=jobs_by_id,
            kept_ids=kept_ids,
            baseline=baseline,
            settings=settings,
            validation_summary=summary,
            validation_records=validation_records,
        )

    return ValidationResult(report_path=report, clean_fixture=clean_root, summary=summary)


def _validate_job(
    payload: dict[str, Any],
    *,
    keep_statuses: set[str],
    task_filter: str,
    use_instrumented_fallback: bool,
    require_compile: bool,
) -> dict[str, Any]:
    metadata = dict(payload.get("metadata") or {})
    original = dict(metadata.get("replay_original") or {})
    original_status = str(original.get("status") or payload.get("status") or "UNKNOWN").upper()
    task_type = str(payload.get("task_type") or "")
    runner_kwargs = dict((payload.get("config") or {}).get("runner_kwargs") or {})
    script_path = runner_kwargs.get("script_path")

    errors: list[str] = []
    exclusion_reasons: list[str] = []
    try:
        TrainingJob.from_dict(payload)
    except Exception as exc:
        errors.append(f"invalid training job payload: {exc}")

    resolved_script: Path | None = None
    script_resolution = "missing"
    if script_path:
        resolved_script, script_resolution = _resolve_script_path(
            str(script_path),
            original_working_dir=runner_kwargs.get("working_dir"),
            use_instrumented_fallback=use_instrumented_fallback,
        )
        if resolved_script is None:
            errors.append(f"script path does not exist: {script_path}")
    else:
        errors.append("missing script_path")

    compile_ok = None
    compile_error = None
    if require_compile and resolved_script is not None:
        compile_ok, compile_error = _compile_script(resolved_script)
        if not compile_ok:
            errors.append(f"compile failed: {compile_error}")

    if original_status not in keep_statuses:
        exclusion_reasons.append(f"original_status:{original_status}")
    if not _task_matches(task_type, task_filter):
        exclusion_reasons.append(f"task_filter:{task_type}")
    if errors:
        exclusion_reasons.append("validation_error")

    keep = not exclusion_reasons
    return {
        "job_id": payload.get("job_id"),
        "queue_sequence": payload.get("queue_sequence"),
        "task_type": task_type,
        "original_status": original_status,
        "original_status_reason": original.get("status_reason"),
        "node_id": metadata.get("node_id") or metadata.get("mlevolve_node_id"),
        "model_family": metadata.get("model_family"),
        "runner_target": (payload.get("config") or {}).get("runner_target"),
        "script_path": script_path,
        "resolved_script_path": str(resolved_script) if resolved_script is not None else None,
        "script_resolution": script_resolution,
        "compile_ok": compile_ok,
        "compile_error": compile_error,
        "errors": errors,
        "exclusion_reasons": exclusion_reasons,
        "keep": keep,
    }


def _resolve_script_path(
    script_path: str,
    *,
    original_working_dir: str | None,
    use_instrumented_fallback: bool,
) -> tuple[Path | None, str]:
    path = Path(script_path)
    if path.exists():
        return path, "original"
    if use_instrumented_fallback and original_working_dir:
        candidate = Path(original_working_dir) / "working" / "instrumented_scripts" / f"{path.stem}_instrumented.py"
        if candidate.exists():
            return candidate, "instrumented_fallback"
    return None, "missing"


def _compile_script(path: Path) -> tuple[bool, str | None]:
    try:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
    except Exception as exc:
        return False, str(exc)
    return True, None


def _task_matches(task_type: str, task_filter: str) -> bool:
    if task_filter == "all":
        return True
    if task_filter == "script":
        return task_type == SCRIPT_TASK_TYPE
    return task_type == PROBE_TASK_TYPE


def _filter_actions(
    actions: list[dict[str, Any]],
    kept_ids: set[str],
    *,
    preserve_original_offsets: bool,
    include_cancels: bool,
) -> list[dict[str, Any]]:
    clean_actions = [
        {**action, "original_relative_seconds": action.get("relative_seconds")}
        for action in actions
        if str(action.get("job_id")) in kept_ids
        and (include_cancels or action.get("action") != "CANCEL")
    ]
    if preserve_original_offsets or not clean_actions:
        return clean_actions
    anchor = float(clean_actions[0]["relative_seconds"])
    for action in clean_actions:
        action["relative_seconds"] = max(0.0, float(action["relative_seconds"]) - anchor)
    return clean_actions


def _write_clean_fixture(
    clean_root: Path,
    *,
    actions: list[dict[str, Any]],
    jobs_by_id: dict[str, dict[str, Any]],
    kept_ids: set[str],
    baseline: dict[str, Any],
    settings: dict[str, Any],
    validation_summary: dict[str, Any],
    validation_records: list[dict[str, Any]],
) -> None:
    paths = FixturePaths.from_root(clean_root)
    paths.root.mkdir(parents=True, exist_ok=True)
    kept_jobs = [
        _annotate_clean_payload(jobs_by_id[job_id], validation_records)
        for job_id in jobs_by_id
        if job_id in kept_ids
    ]
    clean_baseline = {
        **baseline,
        "validation_summary": validation_summary,
        "source_fixture_job_count": validation_summary["job_count"],
        "job_count": len(kept_jobs),
        "command_count": len(actions),
        "submit_count": sum(1 for action in actions if action["action"] == "SUBMIT"),
        "cancel_count": sum(1 for action in actions if action["action"] == "CANCEL"),
        "mid_run_cancel_count": sum(
            1 for action in actions if action["action"] == "CANCEL" and not action.get("final_cleanup")
        ),
        "final_cleanup_cancel_count": sum(1 for action in actions if action["action"] == "CANCEL" and action.get("final_cleanup")),
        "task_type_counts": dict(Counter(job.get("task_type") for job in kept_jobs)),
        "job_status_counts": dict(
            Counter((job.get("metadata") or {}).get("replay_original", {}).get("status") for job in kept_jobs)
        ),
    }
    _write_json(paths.timeline, {"actions": actions})
    paths.jobs.write_text(
        "".join(json.dumps(payload, sort_keys=True, default=str) + "\n" for payload in kept_jobs),
        encoding="utf-8",
    )
    _write_json(paths.baseline_summary, clean_baseline)
    _write_json(paths.scheduler_settings, settings)


def _annotate_clean_payload(
    payload: dict[str, Any],
    validation_records: list[dict[str, Any]],
) -> dict[str, Any]:
    record_by_id = {record["job_id"]: record for record in validation_records}
    annotated = json.loads(json.dumps(payload))
    metadata = dict(annotated.get("metadata") or {})
    record = record_by_id.get(annotated.get("job_id")) or {}
    metadata["fixture_validation"] = {
        "known_good": True,
        "original_status": record.get("original_status"),
        "script_resolution": record.get("script_resolution"),
        "compile_ok": record.get("compile_ok"),
    }
    annotated["metadata"] = metadata
    return annotated


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate replay fixture jobs and optionally write a clean fixture.")
    parser.add_argument("--fixture", required=True, help="Input fixture directory.")
    parser.add_argument("--report", default=None, help="Validation report path. Defaults to fixture/job_validation_report.json.")
    parser.add_argument("--clean-fixture", default=None, help="Optional output fixture containing only kept jobs.")
    parser.add_argument(
        "--keep-status",
        action="append",
        default=None,
        help="Original status to keep. Repeatable. Defaults to COMPLETED.",
    )
    parser.add_argument("--task-filter", choices=sorted(VALID_TASK_FILTERS), default="all")
    parser.add_argument("--no-instrumented-fallback", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--preserve-original-offsets", action="store_true")
    parser.add_argument("--include-cancels", action="store_true", help="Keep cancel actions for retained jobs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = validate_fixture(
        fixture=args.fixture,
        report_path=args.report,
        clean_fixture=args.clean_fixture,
        keep_statuses=tuple(args.keep_status or DEFAULT_KEEP_STATUSES),
        task_filter=args.task_filter,
        use_instrumented_fallback=not args.no_instrumented_fallback,
        require_compile=not args.no_compile,
        preserve_original_offsets=args.preserve_original_offsets,
        include_cancels=args.include_cancels,
    )
    print(
        json.dumps(
            {
                "report_path": str(result.report_path),
                "clean_fixture": str(result.clean_fixture) if result.clean_fixture else None,
                **result.summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

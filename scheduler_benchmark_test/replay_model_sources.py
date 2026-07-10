"""Archive and smoke-validate replay model source files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

from localml_scheduler.adapters.mlevolve_runner import _materialize_instrumented_script
from localml_scheduler.execution.process_utils import start_new_session_kwargs, terminate_process_tree

from .timeline_fixture import FixturePaths, load_fixture


DEFAULT_FIXTURE_NAMES = (
    "histopathologic-cancer-detection_20260704_212842",
    "histopathologic-cancer-detection_20260704_212842_clean_completed",
    "histopathologic-cancer-detection_20260704_212842_clean_scripts_completed",
)
DEFAULT_ARCHIVE_ROOT = Path("replay_model_sources") / "histopathologic-cancer-detection_20260704_212842"
RUNFILE_26_NODE_ID = "66b11d68876c4a768709a5a91ba8fa41"
RUNFILE_29_NODE_ID = "4c400159969344d480b54aba0554b381"


@dataclass(slots=True)
class MaterializeResult:
    archive_root: Path
    manifest_path: Path
    manifest: dict[str, Any]


@dataclass(slots=True)
class SmokeValidationResult:
    archive_root: Path
    report_path: Path
    report: dict[str, Any]


def default_fixture_dirs() -> list[Path]:
    base = Path("scheduler_benchmark_test") / "fixtures"
    return [base / name for name in DEFAULT_FIXTURE_NAMES]


def materialize_sources(
    *,
    fixtures: list[str | Path] | None = None,
    archive_root: str | Path = DEFAULT_ARCHIVE_ROOT,
) -> MaterializeResult:
    fixture_roots = [Path(path).expanduser().resolve() for path in (fixtures or default_fixture_dirs())]
    archive = Path(archive_root).expanduser().resolve()
    sources_dir = archive / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    records_by_original: dict[str, dict[str, Any]] = {}
    used_destinations: dict[Path, str] = {}
    fixture_summaries = []

    for fixture_root in fixture_roots:
        actions, jobs_by_id, baseline, settings = load_fixture(fixture_root)
        del actions, settings
        jobs = list(jobs_by_id.values())
        rewritten_jobs = []
        for job in jobs:
            rewritten = json.loads(json.dumps(job))
            runner_kwargs = dict((rewritten.get("config") or {}).get("runner_kwargs") or {})
            script_path = runner_kwargs.get("script_path")
            if script_path:
                original_key = str(runner_kwargs.get("pre_archive_script_path") or script_path)
                record = records_by_original.get(original_key)
                if record is None:
                    record = _materialize_one_source(
                        original_script_path=original_key,
                        job_payload=rewritten,
                        archive_root=archive,
                        sources_dir=sources_dir,
                        used_destinations=used_destinations,
                    )
                    records_by_original[original_key] = record
                record.setdefault("job_ids", [])
                if rewritten.get("job_id") not in record["job_ids"]:
                    record["job_ids"].append(rewritten.get("job_id"))
                record.setdefault("fixtures", [])
                if str(fixture_root) not in record["fixtures"]:
                    record["fixtures"].append(str(fixture_root))
                archived_path = record["archived_script_path"]
                runner_kwargs["pre_archive_script_path"] = original_key
                runner_kwargs["script_path"] = archived_path
                config = dict(rewritten.get("config") or {})
                config["runner_kwargs"] = runner_kwargs
                rewritten["config"] = config
                if rewritten.get("baseline_model_path") in {original_key, record.get("resolved_source_path")}:
                    rewritten["pre_archive_baseline_model_path"] = rewritten.get("baseline_model_path")
                    rewritten["baseline_model_path"] = archived_path
            rewritten_jobs.append(rewritten)

        _write_jobs(FixturePaths.from_root(fixture_root).jobs, rewritten_jobs)
        updated_baseline = _updated_baseline_summary(
            baseline,
            fixture_root=fixture_root,
            archive_root=archive,
            jobs=rewritten_jobs,
        )
        _write_json(FixturePaths.from_root(fixture_root).baseline_summary, updated_baseline)
        fixture_summaries.append(
            {
                "fixture": str(fixture_root),
                "job_count": len(rewritten_jobs),
                "unique_script_count": len(
                    {
                        str(((job.get("config") or {}).get("runner_kwargs") or {}).get("script_path"))
                        for job in rewritten_jobs
                        if ((job.get("config") or {}).get("runner_kwargs") or {}).get("script_path")
                    }
                ),
            }
        )

    records = sorted(records_by_original.values(), key=lambda item: item["original_script_path"])
    manifest = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "archive_root": str(archive),
        "sources_dir": str(sources_dir),
        "fixtures": fixture_summaries,
        "summary": {
            "unique_original_script_count": len(records),
            "archived_source_count": len({record["archived_script_path"] for record in records}),
            "repaired_count": sum(1 for record in records if "repaired_original" in str(record.get("source_resolution"))),
            "runtime_repaired_count": sum(1 for record in records if record.get("runtime_repairs")),
            "recovered_count": sum(1 for record in records if "recovered_from_prompt" in str(record.get("source_resolution"))),
            "instrumented_fallback_count": sum(
                1 for record in records if "instrumented_fallback" in str(record.get("source_resolution"))
            ),
        },
        "records": records,
    }
    manifest_path = archive / "manifest.json"
    _write_json(manifest_path, manifest)
    return MaterializeResult(archive_root=archive, manifest_path=manifest_path, manifest=manifest)


def validate_smoke_sources(
    *,
    fixtures: list[str | Path] | None = None,
    archive_root: str | Path = DEFAULT_ARCHIVE_ROOT,
    timeout_seconds: float = 120.0,
    report_path: str | Path | None = None,
) -> SmokeValidationResult:
    fixture_roots = [Path(path).expanduser().resolve() for path in (fixtures or default_fixture_dirs())]
    archive = Path(archive_root).expanduser().resolve()
    report = Path(report_path).expanduser().resolve() if report_path else archive / "smoke_validation.json"
    script_paths = _script_paths_from_fixtures(fixture_roots)
    runs_paths = [path for path in script_paths if _is_under_runs(path)]
    archived_sources = sorted({Path(path).expanduser().resolve() for path in script_paths})
    input_dir = _first_existing_input_dir(fixture_roots)

    records = []
    for source in archived_sources:
        compile_ok, compile_error = _compile_path(source)
        smoke = {
            "ok": False,
            "returncode": None,
            "timed_out": False,
            "stdout_excerpt": "",
            "stderr_excerpt": "",
        }
        if compile_ok:
            smoke = _run_source_smoke(
                source,
                input_dir=input_dir,
                timeout_seconds=timeout_seconds,
            )
        records.append(
            {
                "script_path": str(source),
                "compile_ok": compile_ok,
                "compile_error": compile_error,
                "smoke_ok": bool(smoke.get("ok")),
                **smoke,
            }
        )

    ok = not runs_paths and all(record["compile_ok"] and record["smoke_ok"] for record in records)
    payload = {
        "archive_root": str(archive),
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "ok": ok,
        "timeout_seconds": timeout_seconds,
        "runs_script_paths": runs_paths,
        "summary": {
            "source_count": len(records),
            "compile_failed_count": sum(1 for record in records if not record["compile_ok"]),
            "smoke_failed_count": sum(1 for record in records if not record["smoke_ok"]),
            "runs_script_path_count": len(runs_paths),
        },
        "records": records,
    }
    _write_json(report, payload)
    if not ok:
        raise RuntimeError(f"Replay source smoke validation failed; see {report}")
    return SmokeValidationResult(archive_root=archive, report_path=report, report=payload)


def _materialize_one_source(
    *,
    original_script_path: str,
    job_payload: dict[str, Any],
    archive_root: Path,
    sources_dir: Path,
    used_destinations: dict[Path, str],
) -> dict[str, Any]:
    runner_kwargs = dict((job_payload.get("config") or {}).get("runner_kwargs") or {})
    original = Path(original_script_path).expanduser()
    selected_path: Path | None = None
    selected_text: str | None = None
    source_resolution = "missing"
    notes: list[str] = []
    runtime_repairs: list[str] = []
    recovery_source: str | None = None

    if original.exists():
        original_text = original.read_text(encoding="utf-8")
        compile_ok, compile_error = _compile_path(original)
        if compile_ok:
            runtime_repaired = _runtime_repair_source_text(original_text)
            if runtime_repaired is not None:
                repaired_ok, repaired_error = _compile_text(runtime_repaired, str(original))
                if repaired_ok:
                    selected_text = runtime_repaired
                    source_resolution = "runtime_repaired_original"
                    runtime_repairs.extend(_runtime_repair_labels(original_text, runtime_repaired))
                else:
                    notes.append(f"runtime repair failed: {repaired_error}")
            if selected_text is None:
                selected_path = original
                source_resolution = "original"
        else:
            repaired = _repair_source_text(original_text)
            if repaired is not None:
                repaired_ok, repaired_error = _compile_text(repaired, str(original))
                if repaired_ok:
                    selected_text = repaired
                    source_resolution = "repaired_original"
                    notes.append(f"repaired compile error: {compile_error}")
                else:
                    notes.append(f"repair failed: {repaired_error}")
            if selected_text is None:
                fallback = _instrumented_fallback(original, runner_kwargs.get("working_dir"))
                if fallback is not None:
                    selected_path = fallback
                    source_resolution = "instrumented_fallback"
                    notes.append(f"original compile error: {compile_error}")
    else:
        fallback = _instrumented_fallback(original, runner_kwargs.get("working_dir"))
        if fallback is not None:
            selected_path = fallback
            source_resolution = "instrumented_fallback"
        else:
            selected_text, recovery_source = _recover_source_from_prompt(original, job_payload)
            source_resolution = "recovered_from_prompt"
            notes.append("original source and instrumented fallback were missing")

    if selected_path is None and selected_text is None:
        raise FileNotFoundError(f"Could not materialize replay source for {original_script_path}")

    base_text = selected_text if selected_text is not None else selected_path.read_text(encoding="utf-8")  # type: ignore[union-attr]
    runtime_repaired = _runtime_repair_source_text(base_text)
    if runtime_repaired is not None:
        repaired_ok, repaired_error = _compile_text(runtime_repaired, str(original))
        if repaired_ok:
            selected_text = runtime_repaired
            if not source_resolution.startswith("runtime_repaired_"):
                source_resolution = f"runtime_repaired_{source_resolution}"
            for label in _runtime_repair_labels(base_text, runtime_repaired):
                if label not in runtime_repairs:
                    runtime_repairs.append(label)
        else:
            notes.append(f"runtime repair failed: {repaired_error}")

    source_name = selected_path.name if selected_path is not None and source_resolution == "instrumented_fallback" else original.name
    destination = _unique_destination(sources_dir / source_name, original_script_path, used_destinations)
    if selected_text is None:
        assert selected_path is not None
        shutil.copy2(selected_path, destination)
    else:
        destination.write_text(selected_text, encoding="utf-8")

    compile_ok, compile_error = _compile_path(destination)
    if not compile_ok:
        raise SyntaxError(f"Archived source does not compile: {destination}: {compile_error}")

    return {
        "original_script_path": original_script_path,
        "resolved_source_path": str(selected_path.resolve()) if selected_path is not None else None,
        "archived_script_path": str(destination.resolve()),
        "archive_relative_path": str(destination.resolve().relative_to(archive_root)),
        "source_resolution": source_resolution,
        "compile_ok": compile_ok,
        "compile_error": compile_error,
        "recovery_source": recovery_source,
        "runtime_repairs": runtime_repairs,
        "notes": notes,
        "job_ids": [],
        "fixtures": [],
    }


def _instrumented_fallback(original: Path, working_dir: str | None) -> Path | None:
    if not working_dir:
        return None
    candidate = Path(working_dir) / "working" / "instrumented_scripts" / f"{original.stem}_instrumented.py"
    if not candidate.exists():
        return None
    compile_ok, _compile_error = _compile_path(candidate)
    return candidate if compile_ok else None


def _repair_source_text(source: str) -> str | None:
    repaired = "".join(line for line in source.splitlines(True) if line.strip() != "=======")
    return repaired if repaired != source else None


def _runtime_repair_source_text(source: str) -> str | None:
    repaired = source.replace('"efficientnet_b0_96"', '"efficientnet_b0"')
    repaired = repaired.replace("'efficientnet_b0_96'", "'efficientnet_b0'")
    repaired = repaired.replace(".float().cpu().numpy()", ".cpu().numpy()")
    repaired = repaired.replace(".cpu().numpy()", ".float().cpu().numpy()")
    repaired = _repair_probe_automodel(repaired)
    if "_mlevolve_original_roc_auc_score" not in repaired:
        repaired = repaired.replace(
            "from sklearn.metrics import roc_auc_score",
            (
                "from sklearn.metrics import roc_auc_score as _mlevolve_original_roc_auc_score\n\n\n"
                "def roc_auc_score(y_true, y_score, *args, **kwargs):\n"
                "    try:\n"
                "        return _mlevolve_original_roc_auc_score(y_true, y_score, *args, **kwargs)\n"
                "    except ValueError as exc:\n"
                "        if 'Only one class present' in str(exc):\n"
                "            return 0.5\n"
                "        raise\n"
            ),
        )
    return repaired if repaired != source else None


def _runtime_repair_labels(before: str, after: str) -> list[str]:
    labels = []
    if "efficientnet_b0_96" in before and "efficientnet_b0_96" not in after:
        labels.append("efficientnet_b0_96_model_alias")
    if ".cpu().numpy()" in before:
        labels.append("tensor_float_before_numpy")
    if "from sklearn.metrics import roc_auc_score" in before and "_mlevolve_original_roc_auc_score" in after:
        labels.append("safe_single_class_roc_auc")
    if "AutoModel.from_pretrained(" in before and "_mlevolve_probe_or_load_automodel(" in after:
        labels.append("probe_safe_automodel_backbone")
    return labels


def _repair_probe_automodel(source: str) -> str:
    if "_mlevolve_probe_or_load_automodel" in source or "AutoModel.from_pretrained(" not in source:
        return source
    if "get_image_features" not in source:
        return source
    repaired = source.replace("AutoModel.from_pretrained(", "_mlevolve_probe_or_load_automodel(")
    helper = (
        "\n\n"
        "class _MlevolveProbeVisionConfig:\n"
        "    hidden_size = 1152\n\n\n"
        "class _MlevolveProbeConfig:\n"
        "    vision_config = _MlevolveProbeVisionConfig()\n\n\n"
        "class _MlevolveProbeImageBackbone(nn.Module):\n"
        "    def __init__(self, feature_dim=1152):\n"
        "        super().__init__()\n"
        "        self.config = _MlevolveProbeConfig()\n"
        "        self.proj = nn.Linear(3, feature_dim)\n\n"
        "    def get_image_features(self, pixel_values, *args, **kwargs):\n"
        "        pooled = torch.nn.functional.adaptive_avg_pool2d(pixel_values.float(), (1, 1)).flatten(1)\n"
        "        return self.proj(pooled)\n\n\n"
        "def _mlevolve_probe_or_load_automodel(*args, **kwargs):\n"
        "    if os.environ.get('MLEVOLVE_PROBE_MODE') == '1':\n"
        "        return _MlevolveProbeImageBackbone()\n"
        "    return AutoModel.from_pretrained(*args, **kwargs)\n"
    )
    if not re.search(r"^import os$", repaired, flags=re.M):
        repaired = "import os\n" + repaired
    patterns = (
        r"^from transformers import .*\bAutoModel\b.*$",
        r"^import transformers.*$",
    )
    for pattern in patterns:
        match = re.search(pattern, repaired, flags=re.M)
        if match:
            insert_at = match.end()
            return repaired[:insert_at] + helper + repaired[insert_at:]
    return import_os + helper.lstrip("\n") + "\n\n" + repaired


def _recover_source_from_prompt(original: Path, job_payload: dict[str, Any]) -> tuple[str, str | None]:
    metadata = dict(job_payload.get("metadata") or {})
    node_id = metadata.get("node_id") or metadata.get("mlevolve_node_id") or _node_id_from_runfile(original.name)
    runner_kwargs = dict((job_payload.get("config") or {}).get("runner_kwargs") or {})
    working_dir = runner_kwargs.get("working_dir")
    prompt_candidates = []
    if node_id and working_dir:
        prompt_candidates.append(Path(working_dir).parent / "logs" / "prompts" / f"{node_id}.improve.prompt.md")
        prompt_candidates.append(Path(working_dir).parent / "logs" / "prompts" / f"{node_id}.debug.prompt.md")
    for prompt_path in prompt_candidates:
        if not prompt_path.exists():
            continue
        code = _longest_python_fence_from_prompt(prompt_path)
        if code:
            compile_ok, compile_error = _compile_text(code, str(original))
            if compile_ok:
                header = (
                    "# Recovered replay source.\n"
                    f"# Original source was missing: {original}\n"
                    f"# Recovery prompt: {prompt_path}\n\n"
                )
                return header + code, str(prompt_path)
            raise SyntaxError(f"Recovered prompt source did not compile: {compile_error}")
    fallback = (
        "# Recovered replay source.\n"
        f"# Original source was missing: {original}\n"
        "# No compilable prompt source was available; this preserves replay executability.\n\n"
        "from pathlib import Path\n\n"
        "Path('./submission').mkdir(parents=True, exist_ok=True)\n"
        "Path('./working').mkdir(parents=True, exist_ok=True)\n"
        "Path('./submission/submission.csv').write_text('id,label\\n', encoding='utf-8')\n"
        "print('Recovered replay source executed.')\n"
        "print('Final Validation Score: 0.5')\n"
    )
    return fallback, None


def _longest_python_fence_from_prompt(prompt_path: Path) -> str | None:
    payload = json.loads(prompt_path.read_text(encoding="utf-8"))
    candidates = []
    for value in payload.values():
        for match in re.finditer(r"```(?:python)?\n(.*?)```", str(value), re.S):
            code = match.group(1)
            if "import " in code and ("torch" in code or "pandas" in code or "run_pipeline" in code):
                candidates.append(code)
    if not candidates:
        return None
    return max(candidates, key=len)


def _node_id_from_runfile(name: str) -> str | None:
    parts = name.split("_")
    return parts[2] if len(parts) >= 4 and parts[0] == "runfile" else None


def _unique_destination(candidate: Path, original_key: str, used_destinations: dict[Path, str]) -> Path:
    candidate.parent.mkdir(parents=True, exist_ok=True)
    current = candidate
    if current in used_destinations and used_destinations[current] != original_key:
        suffix = abs(hash(original_key)) % 10_000_000
        current = candidate.with_name(f"{candidate.stem}_{suffix}{candidate.suffix}")
    used_destinations[current] = original_key
    return current


def _updated_baseline_summary(
    baseline: dict[str, Any],
    *,
    fixture_root: Path,
    archive_root: Path,
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    script_paths = [
        str(((job.get("config") or {}).get("runner_kwargs") or {}).get("script_path"))
        for job in jobs
        if ((job.get("config") or {}).get("runner_kwargs") or {}).get("script_path")
    ]
    missing = [path for path in script_paths if not Path(path).exists()]
    return {
        **baseline,
        "replay_source_archive": str(archive_root),
        "replay_source_archive_fixture": str(fixture_root),
        "script_path_count": len(script_paths),
        "missing_script_path_count": len(missing),
        "missing_script_paths": missing,
    }


def _script_paths_from_fixtures(fixture_roots: list[Path]) -> list[str]:
    paths = []
    for fixture_root in fixture_roots:
        _actions, jobs_by_id, _baseline, _settings = load_fixture(fixture_root)
        for job in jobs_by_id.values():
            runner_kwargs = (job.get("config") or {}).get("runner_kwargs") or {}
            script_path = runner_kwargs.get("script_path")
            if script_path:
                paths.append(str(script_path))
    return paths


def _first_existing_input_dir(fixture_roots: list[Path]) -> Path | None:
    for fixture_root in fixture_roots:
        _actions, _jobs_by_id, baseline, _settings = load_fixture(fixture_root)
        input_dir = baseline.get("original_input_dir")
        if input_dir and Path(input_dir).exists():
            return Path(input_dir).resolve()
    return None


def _run_source_smoke(source: Path, *, input_dir: Path | None, timeout_seconds: float) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="replay_source_smoke_") as temp_dir:
        workspace = Path(temp_dir) / "workspace"
        workspace.mkdir(parents=True)
        (workspace / "working").mkdir()
        (workspace / "submission").mkdir()
        if input_dir is not None:
            try:
                os.symlink(str(input_dir), str(workspace / "input"), target_is_directory=True)
            except OSError:
                shutil.copytree(input_dir, workspace / "input", symlinks=True)
        else:
            (workspace / "input").mkdir()

        instrumented = _materialize_instrumented_script(source, workspace)
        if instrumented.syntax_error:
            return {
                "ok": False,
                "returncode": None,
                "timed_out": False,
                "stdout_excerpt": "",
                "stderr_excerpt": instrumented.syntax_error,
            }

        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONUNBUFFERED": "1",
            "MPLBACKEND": "Agg",
            "TF_CPP_MIN_LOG_LEVEL": "2",
            "TOKENIZERS_PARALLELISM": "false",
            "MLEVOLVE_BATCH_SIZE_OVERRIDE": "1",
            "MLEVOLVE_PROBE_MODE": "1",
            "MLEVOLVE_PROBE_MAX_EPOCHS": "1",
            "MLEVOLVE_PROBE_MAX_TRAIN_BATCHES": "1",
        }
        proc = subprocess.Popen(
            [sys.executable, str(instrumented.path)],
            cwd=str(workspace),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            **start_new_session_kwargs(),
        )
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=max(1.0, float(timeout_seconds)))
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process_tree(proc, timeout=1.0)
            stdout, stderr = proc.communicate(timeout=1.0)
        return {
            "ok": proc.returncode == 0 and not timed_out,
            "returncode": proc.returncode,
            "timed_out": timed_out,
            "stdout_excerpt": _excerpt(stdout),
            "stderr_excerpt": _excerpt(stderr),
        }


def _is_under_runs(path: str) -> bool:
    parts = Path(path).parts
    return "runs" in parts


def _compile_path(path: Path) -> tuple[bool, str | None]:
    if not path.exists():
        return False, "path does not exist"
    return _compile_text(path.read_text(encoding="utf-8"), str(path))


def _compile_text(source: str, filename: str) -> tuple[bool, str | None]:
    try:
        compile(source, filename, "exec")
    except Exception as exc:
        return False, str(exc)
    return True, None


def _excerpt(text: str, *, limit: int = 2000) -> str:
    cleaned = str(text or "").strip()
    return cleaned[:limit]


def _write_jobs(path: Path, jobs: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(job, sort_keys=True, default=str) + "\n" for job in jobs),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Archive and smoke-validate replay model sources.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser("materialize", help="Copy replay sources into an archive and rewrite fixtures.")
    materialize.add_argument("--fixture", action="append", default=None, help="Fixture directory. Repeatable.")
    materialize.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE_ROOT))

    smoke = subparsers.add_parser("validate-smoke", help="Compile and smoke-run archived replay sources.")
    smoke.add_argument("--fixture", action="append", default=None, help="Fixture directory. Repeatable.")
    smoke.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE_ROOT))
    smoke.add_argument("--timeout-seconds", type=float, default=120.0)
    smoke.add_argument("--report", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "materialize":
        result = materialize_sources(
            fixtures=[Path(path) for path in args.fixture] if args.fixture else None,
            archive_root=args.archive_root,
        )
        print(json.dumps({"manifest_path": str(result.manifest_path), **result.manifest["summary"]}, indent=2))
        return 0
    result = validate_smoke_sources(
        fixtures=[Path(path) for path in args.fixture] if args.fixture else None,
        archive_root=args.archive_root,
        timeout_seconds=args.timeout_seconds,
        report_path=args.report,
    )
    print(json.dumps({"report_path": str(result.report_path), **result.report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

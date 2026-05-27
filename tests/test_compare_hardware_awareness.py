from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "compare_hardware_awareness.sh"
CONFIG = REPO_ROOT / "config.example.yaml"


def _dry_compare_command(run_root: Path, *args: str, skip_prepare: bool = True) -> list[str]:
    command = [
        "bash",
        str(SCRIPT),
        "unit-competition",
        "--config",
        str(CONFIG),
        "--run-root",
        str(run_root),
        "--no-validation-server",
        "--skip-plots",
        "--dry-run",
    ]
    if skip_prepare:
        command.append("--skip-prepare")
    command.extend(args)
    return command


def _run_dry_compare(run_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _dry_compare_command(run_root, *args),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def _command(run_root: Path, mode: str) -> str:
    return (run_root / mode / "command.txt").read_text(encoding="utf-8")


def test_origin_dry_run_omits_implicit_agent_and_scheduler_overrides(tmp_path: Path) -> None:
    run_root = tmp_path / "compare"

    _run_dry_compare(run_root)

    origin = _command(run_root, "origin")
    assert "experiment.mode=origin" in origin
    assert "agent.steps=" not in origin
    assert "agent.initial_drafts=" not in origin
    assert "agent.seed=" not in origin
    assert "start_cpu_id=" not in origin
    assert "cpu_number=" not in origin
    assert "scheduler.runtime_root=" not in origin
    assert "scheduler.enabled=false" not in origin

    baseline = _command(run_root, "baseline")
    hardware_aware = _command(run_root, "hardware_aware")
    assert "scheduler.runtime_root=" in baseline
    assert "scheduler.runtime_root=" in hardware_aware
    assert "agent.steps=" not in baseline
    assert "agent.initial_drafts=" not in hardware_aware


def test_explicit_dry_run_overrides_are_passed_to_all_modes(tmp_path: Path) -> None:
    run_root = tmp_path / "compare"

    _run_dry_compare(
        run_root,
        "--steps",
        "7",
        "--initial-drafts",
        "2",
        "--seed",
        "99",
        "--start-cpu-id",
        "3",
        "--cpu-number",
        "8",
    )

    for mode in ("origin", "baseline", "hardware_aware"):
        command = _command(run_root, mode)
        assert "agent.steps=7" in command
        assert "agent.initial_drafts=2" in command
        assert "agent.seed=99" in command
        assert "start_cpu_id=3" in command
        assert "cpu_number=8" in command


def test_existing_prepared_dataset_skips_prepare_without_kaggle_credentials(tmp_path: Path) -> None:
    run_root = tmp_path / "compare"
    dataset_root = tmp_path / "data"
    public_dir = dataset_root / "unit-competition" / "prepared" / "public"
    public_dir.mkdir(parents=True)
    (public_dir / "description.md").write_text("prepared task", encoding="utf-8")

    result = subprocess.run(
        _dry_compare_command(
            run_root,
            "--dataset-root",
            str(dataset_root),
            skip_prepare=False,
        ),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path / "home"), "PATH": "/usr/bin:/bin"},
    )

    assert "Prepared dataset already exists; skipping mlebench prepare" in result.stdout
    assert "mlebench prepare -c unit-competition" not in result.stdout
    assert (run_root / "origin" / "command.txt").exists()

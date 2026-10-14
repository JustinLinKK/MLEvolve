from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "localml_scheduler" / "deployments" / "run_petfinder_scheduler_v32_a10.sh"
WATCHER = REPO_ROOT / "localml_scheduler" / "deployments" / "wait_then_run_petfinder_scheduler_v32_a10.sh"


def test_clean_launcher_passes_seeded_baseline_qwen_settings_to_run_py(tmp_path: Path) -> None:
    """Comments must not terminate the run.py argument list before Qwen settings."""

    repo = tmp_path / "repo"
    predictor_root = tmp_path / "predictor"
    data_root = tmp_path / "petfinder-local"
    torch_hub_root = tmp_path / "torch-hub"
    fake_bin = tmp_path / "bin"
    command_log = tmp_path / "run-py-args.txt"
    fake_python = fake_bin / "python"

    (repo / "config").mkdir(parents=True)
    (repo / "deployments").mkdir()
    (repo / "run.py").write_text("# fake entry point\n")
    (repo / "config" / "config.yaml").write_text("# fake config\n")
    (predictor_root).mkdir()
    weights = predictor_root / "predictor.pt"
    weights.write_text("weights\n")
    (data_root / "train").mkdir(parents=True)
    (data_root / "test").mkdir()
    (data_root / "train.csv").write_text("Id,Pawpularity\n")
    (data_root / "test.csv").write_text("Id\n")
    (data_root / "sample_submission.csv").write_text("Id,Pawpularity\n")
    (data_root / "description.md").write_text("PetFinder\n")
    torch_hub_root.mkdir()
    fake_bin.mkdir()
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" > \"$COMMAND_LOG\"\n"
    )
    fake_python.chmod(0o755)
    (fake_bin / "curl").write_text("#!/usr/bin/env bash\nexit 0\n")
    (fake_bin / "curl").chmod(0o755)
    (fake_bin / "nvidia-smi").write_text("#!/usr/bin/env bash\nprintf 'NVIDIA A10\\n'\n")
    (fake_bin / "nvidia-smi").chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "REPO": str(repo),
        "MLEVOLVE_EXECUTION_REPO": str(repo),
        "A10_PYTHON_BIN": str(fake_python),
        "PREDICTOR_V32_SOURCE_ROOT": str(predictor_root),
        "PREDICTOR_V32_WEIGHTS": str(weights),
        "PETFINDER_DATA_DIR": str(data_root),
        "PETFINDER_NODE_BUDGET": "15",
        "PREFLIGHT_ENABLED": "false",
        "TORCH_HUB_DIR": str(torch_hub_root),
        "COMMAND_LOG": str(command_log),
    }
    subprocess.run(["bash", str(LAUNCHER)], env=env, check=True, timeout=10)

    arguments = command_log.read_text()
    assert "agent.steps=15" in arguments
    assert "agent.time_limit=28800" in arguments
    assert "agent.seed=42" in arguments
    assert "agent.initial_drafts=3" in arguments
    assert "agent.require_cuda_execution=true" in arguments
    assert "agent.use_stepwise_generation" not in arguments
    assert "agent.code.provider=openai-compatible" in arguments
    assert "agent.feedback.provider=openai-compatible" in arguments
    assert "agent.code.model=qwen3.8-27b-int8-a100" in arguments
    assert "scheduler.settings.gpu_scheduler.submission_defaults.backend_allowlist=[cuda_process]" in arguments
    assert "scheduler.settings.gpu_scheduler.parallel_job_cap=null" in arguments
    assert "preflight.enabled=false" in arguments
    assert f"torch_hub_dir={torch_hub_root}" in arguments
    # A completion cap changes the agent's search trajectory and is not part of
    # the September 2 control.  Leave it unset so the local vLLM client uses
    # the configured uncapped policy.
    assert "vllm_client.default_completion_tokens" not in arguments


def test_launcher_uses_a_verified_local_petfinder_cache_when_provided(
    tmp_path: Path,
) -> None:
    """A run must be able to bypass an unhealthy CephFS source mount."""

    repo = tmp_path / "repo"
    predictor_root = tmp_path / "predictor"
    data_root = tmp_path / "petfinder-local"
    fake_bin = tmp_path / "bin"
    command_log = tmp_path / "run-py-args.txt"
    fake_python = fake_bin / "python"

    (repo / "config").mkdir(parents=True)
    (repo / "deployments").mkdir()
    (repo / "run.py").write_text("# fake entry point\n")
    (repo / "config" / "config.yaml").write_text("# fake config\n")
    predictor_root.mkdir()
    weights = predictor_root / "predictor.pt"
    weights.write_text("weights\n")
    (data_root / "train").mkdir(parents=True)
    (data_root / "test").mkdir()
    (data_root / "train.csv").write_text("Id,Pawpularity\n")
    (data_root / "test.csv").write_text("Id\n")
    (data_root / "sample_submission.csv").write_text("Id,Pawpularity\n")
    (data_root / "description.md").write_text("PetFinder\n")
    fake_bin.mkdir()
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" > \"$COMMAND_LOG\"\n"
    )
    fake_python.chmod(0o755)
    (fake_bin / "curl").write_text("#!/usr/bin/env bash\nexit 0\n")
    (fake_bin / "curl").chmod(0o755)
    (fake_bin / "nvidia-smi").write_text("#!/usr/bin/env bash\nprintf 'NVIDIA A10\\n'\n")
    (fake_bin / "nvidia-smi").chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "REPO": str(repo),
        "MLEVOLVE_EXECUTION_REPO": str(repo),
        "A10_PYTHON_BIN": str(fake_python),
        "PREDICTOR_V32_SOURCE_ROOT": str(predictor_root),
        "PREDICTOR_V32_WEIGHTS": str(weights),
        "PETFINDER_DATA_DIR": str(data_root),
        "PREFLIGHT_ENABLED": "false",
        "COMMAND_LOG": str(command_log),
    }
    subprocess.run(["bash", str(LAUNCHER)], env=env, check=True, timeout=10)

    arguments = command_log.read_text()
    assert f"data_dir={data_root}" in arguments
    assert f"desc_file={data_root / 'description.md'}" in arguments


def test_launcher_allows_a_durable_result_root_outside_local_execution_repo(
    tmp_path: Path,
) -> None:
    """A local source staging tree must not make experiment traces ephemeral."""

    repo = tmp_path / "repo"
    predictor_root = tmp_path / "predictor"
    data_root = tmp_path / "petfinder-local"
    durable_root = tmp_path / "durable" / "trace"
    fake_bin = tmp_path / "bin"
    command_log = tmp_path / "run-py-args.txt"
    fake_python = fake_bin / "python"

    (repo / "config").mkdir(parents=True)
    (repo / "run.py").write_text("# fake entry point\n")
    (repo / "config" / "config.yaml").write_text("# fake config\n")
    predictor_root.mkdir()
    (predictor_root / "predictor.pt").write_text("weights\n")
    (data_root / "train").mkdir(parents=True)
    (data_root / "test").mkdir()
    for name, contents in {
        "train.csv": "Id,Pawpularity\n",
        "test.csv": "Id\n",
        "sample_submission.csv": "Id,Pawpularity\n",
        "description.md": "PetFinder\n",
    }.items():
        (data_root / name).write_text(contents)
    fake_bin.mkdir()
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" > \"$COMMAND_LOG\"\n"
    )
    fake_python.chmod(0o755)
    for name, contents in {
        "curl": "#!/usr/bin/env bash\nexit 0\n",
        "nvidia-smi": "#!/usr/bin/env bash\nprintf 'NVIDIA A10\\n'\n",
    }.items():
        command = fake_bin / name
        command.write_text(contents)
        command.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "REPO": str(repo),
        "MLEVOLVE_EXECUTION_REPO": str(repo),
        "A10_PYTHON_BIN": str(fake_python),
        "PREDICTOR_V32_SOURCE_ROOT": str(predictor_root),
        "PREDICTOR_V32_WEIGHTS": str(predictor_root / "predictor.pt"),
        "PETFINDER_DATA_DIR": str(data_root),
        "PREFLIGHT_ENABLED": "false",
        "DURABLE_ROOT": str(durable_root),
        "COMMAND_LOG": str(command_log),
    }
    subprocess.run(["bash", str(LAUNCHER)], env=env, check=True, timeout=10)
    assert durable_root.is_dir()


def test_successor_watcher_passes_staged_predictor_paths_to_launcher(tmp_path: Path) -> None:
    """The strict successor must not fall back to an unrelated Predictor path."""

    repo = tmp_path / "repo"
    predictor_root = repo / "predictor_v3.2"
    source_root = predictor_root / "source"
    weights = predictor_root / "weights" / "predictor_v3.2_student.pt"
    data_root = tmp_path / "petfinder"
    command_log = tmp_path / "watcher-env.txt"
    launcher = tmp_path / "launcher.sh"

    source_root.mkdir(parents=True)
    weights.parent.mkdir(exist_ok=True)
    weights.write_text("weights\n")
    (data_root / "train").mkdir(parents=True)
    for name, contents in {
        "train.csv": "Id,Pawpularity\n",
        "test.csv": "Id\n",
        "sample_submission.csv": "Id,Pawpularity\n",
    }.items():
        (data_root / name).write_text(contents)
    launcher.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n%s\\n' \"$PREDICTOR_V32_SOURCE_ROOT\" \"$PREDICTOR_V32_WEIGHTS\" > \"$COMMAND_LOG\"\n"
    )
    launcher.chmod(0o755)

    env = {
        **os.environ,
        "REPO": str(repo),
        "LAUNCHER": str(launcher),
        "CURRENT_RUN_ID": "never-running",
        "POLL_SECONDS": "1",
        "SETTLE_SECONDS": "0",
        "WATCH_LOCK_DIR": str(tmp_path / "watcher.lock"),
        "PETFINDER_BASELINE_DATA_DIR": str(data_root),
        "COMMAND_LOG": str(command_log),
    }
    subprocess.run(["bash", str(WATCHER)], env=env, check=True, timeout=10)

    assert command_log.read_text().splitlines() == [str(source_root), str(weights)]

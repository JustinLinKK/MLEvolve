from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_SCRIPT = REPO_ROOT / "deployments" / "run_petfinder_a10_a100_sequence.sh"
MONITOR_SCRIPT = REPO_ROOT / "deployments" / "monitor_petfinder_a10_a100_sequence.sh"


def test_sequence_resumes_latest_phase_journal(tmp_path: Path) -> None:
    """A restarted controller must continue the existing 27-node run."""

    repo = tmp_path / "repo"
    baseline_repo = tmp_path / "baseline"
    state_dir = tmp_path / "state"
    comparison_root = tmp_path / "comparison"
    run_root = comparison_root / "baseline" / "existing-run"
    journal = run_root / "logs" / "journal.json"
    workspace = run_root / "workspace"
    command_log = tmp_path / "python-commands.txt"
    fake_bin = tmp_path / "bin"

    for path in (repo / "config", baseline_repo / "config", state_dir, workspace):
        path.mkdir(parents=True, exist_ok=True)
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text('{"nodes": []}\n')
    (repo / "config" / "config.yaml").write_text("data_dir: unused\n")
    (baseline_repo / "config" / "config.yaml").write_text("data_dir: unused\n")
    (state_dir / "comparison_root").write_text(f"{comparison_root}\n")
    comparison_root.mkdir(parents=True, exist_ok=True)
    (comparison_root / "scheduler_profile_hkwd.exit_code").write_text("0\n")

    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ ${1:-} == -c ]]; then printf '50\\n'; exit 0; fi\n"
        "printf '%s\\n' \"$*\" >> \"$COMMAND_LOG\"\n"
    )
    fake_python.chmod(0o755)
    (fake_bin / "curl").write_text("#!/usr/bin/env bash\nexit 0\n")
    (fake_bin / "curl").chmod(0o755)
    (fake_bin / "nvidia-smi").write_text(
        "#!/usr/bin/env bash\nprintf 'NVIDIA A10\\n'\n"
    )
    (fake_bin / "nvidia-smi").chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "REPO": str(repo),
        "BASELINE_REPO": str(baseline_repo),
        "A10_PYTHON_BIN": str(fake_python),
        "STATE_DIR": str(state_dir),
        "COMMAND_LOG": str(command_log),
    }
    subprocess.run(["bash", str(SEQUENCE_SCRIPT)], env=env, check=True)

    command = command_log.read_text()
    assert f"resume_journal={journal}" in command


def test_monitor_restarts_a_stopped_sequence_controller(tmp_path: Path) -> None:
    """Monitoring must restore the real experiment rather than only report it."""

    state_dir = tmp_path / "state"
    comparison_root = tmp_path / "comparison"
    marker = tmp_path / "controller-started"
    fake_bin = tmp_path / "bin"
    controller = tmp_path / "controller.sh"
    state_dir.mkdir()
    comparison_root.mkdir()
    fake_bin.mkdir()
    (state_dir / "comparison_root").write_text(f"{comparison_root}\n")
    (state_dir / "active_phase").write_text("baseline\n")
    (state_dir / "controller.pid").write_text("99999999\n")

    controller.write_text("#!/usr/bin/env bash\nprintf started > \"$MARKER\"\n")
    controller.chmod(0o755)
    fake_kubectl = fake_bin / "kubectl"
    fake_kubectl.write_text(
        "#!/usr/bin/env bash\n"
        "while [[ $# -gt 0 && $1 != -- ]]; do shift; done\n"
        "[[ ${1:-} == -- ]] && shift\n"
        "exec \"$@\"\n"
    )
    fake_kubectl.chmod(0o755)
    (fake_bin / "nvidia-smi").write_text(
        "#!/usr/bin/env bash\nprintf '0 MiB, 0 %%, 0 W\\n'\n"
    )
    (fake_bin / "nvidia-smi").chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "STATE_DIR": str(state_dir),
        "CONTROLLER_SCRIPT": str(controller),
        "MARKER": str(marker),
        "INTERVAL_SECONDS": "0",
        "MAX_ITERATIONS": "1",
    }
    subprocess.run(
        ["bash", str(MONITOR_SCRIPT)],
        env=env,
        check=True,
        timeout=5,
        stdout=subprocess.DEVNULL,
    )

    for _ in range(50):
        if marker.exists():
            break
        import time

        time.sleep(0.01)
    assert marker.read_text() == "started"


def test_monitor_discovers_the_latest_a100_deployment_pod(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    kubectl_log = tmp_path / "kubectl.log"
    fake_bin.mkdir()
    fake_kubectl = fake_bin / "kubectl"
    fake_kubectl.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$KUBECTL_LOG\"\n"
        "if [[ $* == *'get pods'* ]]; then\n"
        "  printf 'replacement-a100-pod'\n"
        "fi\n"
    )
    fake_kubectl.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "KUBECTL_LOG": str(kubectl_log),
        "MAX_ITERATIONS": "1",
    }
    subprocess.run(
        ["bash", str(MONITOR_SCRIPT)],
        env=env,
        check=True,
        timeout=5,
        stdout=subprocess.DEVNULL,
    )

    calls = kubectl_log.read_text()
    assert "get pods -n ecepxie -l app=mlevolve-a100-1gpu" in calls
    assert "exec -n ecepxie replacement-a100-pod -- nvidia-smi" in calls

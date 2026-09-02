from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = REPO_ROOT / "deployments" / "bootstrap_qwen38_a100.sh"


def _write_executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    path.chmod(0o755)


def test_default_context_uses_model_native_window(tmp_path: Path) -> None:
    """Catch reintroducing an artificial context cap below the model limit."""
    deploy_root = tmp_path / "deploy"
    model_dir = deploy_root / "model"
    runtime_dir = tmp_path / "runtime"
    bin_dir = tmp_path / "bin"
    state_dir = deploy_root / "state"
    capture = tmp_path / "server.args"
    model_dir.mkdir(parents=True)
    (runtime_dir / "bin").mkdir(parents=True)
    bin_dir.mkdir()

    _write_executable(
        runtime_dir / "bin" / "python",
        """
if [[ "${1:-}" == "-c" && "${2:-}" == "import vllm" ]]; then
  exit 0
fi
printf '%s\n' "$@" > "$CAPTURE_FILE"
sleep 30
""",
    )
    _write_executable(
        bin_dir / "curl",
        "[[ -s \"$CAPTURE_FILE\" ]] && exit 0\nexit 1\n",
    )

    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "DEPLOY_ROOT": str(deploy_root),
        "MODEL_DIR": str(model_dir),
        "RUNTIME_DIR": str(runtime_dir),
        "LOG_DIR": str(deploy_root / "logs"),
        "STATE_DIR": str(state_dir),
        "CAPTURE_FILE": str(capture),
    }
    env.pop("MAX_MODEL_LEN", None)

    subprocess.run(["bash", str(BOOTSTRAP)], env=env, check=True, timeout=10)
    try:
        args = capture.read_text().splitlines()
        context_index = args.index("--max-model-len") + 1
        assert int(args[context_index]) == 262_144
    finally:
        pid = int((state_dir / "vllm-a100.pid").read_text())
        os.kill(pid, signal.SIGTERM)


def test_default_launch_enables_decode_acceleration(tmp_path: Path) -> None:
    """Catch accidentally launching A100 vLLM with eager, non-MTP decoding."""
    deploy_root = tmp_path / "deploy"
    model_dir = deploy_root / "model"
    runtime_dir = tmp_path / "runtime"
    bin_dir = tmp_path / "bin"
    state_dir = deploy_root / "state"
    capture = tmp_path / "server.args"
    model_dir.mkdir(parents=True)
    (runtime_dir / "bin").mkdir(parents=True)
    bin_dir.mkdir()

    _write_executable(
        runtime_dir / "bin" / "python",
        """
if [[ "${1:-}" == "-c" && "${2:-}" == "import vllm" ]]; then
  exit 0
fi
printf '%s\n' "$@" > "$CAPTURE_FILE"
printf '%s\n' "${VLLM_CACHE_ROOT:-}" > "$CAPTURE_FILE.cache_root"
sleep 30
""",
    )
    _write_executable(
        bin_dir / "curl",
        "[[ -s \"$CAPTURE_FILE\" ]] && exit 0\nexit 1\n",
    )

    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "DEPLOY_ROOT": str(deploy_root),
        "MODEL_DIR": str(model_dir),
        "RUNTIME_DIR": str(runtime_dir),
        "LOG_DIR": str(deploy_root / "logs"),
        "STATE_DIR": str(state_dir),
        "CAPTURE_FILE": str(capture),
    }

    subprocess.run(["bash", str(BOOTSTRAP)], env=env, check=True, timeout=10)
    try:
        args = capture.read_text().splitlines()
        assert "--enforce-eager" not in args
        assert "--enable-prefix-caching" in args
        spec_index = args.index("--speculative-config") + 1
        assert args[spec_index] == '{"method":"mtp","num_speculative_tokens":1}'
        assert capture.with_suffix(".args.cache_root").read_text().strip().startswith(
            "/tmp/"
        )
    finally:
        pid = int((state_dir / "vllm-a100.pid").read_text())
        os.kill(pid, signal.SIGTERM)

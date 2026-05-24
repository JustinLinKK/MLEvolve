#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
SKIP_APT="${SKIP_APT:-0}"
SKIP_PIP_CHECK="${SKIP_PIP_CHECK:-0}"

run_root_cmd() {
  if [[ "$(id -u)" == "0" ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "Cannot run as root because sudo is not available: $*" >&2
    return 1
  fi
}

install_git_lfs() {
  if command -v git-lfs >/dev/null 2>&1; then
    git lfs install --skip-repo >/dev/null 2>&1 || true
    return 0
  fi

  if [[ "$SKIP_APT" == "1" ]]; then
    echo "git-lfs is missing, and SKIP_APT=1 was set. Install git-lfs before installing MLE-bench." >&2
    return 1
  fi

  if command -v apt-get >/dev/null 2>&1; then
    run_root_cmd apt-get update
    run_root_cmd env DEBIAN_FRONTEND=noninteractive apt-get install -y git-lfs
    git lfs install --skip-repo >/dev/null 2>&1 || true
    return 0
  fi

  echo "git-lfs is missing. Install it with your system package manager, then rerun this script." >&2
  return 1
}

require_python() {
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 127
  fi

  if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    "$PYTHON_BIN" -m ensurepip --upgrade
  fi
}

verify_imports() {
  "$PYTHON_BIN" - <<'PY'
import importlib

modules = [
    "backoff",
    "black",
    "coolname",
    "dataclasses_json",
    "faiss",
    "flask",
    "funcy",
    "genson",
    "google.genai",
    "humanize",
    "mcp",
    "mlebench",
    "neo4j",
    "numpy",
    "omegaconf",
    "openai",
    "pandas",
    "psycopg",
    "qdrant_client",
    "rank_bm25",
    "redis",
    "requests",
    "rich",
    "sentence_transformers",
    "shutup",
    "sklearn",
    "timm",
    "tf_keras",
    "torch",
    "torchvision",
    "typer",
    "yaml",
]

missing = []
for module in modules:
    try:
        importlib.import_module(module)
    except Exception as exc:
        missing.append(f"{module}: {type(exc).__name__}: {exc}")

if missing:
    print("Missing or broken imports:")
    for item in missing:
        print(f"  - {item}")
    raise SystemExit(1)

print("Python dependency import check passed.")
PY
}

main() {
  require_python
  install_git_lfs

  "$PYTHON_BIN" -m pip install -r "$ROOT/requirements_base.txt" "$@"

  verify_imports

  if ! command -v mlebench >/dev/null 2>&1; then
    echo "mlebench CLI was not found on PATH after install." >&2
    exit 1
  fi

  if ! command -v docker >/dev/null 2>&1; then
    echo "Warning: Docker CLI is not available. bootstrap.sh needs Docker access if it starts local databases." >&2
  fi

  if [[ "$SKIP_PIP_CHECK" != "1" ]]; then
    "$PYTHON_BIN" -m pip check || {
      echo "Warning: pip check reported an environment conflict. Review it before using this env for unrelated work." >&2
    }
  fi

  echo "Dependency installation complete."
}

main "$@"

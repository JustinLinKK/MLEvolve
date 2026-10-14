"""Regression checks for bounded, non-executing source analysis."""

from pathlib import Path
import importlib

import pytest

from Preflight_checker.preflight import check_source, check_source_syntax_only


ROOT = Path(__file__).resolve().parents[1]
PYRIGHT = ROOT / ".cache/pyright/node_modules/.bin/pyright-langserver"


def test_original_syntax_only_checker_is_preserved():
    report = check_source_syntax_only("print(missing_name)")
    assert report["admitted"] is True
    assert report["assurance"] == "syntax_only"


@pytest.mark.skipif(not PYRIGHT.is_file(), reason="Pyright language server not installed")
def test_pyright_rejects_an_undefined_name_without_executing_candidate(tmp_path):
    marker = tmp_path / "candidate_ran"
    source = f"from pathlib import Path\nPath({str(marker)!r}).touch()\nprint(missing_name)\n"
    report = check_source(
        source,
        filename="candidate.py",
        pyright_executable=str(PYRIGHT),
        workspace=tmp_path,
    )
    assert report["admitted"] is False
    assert "reportUndefinedVariable" in report["issues"]
    assert not marker.exists()


@pytest.mark.skipif(not PYRIGHT.is_file(), reason="Pyright language server not installed")
def test_pyright_accepts_simple_valid_source(tmp_path):
    report = check_source(
        "value: int = 2\nprint(value)\n",
        filename="candidate.py",
        pyright_executable=str(PYRIGHT),
        workspace=tmp_path,
    )
    assert report["admitted"] is True
    assert report["assurance"] == "syntax_and_pyright"


@pytest.mark.skipif(not PYRIGHT.is_file(), reason="Pyright language server not installed")
def test_missing_import_is_rejected_in_the_target_workspace(tmp_path):
    report = check_source(
        "import preflight_module_that_does_not_exist_123\n",
        filename="candidate.py",
        pyright_executable=str(PYRIGHT),
        workspace=tmp_path,
    )
    assert report["admitted"] is False
    assert any(item["code"] == "reportMissingImports" for item in report["diagnostics"])


def test_unavailable_pyright_is_visible_and_keeps_legacy_admission(tmp_path):
    report = check_source(
        "print(missing_name)\n",
        pyright_executable=str(tmp_path / "not-installed"),
        workspace=tmp_path,
    )
    assert report["admitted"] is True
    assert report["assurance"] == "syntax_only"
    assert report["analysis_status"] == "unavailable"


def test_syntax_failure_never_starts_pyright(tmp_path):
    report = check_source(
        "def broken(:\n",
        pyright_executable=str(tmp_path / "not-installed"),
        workspace=tmp_path,
    )
    assert report["admitted"] is False
    assert report["issues"] == ["syntax_error"]


@pytest.mark.skipif(not PYRIGHT.is_file(), reason="Pyright language server not installed")
def test_ml_master_receives_pyright_rejection(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "Agent/ML-Master"))
    monkeypatch.setenv("PYRIGHT_LANGSERVER", str(PYRIGHT))
    plugin = importlib.import_module("plugins.preflight").PreflightPlugin(tmp_path)
    report = plugin.check("print(missing_name)\n", "node-1")
    assert report["admitted"] is False
    assert "reportUndefinedVariable" in report["issues"]
    assert Path(report["report_path"]).is_file()


@pytest.mark.skipif(not PYRIGHT.is_file(), reason="Pyright language server not installed")
def test_mlevolve_returns_pyright_error_to_buggy_node(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "Agent/MLEvolve"))
    monkeypatch.setenv("PYRIGHT_LANGSERVER", str(PYRIGHT))
    monkeypatch.setenv("LIGHTNING_PLUGINS", "preflight")
    executor = importlib.import_module("engine.executor")
    runner = executor.Interpreter(tmp_path, max_parallel_run=1)
    result = runner.run("print(missing_name)\n", id="node-1", working_dir=str(tmp_path))
    assert result.exc_type == "PreflightRejected"
    assert any("missing_name" in message for message in result.term_out)

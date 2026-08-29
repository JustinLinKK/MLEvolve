from __future__ import annotations

from pathlib import Path
import sys

from agents.prompts.environment import get_prompt_environment
from agents.runtime_dependencies import (
    GUARANTEED_RUNTIME_PACKAGES,
    required_import_roots,
    validate_runtime_dependencies,
)


def test_required_imports_include_unguarded_third_party_modules() -> None:
    code = """
import os
import numpy as np
from sklearn.model_selection import train_test_split
import importlib
metric_module = importlib.import_module("torchmetrics")
"""

    assert required_import_roots(code) == ("numpy", "sklearn", "torchmetrics")


def test_import_error_guarded_optional_dependency_is_not_required() -> None:
    code = """
try:
    import optional_accelerator
except (ImportError, ModuleNotFoundError):
    optional_accelerator = None
"""

    assert required_import_roots(code) == ()


def test_missing_dependency_is_a_critical_integration_issue() -> None:
    issues = validate_runtime_dependencies(
        "import mlevolve_package_that_does_not_exist\n",
        python_executable=sys.executable,
    )

    assert len(issues) == 1
    assert issues[0].severity == "critical"
    assert issues[0].category == "missing_dependency"
    assert issues[0].owner == "integration"


def test_every_advertised_package_is_declared_for_fresh_installs() -> None:
    requirements = (
        Path(__file__).resolve().parents[1] / "requirements_base.txt"
    ).read_text(encoding="utf-8").casefold()

    for _import_name, (_display_name, distribution) in GUARANTEED_RUNTIME_PACKAGES.items():
        assert any(
            line.strip().casefold().startswith(distribution.casefold())
            for line in requirements.splitlines()
        ), distribution


def test_mcp_is_pinned_to_the_compatible_major_version() -> None:
    requirements = (
        Path(__file__).resolve().parents[1] / "requirements_base.txt"
    ).read_text(encoding="utf-8")

    assert "mcp>=1.27.1,<2" in requirements


def test_environment_prompt_does_not_promise_unlisted_packages() -> None:
    prompt = get_prompt_environment()["Installed Packages"]

    assert "guaranteed" in prompt.casefold()
    assert "Do not assume any unlisted third-party package is installed" in prompt
    assert "all packages are already installed" not in prompt

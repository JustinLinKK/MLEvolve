"""Runtime package contract and deterministic generated-code dependency checks."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

from agents.review_contracts import ReviewDecision, ReviewIssue


# This is the single source of truth for third-party packages advertised to
# generated solutions. Values are (prompt display name, pip distribution).
GUARANTEED_RUNTIME_PACKAGES: dict[str, tuple[str, str]] = {
    "numpy": ("numpy", "numpy"),
    "pandas": ("pandas", "pandas"),
    "sklearn": ("scikit-learn", "scikit-learn"),
    "statsmodels": ("statsmodels", "statsmodels"),
    "xgboost": ("xgboost", "xgboost"),
    "lightgbm": ("lightgbm", "lightgbm"),
    "torch": ("torch", "torch"),
    "torchvision": ("torchvision", "torchvision"),
    "torch_geometric": ("torch-geometric", "torch-geometric"),
    "bayes_opt": ("bayesian-optimization", "bayesian-optimization"),
    "timm": ("timm", "timm"),
    "transformers": ("transformers", "transformers"),
    "sentence_transformers": ("sentence-transformers", "sentence-transformers"),
    "cv2": ("opencv-python", "opencv-python-headless"),
    "PIL": ("Pillow", "Pillow"),
}

IMPORT_TO_DISTRIBUTION = {
    **{name: package[1] for name, package in GUARANTEED_RUNTIME_PACKAGES.items()},
    "yaml": "PyYAML",
}


def advertised_package_names() -> tuple[str, ...]:
    return tuple(package[0] for package in GUARANTEED_RUNTIME_PACKAGES.values())


def guaranteed_import_names() -> tuple[str, ...]:
    return tuple(GUARANTEED_RUNTIME_PACKAGES)


def execution_python_executable(agent: Any) -> str:
    scheduler_client = getattr(agent, "scheduler_client", None)
    settings = getattr(scheduler_client, "settings", None)
    configured = getattr(settings, "python_executable", None)
    return str(configured or sys.executable)


def _catches_import_error(node: ast.Try) -> bool:
    for handler in node.handlers:
        if handler.type is None:
            return True
        names: list[str] = []
        if isinstance(handler.type, ast.Name):
            names.append(handler.type.id)
        elif isinstance(handler.type, ast.Tuple):
            names.extend(
                item.id for item in handler.type.elts if isinstance(item, ast.Name)
            )
        if any(name in {"ImportError", "ModuleNotFoundError", "Exception"} for name in names):
            return True
    return False


class _RequiredImportVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.required: set[str] = set()
        self._optional_depth = 0

    def _add(self, module: str | None) -> None:
        if self._optional_depth or not module:
            return
        root = module.split(".", 1)[0]
        if root and root != "__future__":
            self.required.add(root)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            self._add(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        if node.level == 0:
            self._add(node.module)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        dynamic_import = (
            isinstance(node.func, ast.Name)
            and node.func.id == "__import__"
        ) or (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "importlib"
            and node.func.attr == "import_module"
        )
        if dynamic_import and node.args and isinstance(node.args[0], ast.Constant):
            if isinstance(node.args[0].value, str):
                self._add(node.args[0].value)
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:  # noqa: N802
        if not _catches_import_error(node):
            self.generic_visit(node)
            return
        self._optional_depth += 1
        for child in node.body:
            self.visit(child)
        self._optional_depth -= 1
        for group in (node.handlers, node.orelse, node.finalbody):
            for child in group:
                self.visit(child)

    def visit_If(self, node: ast.If) -> None:  # noqa: N802
        is_type_checking = isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"
        if not is_type_checking:
            self.generic_visit(node)
            return
        self._optional_depth += 1
        for child in node.body:
            self.visit(child)
        self._optional_depth -= 1
        for child in node.orelse:
            self.visit(child)


def required_import_roots(code: str) -> tuple[str, ...]:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return ()
    visitor = _RequiredImportVisitor()
    visitor.visit(tree)
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    return tuple(sorted(visitor.required.difference(stdlib)))


def _missing_import_roots(
    modules: Iterable[str],
    *,
    python_executable: str,
    timeout_seconds: float = 10.0,
) -> tuple[str, ...]:
    requested = tuple(sorted(set(modules)))
    if not requested:
        return ()
    checker = (
        "import importlib.util,json,sys; "
        "names=json.loads(sys.argv[1]); missing=[]; "
        "exec('for name in names:\\n"
        " try:\\n  found=importlib.util.find_spec(name) is not None\\n"
        " except Exception:\\n  found=False\\n"
        " if not found:\\n  missing.append(name)'); "
        "print(json.dumps(missing))"
    )
    completed = subprocess.run(
        [python_executable, "-c", checker, json.dumps(requested)],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "unknown checker failure").strip()
        raise RuntimeError(
            f"dependency checker failed under {python_executable}: {detail}"
        )
    try:
        return tuple(str(name) for name in json.loads(completed.stdout or "[]"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"dependency checker returned invalid output under {python_executable}"
        ) from exc


def validate_runtime_dependencies(
    code: str,
    *,
    python_executable: str,
) -> tuple[ReviewIssue, ...]:
    required = required_import_roots(code)
    try:
        missing = _missing_import_roots(
            required,
            python_executable=python_executable,
        )
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        return (
            ReviewIssue(
                source="dependency_preflight",
                severity="critical",
                category="dependency_preflight_unavailable",
                owner="integration",
                evidence=str(exc),
                repair_instruction=(
                    "Repair the configured execution interpreter and rerun "
                    "install_dependencies.sh before submitting jobs."
                ),
            ),
        )
    if not missing:
        return ()
    distributions = sorted(
        {IMPORT_TO_DISTRIBUTION.get(module, module) for module in missing}
    )
    return (
        ReviewIssue(
            source="dependency_preflight",
            severity="critical",
            category="missing_dependency",
            owner="integration",
            evidence=(
                f"Execution interpreter {python_executable} cannot resolve required import(s): "
                f"{', '.join(missing)}."
            ),
            repair_instruction=(
                "Install the declared runtime environment with install_dependencies.sh, "
                f"or guard/remove the optional dependency. Required distribution(s): {', '.join(distributions)}."
            ),
        ),
    )


def merge_dependency_review_issues(
    decision: ReviewDecision | None,
    issues: tuple[ReviewIssue, ...],
) -> ReviewDecision | None:
    if not issues:
        return decision
    existing = list(decision.issues if decision is not None else ())
    identities = {(item.source, item.category, item.evidence) for item in existing}
    for issue in issues:
        identity = (issue.source, issue.category, issue.evidence)
        if identity not in identities:
            existing.append(issue)
            identities.add(identity)
    prefix = (decision.reasoning + " ") if decision is not None else ""
    return ReviewDecision(
        approved=False,
        reasoning=(
            prefix + "Deterministic dependency validation found an unavailable required import."
        ).strip(),
        issues=tuple(existing),
    )

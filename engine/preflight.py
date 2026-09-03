"""CPU model-preflight admission gate for MLEvolve candidates."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from agents.review_contracts import ReviewIssue
from engine.script_introspection import introspect_training_script

logger = logging.getLogger("MLEvolve")

REPO_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_PROFILE_DIR = REPO_ROOT / "config" / "preflight_profiles"
ALL_STAGES = {
    "static_source",
    "hardware",
    "construction",
    "data_contract",
    "abstract_forward",
    "cpu_training",
    "validation",
    "memory",
}
FALLBACK_STAGES = {"static_source", "hardware"}
ADAPTER_METHODS = {
    "build_model",
    "build_optimizer",
    "build_train_batch",
    "build_validation_batch",
    "training_step",
    "validation_step",
}


@dataclass(frozen=True)
class ProfileSelection:
    """Resolved checker profile and whether GPU-dependent CPU checks are valid."""

    manifest_profile: str
    detected_gpu: str | None
    hardware_checks_enabled: bool = True
    warning: str | None = None


@dataclass(frozen=True)
class AdapterInspection:
    """Static adapter/import-safety facts used before candidate import."""

    entrypoint_present: bool
    complete: bool
    main_guard_present: bool
    missing_methods: tuple[str, ...] = ()
    unsafe_top_level_lines: tuple[int, ...] = ()
    syntax_error: str | None = None


@dataclass
class PreflightOutcome:
    """Compact, integration-owned result used by search and scheduling."""

    status: str
    mode: str
    code_hash: str
    admitted: bool
    gpu_check_required: bool
    diagnostic_codes: list[str] = field(default_factory=list)
    report_path: str | None = None
    summary_path: str | None = None
    issues: list[ReviewIssue] = field(default_factory=list)
    profile: str | None = None
    warning: str | None = None
    internal_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "mode": self.mode,
            "code_hash": self.code_hash,
            "admitted": self.admitted,
            "gpu_check_required": self.gpu_check_required,
            "diagnostic_codes": list(self.diagnostic_codes),
            "report_path": self.report_path,
            "summary_path": self.summary_path,
            "profile": self.profile,
            "warning": self.warning,
            "internal_error": self.internal_error,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def candidate_code_hash(code: str) -> str:
    """Return the stable hash tied to an admission decision."""

    return "sha256:" + hashlib.sha256((code or "").encode("utf-8")).hexdigest()


def is_fresh_preflight(node: Any) -> bool:
    """Check that the current candidate source is exactly the admitted source."""

    return bool(getattr(node, "preflight_code_hash", None)) and (
        getattr(node, "preflight_code_hash")
        == candidate_code_hash(getattr(node, "code", ""))
    )


def inspect_adapter(code: str) -> AdapterInspection:
    """Find the explicit no-argument CandidateAdapter and main guard without importing."""

    try:
        tree = ast.parse(code or "")
    except SyntaxError as exc:
        return AdapterInspection(
            entrypoint_present=False,
            complete=False,
            main_guard_present=False,
            syntax_error=f"{exc.msg} at line {exc.lineno}",
        )

    candidate: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef | None = None
    main_guard = False
    unsafe_lines: list[int] = []
    for node in tree.body:
        if (
            isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "CandidateAdapter"
        ):
            candidate = node
        if isinstance(node, ast.If) and _is_main_guard(node.test):
            main_guard = True
            continue
        if _is_unsafe_top_level(node):
            unsafe_lines.append(int(getattr(node, "lineno", 0) or 0))

    if candidate is None:
        return AdapterInspection(
            False,
            False,
            main_guard,
            tuple(sorted(ADAPTER_METHODS)),
            tuple(unsafe_lines),
        )
    if not isinstance(candidate, ast.ClassDef):
        return AdapterInspection(
            True, True, main_guard, unsafe_top_level_lines=tuple(unsafe_lines)
        )

    methods = {
        child.name
        for child in candidate.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = tuple(sorted(ADAPTER_METHODS - methods))
    return AdapterInspection(
        True,
        not missing,
        main_guard,
        missing,
        tuple(unsafe_lines),
    )


def _pandas_row_values_tensor_lines(code: str) -> tuple[int, ...]:
    """Find tensor construction from a pandas row's untyped ``.values`` view."""

    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return ()
    lines: list[int] = []
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not call.args:
            continue
        if _call_name(call) not in {"torch.tensor", "torch.as_tensor"}:
            continue
        value = call.args[0]
        if not (
            isinstance(value, ast.Attribute)
            and value.attr == "values"
            and isinstance(value.value, ast.Subscript)
            and isinstance(value.value.value, ast.Name)
            and value.value.value.id == "row"
        ):
            continue
        lines.append(int(getattr(call, "lineno", 0) or 0))
    return tuple(sorted(set(line for line in lines if line > 0)))


def _is_main_guard(test: ast.AST) -> bool:
    if (
        not isinstance(test, ast.Compare)
        or len(test.ops) != 1
        or len(test.comparators) != 1
    ):
        return False
    if not isinstance(test.ops[0], ast.Eq):
        return False
    left, right = test.left, test.comparators[0]
    values = (left, right)
    return any(
        isinstance(value, ast.Name) and value.id == "__name__" for value in values
    ) and any(
        isinstance(value, ast.Constant) and value.value == "__main__"
        for value in values
    )


_SAFE_TOP_LEVEL_CONFIGURATION_CALLS = {
    "Path",
    "len",
    "os.environ.get",
    "os.getenv",
    "os.path.join",
    "pathlib.Path",
    "torch.cuda.is_available",
    "torch.device",
}


def _call_name(node: ast.Call) -> str | None:
    """Return the dotted name for a direct call without executing it."""

    parts: list[str] = []
    value: ast.AST = node.func
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if not isinstance(value, ast.Name):
        return None
    parts.append(value.id)
    return ".".join(reversed(parts))


def _is_lightweight_configuration_value(value: ast.AST) -> bool:
    """Allow pure configuration reads/constructors in top-level assignments."""

    if any(
        isinstance(child, (ast.Await, ast.Yield, ast.YieldFrom))
        for child in ast.walk(value)
    ):
        return False
    calls = [child for child in ast.walk(value) if isinstance(child, ast.Call)]
    return all(
        _call_name(call) in _SAFE_TOP_LEVEL_CONFIGURATION_CALLS for call in calls
    )


def _is_unsafe_top_level(node: ast.stmt) -> bool:
    """Identify import-time work while allowing definitions and constants."""

    if isinstance(
        node,
        (
            ast.Import,
            ast.ImportFrom,
            ast.ClassDef,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.Pass,
        ),
    ):
        return False
    if isinstance(node, ast.Expr):
        return not isinstance(node.value, ast.Constant)
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        value = node.value
        return value is not None and not _is_lightweight_configuration_value(value)
    return isinstance(
        node,
        (
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.With,
            ast.AsyncWith,
            ast.Match,
        ),
    ) or (isinstance(node, ast.If) and not _is_main_guard(node.test))


def normalize_preflight_precision(metadata: Mapping[str, Any] | str | None) -> str:
    """Translate scheduler/script precision vocabulary to the checker schema."""

    if isinstance(metadata, Mapping):
        raw = metadata.get("precision_mode")
        uses_amp = bool(metadata.get("uses_amp", False))
    else:
        raw = metadata
        uses_amp = False
    normalized = str(raw or "").strip().lower().replace("-", "_")
    if normalized in {"bf16", "bfloat16"}:
        return "bf16"
    if normalized in {"fp16", "float16", "half"}:
        return "fp16"
    if normalized == "tf32":
        return "tf32"
    if normalized in {"fp8", "fp8_te", "fp8_e5m2_pure", "mxfp8_te"}:
        return "fp8"
    if normalized in {"nvfp4", "nvfp4_te", "generic_fp4", "mxfp4"}:
        return "nvfp4"
    if normalized in {"fp32", "float32"}:
        return "fp32"
    return "fp16" if uses_amp else "fp32"


def derive_batch_scenarios(code: str, cfg: Any) -> list[int]:
    """Mirror the scheduler's power-of-two proposal offsets for preflight."""

    metadata = introspect_training_script(code or "")
    try:
        proposed = max(1, int(metadata.get("proposed_batch_size", 1)))
    except (TypeError, ValueError):
        proposed = 1
    base = 1 << int(math.log2(proposed))
    offsets: Iterable[int] = (-2, -1, 0, 1, 2)
    cap: int | None = None
    try:
        gpu_cfg = cfg.scheduler.settings.gpu_scheduler
        offsets = tuple(int(value) for value in gpu_cfg.batch_options.exponent_offsets)
        cap_raw = gpu_cfg.submission_defaults.batch_probe_max_batch_size
        cap = int(cap_raw) if cap_raw is not None else None
    except (AttributeError, TypeError, ValueError):
        pass
    values = sorted(
        {
            max(1, base * (2**offset)) if offset >= 0 else max(1, base // (2**-offset))
            for offset in offsets
        }
    )
    if cap is not None:
        values = [value for value in values if value <= cap]
    return values or [min(base, cap) if cap is not None else base]


def select_target_profile(
    target_profile: str,
    *,
    detected_hardware: Any | None = None,
) -> ProfileSelection:
    """Resolve an explicit target or map the current GPU to a supported profile."""

    requested = str(target_profile or "auto").strip()
    if requested.lower() != "auto":
        supplied = Path(requested).expanduser()
        if supplied.suffix.lower() in {".yaml", ".yml"}:
            resolved = supplied if supplied.is_absolute() else (REPO_ROOT / supplied)
            requested = str(resolved.resolve())
        return ProfileSelection(requested, None)

    hardware = detected_hardware
    if hardware is None:
        try:
            from localml_scheduler.hardware import detect_hardware_profile

            hardware = detect_hardware_profile()
        except Exception as exc:
            warning = f"Could not detect the target GPU; hardware and memory checks were skipped: {exc}"
            return ProfileSelection("nvidia/a100_40gb", None, False, warning)

    name = str(getattr(hardware, "gpu_name", "") or "")
    vram_mb = getattr(hardware, "total_vram_mb", None)
    lowered = name.lower()
    if "rtx 5090" in lowered:
        return ProfileSelection(
            str((CUSTOM_PROFILE_DIR / "rtx_5090_32gb.yaml").resolve()), name
        )
    if "a100" in lowered:
        if vram_mb is not None and int(vram_mb) > 61_440:
            return ProfileSelection(
                str((CUSTOM_PROFILE_DIR / "a100_80gb.yaml").resolve()), name
            )
        return ProfileSelection("nvidia/a100_40gb", name)
    if "a10" in lowered:
        return ProfileSelection("nvidia/a10_24gb", name)
    if "v100" in lowered:
        profile = (
            "nvidia/v100_32gb"
            if vram_mb is not None and int(vram_mb) > 24_576
            else "nvidia/v100_16gb"
        )
        return ProfileSelection(profile, name)

    warning = (
        f"No model-preflight profile is available for {name or 'the detected target'}; "
        "hardware and memory checks were skipped."
    )
    return ProfileSelection("nvidia/a100_40gb", name or None, False, warning)


def diagnostic_owner(code: str, stage: str) -> str:
    """Deterministically map checker diagnostics to stage-repair ownership."""

    code = str(code or "").upper()
    stage = str(stage or "").lower()
    if code.startswith(("SRC", "NET")) or stage == "static_source":
        return "integration"
    if code.startswith(("GPU", "DEV")) or stage == "hardware":
        return "datatype_precision"
    if code.startswith(("CON", "DAT", "SHP", "OUT", "FIX")) or stage in {
        "construction",
        "data_contract",
        "abstract_forward",
    }:
        return "model_design"
    if code.startswith(("OPT", "LOS", "GRD", "NUM", "AUT", "VAL", "MEM")) or stage in {
        "cpu_training",
        "validation",
        "memory",
    }:
        return "training_evaluation"
    return "integration"


def _is_uncached_pretrained_dependency(diagnostic: Mapping[str, Any]) -> bool:
    """Return whether an offline preflight reproduced an unavailable weight dependency.

    A balanced policy may admit ordinary inconclusive hardware checks, but it must
    not admit code which the isolated run already proved needs a Hugging Face
    checkpoint absent from the worker cache.  Such a candidate would otherwise
    reach the GPU worker and attempt a network download during the real run.
    """

    exception_type = str(diagnostic.get("exception_type") or "")
    text = " ".join(
        str(diagnostic.get(field) or "")
        for field in ("message", "stack_trace")
    ).lower()
    if exception_type == "LocalEntryNotFoundError":
        return True
    return (
        "huggingface" in text
        and "cached files" in text
        and any(
            marker in text
            for marker in (
                "network is disabled",
                "outgoing traffic has been disabled",
                "offline mode",
                "couldn't connect",
                "cannot find the requested files in the disk cache",
            )
        )
    )


def preflight_diagnostics_require_rejection(
    diagnostics: Iterable[Mapping[str, Any]],
) -> bool:
    """Whether inconclusive diagnostics prove the candidate cannot run offline."""

    return any(_is_uncached_pretrained_dependency(item) for item in diagnostics)


def diagnostic_to_review_issue(diagnostic: Mapping[str, Any]) -> ReviewIssue | None:
    """Convert confirmed candidate failures into targeted repair input."""

    unavailable_pretrained_dependency = _is_uncached_pretrained_dependency(diagnostic)
    if (
        diagnostic.get("classification") != "confirmed_candidate_failure"
        and not unavailable_pretrained_dependency
    ):
        return None
    code = str(diagnostic.get("code") or "CHK001")
    stage = str(diagnostic.get("stage") or "unknown")
    message = str(
        diagnostic.get("message") or "Model preflight reproduced a candidate defect."
    )
    exception_type = str(diagnostic.get("exception_type") or "")
    stack_trace = str(diagnostic.get("stack_trace") or "")
    targeted_guidance = ""
    if stage == "construction" and exception_type == "KeyError":
        targeted_guidance = (
            " Treat checker-supplied context as a partial mapping: merge it over "
            "CandidateAdapter defaults before reading optional keys, while preserving "
            "caller-provided values."
        )
    elif unavailable_pretrained_dependency:
        targeted_guidance = (
            " Keep the same real model family and make isolated construction network-free, "
            "for example by retrying the same model with pretrained=False when cached weights "
            "are unavailable; do not substitute a mock architecture."
        )
    elif (
        exception_type == "ValueError"
        and "array is read-only" in message.lower()
        and "shuffle" in stack_trace.lower()
    ):
        targeted_guidance = (
            " The failing rng.shuffle(ids) mutates ids in place, so copy the exact array being "
            "shuffled into writable storage at its source, for example "
            "ids = series.astype(str).to_numpy(copy=True). A DataFrame copy alone does not "
            "guarantee that its exported NumPy view is writable."
        )
    elif (
        exception_type == "TypeError"
        and "'nonetype' object is not callable" in message.lower()
        and "criterion" in stack_trace.lower()
    ):
        targeted_guidance = (
            " CandidateAdapter context mutations do not persist across checker method calls. "
            "Make training_step and validation_step resolve a real criterion when the caller's "
            "partial context omits it or supplies None, for example context.get('criterion') "
            "or the script's real loss constructor; do not rely on build_model mutating a local context."
        )
    location = ""
    if diagnostic.get("file"):
        location = f" ({diagnostic['file']}:{diagnostic.get('line') or '?'})"
    return ReviewIssue(
        source="model_preflight",
        severity="critical",
        category=f"preflight_{code.lower()}",
        owner=diagnostic_owner(code, stage),
        evidence=f"[{code}] {message}{location}",
        repair_instruction=(
            f"Repair the confirmed {stage} defect [{code}] and preserve the CandidateAdapter contract; "
            "do not suppress the check or replace real training behavior with mocks."
            f"{targeted_guidance}"
        ),
    )


def admission_for_status(status: str, policy_mode: str, fail_open: bool) -> bool:
    """Apply the integration admission policy, including infrastructure fail-open."""

    status = str(status).upper()
    policy = str(policy_mode or "balanced").lower()
    if status == "INTERNAL_ERROR":
        return bool(fail_open)
    if status == "FAIL":
        return False
    if policy == "audit":
        return True
    if status == "PASS":
        return True
    if status == "INCONCLUSIVE":
        return policy == "balanced"
    return False


def preflight_enabled(cfg: Any) -> bool:
    settings = getattr(cfg, "preflight", None)
    if settings is None or not bool(getattr(settings, "enabled", True)):
        return False
    mode = (
        str(getattr(getattr(cfg, "experiment", None), "mode", "hardware_aware") or "")
        .lower()
        .replace("-", "_")
    )
    enabled_modes = {
        str(value).lower().replace("-", "_")
        for value in (getattr(settings, "enabled_modes", None) or ["hardware_aware"])
    }
    return mode in enabled_modes


class ModelPreflightGate:
    """Materialize and check one candidate without importing it in the controller."""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.settings = cfg.preflight

    def run(self, node: Any, *, generated: bool, attempt: int = 0) -> PreflightOutcome:
        code = str(getattr(node, "code", "") or "")
        code_hash = candidate_code_hash(code)
        inspection = inspect_adapter(code)
        profile = select_target_profile(
            str(getattr(self.settings, "target_profile", "auto"))
        )
        full_mode = inspection.entrypoint_present
        mode = "full_cpu" if full_mode else "static_hardware_fallback"
        stages = set(ALL_STAGES if full_mode else FALLBACK_STAGES)
        if not profile.hardware_checks_enabled:
            stages.discard("hardware")
            stages.discard("memory")

        node_dir = self._node_dir(node)
        candidate_dir = node_dir / "candidate"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = candidate_dir / "candidate.py"
        candidate_path.write_text(code, encoding="utf-8")
        manifest_path = node_dir / "preflight.yaml"
        manifest = self._manifest(node, candidate_dir, profile.manifest_profile, code)
        manifest_path.write_text(
            yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
        )

        contract_issues = self._contract_issues(inspection, code) if generated else []
        report_path = node_dir / "report.json"
        try:
            from model_preflight import check
            from model_preflight.reporting.json_report import write_json

            # Worker environments inherit this non-secret path before replacing
            # HOME/TMPDIR, allowing adapters to reuse real input transforms.
            os.environ["MLEVOLVE_INPUT_DIR"] = str(
                (Path(self.cfg.workspace_dir) / "input").resolve()
            )
            report = check(
                manifest_path,
                only=stages,
                use_cache=bool(getattr(self.settings, "cache", False)),
            )
            write_json(report, report_path)
            attempt_path = node_dir / f"report_attempt_{attempt}.json"
            shutil.copyfile(report_path, attempt_path)
            report_dict = report.to_dict()
            diagnostics = list(report_dict.get("diagnostics", []))
            issues = [
                issue
                for item in diagnostics
                if (issue := diagnostic_to_review_issue(item)) is not None
            ]
            issues.extend(contract_issues)
            diagnostic_codes = [str(item.get("code")) for item in diagnostics]
            diagnostic_codes.extend(self._contract_codes(contract_issues))
            status = str(report_dict["overall_status"])
            gpu_check_required = bool(report_dict.get("gpu_check_required", False))
            if contract_issues or preflight_diagnostics_require_rejection(diagnostics):
                status = "FAIL"
            elif profile.warning and status == "PASS":
                status = "INCONCLUSIVE"
            gpu_check_required = gpu_check_required or bool(profile.warning)
            admitted = admission_for_status(
                status,
                str(getattr(self.settings, "policy_mode", "balanced")),
                bool(getattr(self.settings, "fail_open_on_internal_error", True)),
            )
            outcome = PreflightOutcome(
                status=status,
                mode=mode,
                code_hash=code_hash,
                admitted=admitted,
                gpu_check_required=gpu_check_required,
                diagnostic_codes=sorted(set(diagnostic_codes)),
                report_path=str(report_path.resolve()),
                issues=issues,
                profile=(
                    profile.manifest_profile
                    if profile.hardware_checks_enabled
                    else None
                ),
                warning=profile.warning,
            )
        except Exception as exc:
            logger.exception(
                "Model preflight infrastructure failed for node %s",
                getattr(node, "id", "unknown"),
            )
            status = "FAIL" if contract_issues else "INTERNAL_ERROR"
            admitted = (
                False
                if contract_issues
                else admission_for_status(
                    status,
                    str(getattr(self.settings, "policy_mode", "balanced")),
                    bool(getattr(self.settings, "fail_open_on_internal_error", True)),
                )
            )
            outcome = PreflightOutcome(
                status=status,
                mode=mode,
                code_hash=code_hash,
                admitted=admitted,
                gpu_check_required=True,
                diagnostic_codes=[
                    "PREFLIGHT_INTERNAL_ERROR",
                    *self._contract_codes(contract_issues),
                ],
                report_path=(
                    str(report_path.resolve()) if report_path.exists() else None
                ),
                issues=contract_issues,
                profile=(
                    profile.manifest_profile
                    if profile.hardware_checks_enabled
                    else None
                ),
                warning=profile.warning,
                internal_error=f"{type(exc).__name__}: {exc}",
            )

        summary_path = node_dir / "admission_summary.json"
        outcome.summary_path = str(summary_path.resolve())
        summary_path.write_text(
            json.dumps(outcome.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return outcome

    def _node_dir(self, node: Any) -> Path:
        safe_id = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_"
            for char in str(node.id)
        )
        return Path(self.cfg.workspace_dir) / "working" / "preflight" / safe_id

    def _manifest(
        self, node: Any, candidate_dir: Path, profile: str, code: str
    ) -> dict[str, Any]:
        metadata = introspect_training_script(code)
        task_name = str(
            getattr(self.cfg, "exp_id", None)
            or getattr(self.cfg, "exp_name", "mlevolve")
        )
        return {
            "schema_version": 1,
            "candidate": {
                "id": str(node.id),
                "root": str(candidate_dir.resolve()),
                "adapter": "candidate:CandidateAdapter",
            },
            "task": {"name": task_name},
            "scenarios": {
                "train_batch_sizes": derive_batch_scenarios(code, self.cfg),
                "test_last_batch": True,
                "last_batch_size": 1,
                "run_validation": True,
                "precision": [normalize_preflight_precision(metadata)],
            },
            "target": {"profile": profile},
            "execution": {
                "abstract_timeout_seconds": float(
                    getattr(self.settings, "abstract_timeout_seconds", 30.0)
                ),
                "cpu_timeout_seconds": float(
                    getattr(self.settings, "cpu_timeout_seconds", 90.0)
                ),
                "maximum_cpu_memory_mb": int(
                    getattr(self.settings, "maximum_cpu_memory_mb", 8192)
                ),
                "maximum_processes": int(
                    getattr(self.settings, "maximum_processes", 32)
                ),
                "maximum_output_bytes": int(
                    getattr(self.settings, "maximum_output_bytes", 1_000_000)
                ),
                "disable_network": bool(
                    getattr(self.settings, "disable_network", True)
                ),
                "allow_real_cpu_abstract_fallback": bool(
                    getattr(self.settings, "allow_real_cpu_abstract_fallback", False)
                ),
                "cache": bool(getattr(self.settings, "cache", False)),
            },
            "policy": {"mode": str(getattr(self.settings, "policy_mode", "balanced"))},
        }

    def _contract_issues(
        self, inspection: AdapterInspection, code: str = ""
    ) -> list[ReviewIssue]:
        if not bool(getattr(self.settings, "require_adapter_for_generated", True)):
            return []
        issues: list[ReviewIssue] = []
        if not inspection.entrypoint_present or not inspection.complete:
            detail = "CandidateAdapter is missing"
            if inspection.missing_methods and inspection.entrypoint_present:
                detail = "CandidateAdapter is missing: " + ", ".join(
                    inspection.missing_methods
                )
            issues.append(
                ReviewIssue(
                    source="model_preflight",
                    severity="critical",
                    category="preflight_adapter_contract",
                    owner="integration",
                    evidence=detail,
                    repair_instruction=(
                        "Add a no-argument CandidateAdapter implementing build_model, build_optimizer, "
                        "build_train_batch, build_validation_batch, training_step, and validation_step."
                    ),
                )
            )
        if not inspection.main_guard_present or inspection.unsafe_top_level_lines:
            evidence = (
                "Training execution is not protected by if __name__ == '__main__'."
            )
            repair_instruction = (
                "Move all training and submission execution behind an "
                "if __name__ == '__main__' guard."
            )
            if inspection.main_guard_present:
                lines = ", ".join(
                    str(line) for line in inspection.unsafe_top_level_lines
                )
                evidence = (
                    "Executable import-time statements remain outside the main guard "
                    f"at line(s): {lines}."
                )
                repair_instruction = (
                    "The main guard already exists. Move each listed side-effecting call "
                    "into that existing main guard while keeping lightweight constants and "
                    "definitions importable. Calling the wrapper at module scope remains unsafe, "
                    "so merely wrapping the same operation in a helper does not repair it. Do not "
                    "add, remove, or duplicate the existing main guard."
                )
                source_lines = (code or "").splitlines()
                unsafe_source = [
                    source_lines[line - 1].strip()
                    for line in inspection.unsafe_top_level_lines
                    if 0 < line <= len(source_lines)
                ]
                if any(
                    line.replace(" ", "").startswith("DEVICE=_resolve_device")
                    for line in unsafe_source
                ):
                    repair_instruction = (
                        "The main guard already exists. Replace the top-level device helper call "
                        "with the import-safe direct configuration "
                        'DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu"). '
                        "Do not add another main guard or move/remove the existing main() call."
                    )
            issues.append(
                ReviewIssue(
                    source="model_preflight",
                    severity="critical",
                    category="preflight_import_safety",
                    owner="integration",
                    evidence=evidence,
                    repair_instruction=repair_instruction,
                )
            )
        pandas_row_lines = _pandas_row_values_tensor_lines(code)
        if pandas_row_lines:
            rendered_lines = ", ".join(str(line) for line in pandas_row_lines)
            issues.append(
                ReviewIssue(
                    source="model_preflight",
                    severity="critical",
                    category="preflight_pandas_row_tensor",
                    owner="model_design",
                    evidence=(
                        "Pandas row slice `.values` is passed directly to a torch tensor "
                        f"at line(s) {rendered_lines}; it can retain object dtype when the "
                        "source row also contains identifiers."
                    ),
                    repair_instruction=(
                        "Convert the selected pandas row features to an explicit numeric array "
                        "before tensor construction, for example "
                        "row[METADATA_COLS].to_numpy(dtype=np.float32), and preserve the "
                        "existing feature list and model input shape."
                    ),
                )
            )
        return issues

    @staticmethod
    def _contract_codes(issues: Iterable[ReviewIssue]) -> list[str]:
        mapping = {
            "preflight_adapter_contract": "MLE_ADAPTER001",
            "preflight_import_safety": "MLE_IMPORT001",
        }
        return [mapping.get(issue.category, "MLE_CONTRACT001") for issue in issues]


def apply_outcome_to_node(
    node: Any, outcome: PreflightOutcome, *, repair_count: int
) -> None:
    """Copy the compact admission record into the serializable SearchNode journal."""

    node.preflight_status = outcome.status
    node.preflight_mode = outcome.mode
    node.preflight_code_hash = outcome.code_hash
    node.preflight_diagnostic_codes = list(outcome.diagnostic_codes)
    node.preflight_report_path = outcome.report_path
    node.preflight_summary_path = outcome.summary_path
    node.preflight_repair_count = int(repair_count)
    node.preflight_gpu_check_required = bool(outcome.gpu_check_required)
    node.preflight_admitted = bool(outcome.admitted)


def node_preflight_metadata(node: Any | None) -> dict[str, Any]:
    """Return the bounded metadata shared by events and scheduler jobs."""

    if node is None:
        return {}
    return {
        "preflight_status": getattr(node, "preflight_status", None),
        "preflight_mode": getattr(node, "preflight_mode", None),
        "preflight_code_hash": getattr(node, "preflight_code_hash", None),
        "preflight_diagnostic_codes": list(
            getattr(node, "preflight_diagnostic_codes", None) or []
        ),
        "preflight_report_path": getattr(node, "preflight_report_path", None),
        "preflight_summary_path": getattr(node, "preflight_summary_path", None),
        "preflight_repair_count": int(getattr(node, "preflight_repair_count", 0) or 0),
        "gpu_check_required": bool(
            getattr(node, "preflight_gpu_check_required", False)
        ),
        "preflight_admitted": getattr(node, "preflight_admitted", None),
    }

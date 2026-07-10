"""Best-effort timing instrumentation for generated candidate scripts."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PHASES = ("training", "inference", "validation")


@dataclass(slots=True)
class PhaseInstrumentationResult:
    code: str
    instrumented: bool
    phase_log_path: str
    instrumented_region_count: int = 0
    reason: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "phase_log_path": self.phase_log_path,
            "phase_instrumented": self.instrumented,
            "phase_instrumented_region_count": self.instrumented_region_count,
            "phase_instrumentation_reason": self.reason,
        }


def instrument_code_for_phase_timing(code: str, phase_log_path: str | Path) -> PhaseInstrumentationResult:
    """Inject phase timing into common train/inference/validation regions."""
    phase_log = str(phase_log_path)
    try:
        module = ast.parse(code)
    except SyntaxError as exc:
        return PhaseInstrumentationResult(
            code=code,
            instrumented=False,
            phase_log_path=phase_log,
            reason=f"syntax_error:{exc.__class__.__name__}",
        )

    transformer = _PhaseTimingTransformer()
    module = transformer.visit(module)
    ast.fix_missing_locations(module)
    if transformer.instrumented_region_count <= 0:
        return PhaseInstrumentationResult(
            code=code,
            instrumented=False,
            phase_log_path=phase_log,
            reason="no_recognized_phase_regions",
        )

    helper = ast.parse(_helper_source(phase_log))
    module.body = helper.body + module.body
    ast.fix_missing_locations(module)
    return PhaseInstrumentationResult(
        code=ast.unparse(module),
        instrumented=True,
        phase_log_path=phase_log,
        instrumented_region_count=transformer.instrumented_region_count,
        reason="ok",
    )


def materialize_phase_instrumented_file(
    source_path: str | Path,
    output_path: str | Path,
    phase_log_path: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Return the executable path and metadata for an optionally instrumented file."""
    source = Path(source_path)
    result = instrument_code_for_phase_timing(source.read_text(encoding="utf-8"), phase_log_path)
    if not result.instrumented:
        return source, result.to_metadata()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.code, encoding="utf-8")
    return output, result.to_metadata()


def parse_phase_timing_log(
    phase_log_path: str | Path,
    *,
    exec_time: float | None = None,
    instrumentation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse phase events and return unioned duration totals."""
    path = Path(phase_log_path)
    metadata = dict(instrumentation or {})
    if not path.exists():
        return {
            "phase_durations_seconds": {phase: None for phase in (*PHASES, "other_candidate")},
            "phase_interval_count": 0,
            "phase_timing_event_count": 0,
            "phase_timing_coverage_seconds": 0.0,
            "phase_timing_coverage_ratio": None,
            "phase_timing_available": False,
            "phase_instrumented": bool(metadata.get("phase_instrumented")),
            "phase_instrumentation_reason": metadata.get("phase_instrumentation_reason") or "phase_log_missing",
        }

    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("phase") in PHASES and event.get("event") in {"start", "end"}:
            try:
                event["time"] = float(event["time"])
            except (TypeError, ValueError):
                continue
            events.append(event)

    intervals_by_phase: dict[str, list[tuple[float, float]]] = {phase: [] for phase in PHASES}
    stacks: dict[str, list[float]] = {phase: [] for phase in PHASES}
    for event in sorted(events, key=lambda item: item["time"]):
        phase = str(event["phase"])
        if event["event"] == "start":
            stacks[phase].append(float(event["time"]))
            continue
        if stacks[phase]:
            started = stacks[phase].pop()
            ended = float(event["time"])
            if ended >= started:
                intervals_by_phase[phase].append((started, ended))

    durations = {phase: _union_duration(intervals) for phase, intervals in intervals_by_phase.items()}
    all_intervals = [interval for intervals in intervals_by_phase.values() for interval in intervals]
    covered_seconds = _union_duration(all_intervals)
    if exec_time is not None:
        other = max(0.0, float(exec_time) - covered_seconds)
        coverage_ratio = min(1.0, covered_seconds / float(exec_time)) if exec_time > 0 else None
    else:
        other = None
        coverage_ratio = None
    durations["other_candidate"] = other
    return {
        "phase_durations_seconds": durations,
        "phase_interval_count": sum(len(intervals) for intervals in intervals_by_phase.values()),
        "phase_timing_event_count": len(events),
        "phase_timing_coverage_seconds": covered_seconds,
        "phase_timing_coverage_ratio": coverage_ratio,
        "phase_timing_available": bool(events),
        "phase_instrumented": bool(metadata.get("phase_instrumented")),
        "phase_instrumentation_reason": metadata.get("phase_instrumentation_reason"),
    }


def _union_duration(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    start, end = ordered[0]
    for cur_start, cur_end in ordered[1:]:
        if cur_start <= end:
            end = max(end, cur_end)
        else:
            total += max(0.0, end - start)
            start, end = cur_start, cur_end
    total += max(0.0, end - start)
    return total


def _helper_source(phase_log_path: str) -> str:
    return f"""
import json as _mlevolve_phase_json
import os as _mlevolve_phase_os
import time as _mlevolve_phase_time
_MLEVOLVE_PHASE_LOG_PATH = {phase_log_path!r}
def _mlevolve_phase_event(phase, event):
    try:
        _mlevolve_phase_os.makedirs(_mlevolve_phase_os.path.dirname(_MLEVOLVE_PHASE_LOG_PATH), exist_ok=True)
        with open(_MLEVOLVE_PHASE_LOG_PATH, "a", encoding="utf-8") as _mlevolve_phase_handle:
            _mlevolve_phase_handle.write(_mlevolve_phase_json.dumps({{"phase": phase, "event": event, "time": _mlevolve_phase_time.time()}}) + "\\n")
    except Exception:
        pass
class _mlevolve_phase:
    def __init__(self, phase):
        self.phase = phase
    def __enter__(self):
        _mlevolve_phase_event(self.phase, "start")
        return self
    def __exit__(self, exc_type, exc, tb):
        _mlevolve_phase_event(self.phase, "end")
        return False
"""


class _PhaseTimingTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.instrumented_region_count = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node = self.generic_visit(node)
        phase = _classify_name(node.name)
        if phase is None or not node.body:
            return node
        docstring = []
        body = list(node.body)
        if body and _is_docstring_expr(body[0]):
            docstring = [body.pop(0)]
        if body:
            node.body = docstring + [_phase_with(phase, body, node)]
            self.instrumented_region_count += 1
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node = self.generic_visit(node)
        phase = _classify_name(node.name)
        if phase is None or not node.body:
            return node
        docstring = []
        body = list(node.body)
        if body and _is_docstring_expr(body[0]):
            docstring = [body.pop(0)]
        if body:
            node.body = docstring + [_phase_with(phase, body, node)]
            self.instrumented_region_count += 1
        return node

    def visit_For(self, node: ast.For) -> ast.AST:
        node = self.generic_visit(node)
        phase = _classify_loop(node)
        if phase is None:
            return node
        self.instrumented_region_count += 1
        return _phase_with(phase, [node], node)


def _phase_with(phase: str, body: list[ast.stmt], source_node: ast.AST) -> ast.With:
    with_node = ast.With(
        items=[
            ast.withitem(
                context_expr=ast.Call(func=ast.Name(id="_mlevolve_phase", ctx=ast.Load()), args=[ast.Constant(phase)], keywords=[]),
                optional_vars=None,
            )
        ],
        body=body,
    )
    return ast.copy_location(with_node, source_node)


def _is_docstring_expr(node: ast.stmt) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def _classify_name(name: str) -> str | None:
    lowered = name.lower()
    if any(token in lowered for token in ("train", "fit", "finetune")):
        return "training"
    if any(token in lowered for token in ("submission", "predict", "inference", "infer")):
        return "inference"
    if any(token in lowered for token in ("valid", "eval", "score")):
        return "validation"
    return None


def _classify_loop(node: ast.For) -> str | None:
    target_name = _name_text(node.target)
    iter_name = _unparse_lower(node.iter)
    if target_name in {"epoch", "ep", "epoch_idx"}:
        return "training"
    if any(token in iter_name for token in ("train_loader", "train_dl", "training_loader")):
        return "training"
    if any(token in iter_name for token in ("val_loader", "valid_loader", "validation_loader")):
        return "validation"
    if any(token in iter_name for token in ("test_loader", "submission_loader", "predict_loader", "inference_loader")):
        return "inference"
    return None


def _name_text(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id.lower()
    return ""


def _unparse_lower(node: ast.AST) -> str:
    try:
        return ast.unparse(node).lower()
    except Exception:
        return ""

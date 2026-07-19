"""Lightweight introspection for generated MLEvolve training scripts."""

from __future__ import annotations

import ast
import re
import logging
from dataclasses import asdict, dataclass, field
from hashlib import sha1
from typing import Any

logger = logging.getLogger("MLEvolve")


BATCH_PARAM_NAMES = (
    "BS",
    "BATCH_SIZE",
    "batch_size",
    "train_batch_size",
    "eval_batch_size",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "BASE_BATCH_SIZE",
    "PHYSICAL_BATCH_SIZE",
    "TRAIN_BATCH_SIZE",
)

EPOCH_PARAM_NAMES = (
    "EPOCHS",
    "NUM_EPOCHS",
    "epochs",
    "num_epochs",
    "max_epochs",
)

RESOLUTION_PARAM_NAMES = (
    "IMG_SIZE",
    "IMAGE_SIZE",
    "INPUT_SIZE",
    "RESOLUTION",
    "img_size",
    "image_size",
    "input_size",
    "resize_size",
)

FOLD_PARAM_NAMES = (
    "N_FOLDS",
    "NUM_FOLDS",
    "K_FOLDS",
    "n_folds",
    "num_folds",
    "fold_count",
)

ENSEMBLE_PARAM_NAMES = (
    "ENSEMBLE_SIZE",
    "NUM_MODELS",
    "N_MODELS",
    "ensemble_count",
    "num_models",
)

TTA_PARAM_NAMES = (
    "TTA_STEPS",
    "TTA_COUNT",
    "N_TTA",
    "tta_steps",
    "tta_count",
    "num_tta",
)

MODEL_PARAM_NAMES = (
    "MODEL_NAME",
    "model_name",
    "BACKBONE",
    "backbone",
    "MODEL",
    "model_id",
    "checkpoint",
    "CHECKPOINT",
)

MODEL_FAMILY_PARAM_NAMES = (
    "MODEL_FAMILY",
    "model_family",
    "SCHEDULER_MODEL_FAMILY",
    "scheduler_model_family",
)

MODEL_BRANCH_PARAM_NAMES = (
    "MODEL_BRANCH",
    "model_branch",
    "BRANCH_NAME",
    "branch_name",
    "SCHEDULER_BRANCH_NAME",
    "scheduler_branch_name",
)

PRECISION_PARAM_NAMES = (
    "PRECISION",
    "PRECISION_MODE",
    "AMP_DTYPE",
    "DTYPE",
    "precision",
    "precision_mode",
    "amp_dtype",
    "dtype",
)

LEARNING_RATE_PARAM_NAMES = (
    "LR",
    "LEARNING_RATE",
    "learning_rate",
    "lr",
)

WEIGHT_DECAY_PARAM_NAMES = (
    "WEIGHT_DECAY",
    "weight_decay",
)

GRADIENT_ACCUMULATION_PARAM_NAMES = (
    "GRADIENT_ACCUMULATION_STEPS",
    "GRAD_ACCUM_STEPS",
    "ACCUMULATION_STEPS",
    "gradient_accumulation_steps",
    "grad_accum_steps",
    "accumulation_steps",
)

NUM_WORKERS_PARAM_NAMES = (
    "NUM_WORKERS",
    "DATALOADER_WORKERS",
    "num_workers",
    "dataloader_workers",
)

_FLOAT_LITERAL = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"

_BATCH_PARAM_PATTERN = re.compile(
    rf"\b(?:{'|'.join(re.escape(name) for name in BATCH_PARAM_NAMES)})\b\s*=\s*(\d+)"
)
_EPOCH_PARAM_PATTERN = re.compile(
    rf"\b(?:{'|'.join(re.escape(name) for name in EPOCH_PARAM_NAMES)})\b\s*=\s*(\d+)"
)
_RESOLUTION_TUPLE_PATTERN = re.compile(
    rf"\b(?:{'|'.join(re.escape(name) for name in RESOLUTION_PARAM_NAMES)})\b\s*=\s*\(?\s*(\d+)\s*,\s*(\d+)"
)
_RESOLUTION_SINGLE_PATTERN = re.compile(
    rf"\b(?:{'|'.join(re.escape(name) for name in RESOLUTION_PARAM_NAMES)})\b\s*=\s*(\d+)"
)
_FOLD_PARAM_PATTERN = re.compile(
    rf"\b(?:{'|'.join(re.escape(name) for name in FOLD_PARAM_NAMES)})\b\s*=\s*(\d+)"
)
_ENSEMBLE_PARAM_PATTERN = re.compile(
    rf"\b(?:{'|'.join(re.escape(name) for name in ENSEMBLE_PARAM_NAMES)})\b\s*=\s*(\d+)"
)
_TTA_PARAM_PATTERN = re.compile(
    rf"\b(?:{'|'.join(re.escape(name) for name in TTA_PARAM_NAMES)})\b\s*=\s*(\d+)"
)
_TTA_BOOL_PATTERN = re.compile(r"\b(?:USE_TTA|use_tta|tta)\b\s*=\s*True\b")
_MODEL_PARAM_PATTERN = re.compile(
    rf"\b(?:{'|'.join(re.escape(name) for name in MODEL_PARAM_NAMES)})\b\s*=\s*['\"]([^'\"]+)['\"]"
)
_MODEL_FAMILY_PARAM_PATTERN = re.compile(
    rf"\b(?:{'|'.join(re.escape(name) for name in MODEL_FAMILY_PARAM_NAMES)})\b\s*=\s*['\"]([^'\"]+)['\"]"
)
_MODEL_BRANCH_PARAM_PATTERN = re.compile(
    rf"\b(?:{'|'.join(re.escape(name) for name in MODEL_BRANCH_PARAM_NAMES)})\b\s*=\s*['\"]([^'\"]+)['\"]"
)
_PRECISION_STRING_PATTERN = re.compile(
    rf"\b(?:{'|'.join(re.escape(name) for name in PRECISION_PARAM_NAMES)})\b\s*=\s*['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)
_PRECISION_TORCH_DTYPE_PATTERN = re.compile(
    rf"\b(?:{'|'.join(re.escape(name) for name in PRECISION_PARAM_NAMES)})\b\s*=\s*torch\.(bfloat16|float16|float32)",
    re.IGNORECASE,
)
_AUTOCAST_DTYPE_PATTERN = re.compile(
    r"autocast\([^\)]*dtype\s*=\s*torch\.(bfloat16|float16|float32)",
    re.IGNORECASE,
)
_LEARNING_RATE_PARAM_PATTERN = re.compile(
    rf"\b(?:{'|'.join(re.escape(name) for name in LEARNING_RATE_PARAM_NAMES)})\b\s*=\s*{_FLOAT_LITERAL}"
)
_WEIGHT_DECAY_PARAM_PATTERN = re.compile(
    rf"\b(?:{'|'.join(re.escape(name) for name in WEIGHT_DECAY_PARAM_NAMES)})\b\s*=\s*{_FLOAT_LITERAL}"
)
_GRADIENT_ACCUMULATION_PARAM_PATTERN = re.compile(
    rf"\b(?:{'|'.join(re.escape(name) for name in GRADIENT_ACCUMULATION_PARAM_NAMES)})\b\s*=\s*(\d+)"
)
_NUM_WORKERS_PARAM_PATTERN = re.compile(
    rf"\b(?:{'|'.join(re.escape(name) for name in NUM_WORKERS_PARAM_NAMES)})\b\s*=\s*(\d+)"
)
_BATCH_PROBE_NORMALIZE_PATTERNS = (
    rf"(\b(?:{'|'.join(re.escape(name) for name in BATCH_PARAM_NAMES)})\b\s*=\s*)([^,\n\)]*)",
)
_BATCH_PROBE_ENABLE_PATTERN = re.compile(
    rf"\b(?:{'|'.join(re.escape(name) for name in BATCH_PARAM_NAMES)})\b"
)

_NON_TRAIN_BATCH_TOKENS = ("eval", "valid", "val_", "test", "predict", "infer", "submission")
_TRAIN_ROLE_TOKENS = ("train", "training")
_NON_TRAIN_ROLE_TOKENS = ("valid", "validation", "val", "test", "predict", "inference", "submission")


@dataclass(frozen=True, slots=True)
class TrainingBatchSite:
    """A statically proven training-loader batch argument."""

    lineno: int
    col_offset: int
    argument: str
    expression: str


@dataclass(frozen=True, slots=True)
class TrainingBatchContract:
    """Static contract used to decide whether scheduler batch probing is safe."""

    initial_batch_size: int | None = None
    minimum_batch_size: int = 1
    batch_symbols: tuple[str, ...] = ()
    train_sites: tuple[TrainingBatchSite, ...] = ()
    confidence: str = "unsupported"
    unsupported_reason: str | None = None
    diagnostics: tuple[str, ...] = ()

    @property
    def supported(self) -> bool:
        return self.confidence == "high" and self.initial_batch_size is not None and bool(self.train_sites)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["supported"] = self.supported
        return payload


def _name_looks_like_train_batch(name: str) -> bool:
    lowered = str(name or "").lower()
    if "batch" not in lowered:
        return False
    return not any(token in lowered for token in _NON_TRAIN_BATCH_TOKENS)


def _assignment_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _assignment_value(node: ast.AST) -> ast.expr | None:
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        return node.value
    return None


def _resolve_static_int(expr: ast.expr | None, assignments: dict[str, ast.expr], seen: set[str] | None = None) -> int | None:
    if expr is None:
        return None
    if isinstance(expr, ast.Constant) and isinstance(expr.value, int) and not isinstance(expr.value, bool):
        return int(expr.value)
    if isinstance(expr, ast.Name):
        seen = set(seen or ())
        if expr.id in seen:
            return None
        seen.add(expr.id)
        return _resolve_static_int(assignments.get(expr.id), assignments, seen)
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.UAdd, ast.USub)):
        value = _resolve_static_int(expr.operand, assignments, seen)
        if value is not None:
            return value if isinstance(expr.op, ast.UAdd) else -value
    if isinstance(expr, ast.BinOp):
        left = _resolve_static_int(expr.left, assignments, seen)
        right = _resolve_static_int(expr.right, assignments, seen)
        if left is None or right is None:
            return None
        if isinstance(expr.op, ast.Mult):
            return left * right
        if isinstance(expr.op, ast.FloorDiv) and right:
            return left // right
        if isinstance(expr.op, ast.Add):
            return left + right
        if isinstance(expr.op, ast.Sub):
            return left - right
    if isinstance(expr, ast.IfExp):
        left = _resolve_static_int(expr.body, assignments, seen)
        right = _resolve_static_int(expr.orelse, assignments, seen)
        return left if left == right else (left if right is None else right if left is None else None)
    return None


def _call_name(call: ast.Call) -> str:
    try:
        return ast.unparse(call.func).lower()
    except Exception:
        return ""


def _expr_names(expr: ast.AST | None) -> set[str]:
    return {node.id for node in ast.walk(expr) if isinstance(node, ast.Name)} if expr is not None else set()


def _loader_role(call: ast.Call, parent_target: str | None) -> str | None:
    call_name = _call_name(call)
    if "loader" not in call_name and "dataloader" not in call_name:
        return None
    evidence = " ".join(
        [
            str(parent_target or ""),
            call_name,
            *(ast.unparse(arg) for arg in call.args[:1]),
        ]
    ).lower()
    if any(token in evidence for token in _NON_TRAIN_ROLE_TOKENS):
        return "non_train"
    if any(token in evidence for token in _TRAIN_ROLE_TOKENS):
        return "train"
    for keyword in call.keywords:
        if keyword.arg == "shuffle" and isinstance(keyword.value, ast.Constant):
            return "train" if keyword.value.value is True else None
    return None


def analyze_training_batch_contract(code: str) -> TrainingBatchContract:
    """Find a role-scoped, statically controllable training batch-size knob."""
    try:
        module = ast.parse(code or "")
    except SyntaxError as exc:
        return TrainingBatchContract(unsupported_reason=f"syntax error: {exc.msg}")

    assignments: dict[str, ast.expr] = {}
    parent_targets: dict[int, str] = {}
    enclosing_functions: dict[int, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    loader_factory_batch_parameters: dict[str, tuple[str, int]] = {}

    class _FunctionContextVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[ast.FunctionDef | ast.AsyncFunctionDef] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.stack.append(node)
            self.generic_visit(node)
            self.stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.stack.append(node)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            if self.stack:
                enclosing_functions[id(node)] = self.stack[-1]
            self.generic_visit(node)

    _FunctionContextVisitor().visit(module)
    for function in (
        node for node in ast.walk(module) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ):
        parameters = [*function.args.posonlyargs, *function.args.args]
        parameter_indexes = {parameter.arg: index for index, parameter in enumerate(parameters)}
        factory_parameters: set[str] = set()
        for inner_call in (node for node in ast.walk(function) if isinstance(node, ast.Call)):
            if "dataloader" not in _call_name(inner_call):
                continue
            batch_expr = next(
                (keyword.value for keyword in inner_call.keywords if keyword.arg == "batch_size"),
                inner_call.args[1] if len(inner_call.args) >= 2 else None,
            )
            if isinstance(batch_expr, ast.Name) and batch_expr.id in parameter_indexes:
                factory_parameters.add(batch_expr.id)
        if len(factory_parameters) == 1:
            parameter_name = next(iter(factory_parameters))
            loader_factory_batch_parameters[function.name.lower()] = (
                parameter_name,
                parameter_indexes[parameter_name],
            )
    for node in ast.walk(module):
        name = _assignment_name(node)
        value = _assignment_value(node)
        if name and value is not None:
            assignments.setdefault(name, value)
            if isinstance(value, ast.Call):
                parent_targets[id(value)] = name

    sites: list[TrainingBatchSite] = []
    symbols: set[str] = set()
    diagnostics: list[str] = []
    site_values: list[int] = []
    for call in (node for node in ast.walk(module) if isinstance(node, ast.Call)):
        role = _loader_role(call, parent_targets.get(id(call)))
        if role != "train":
            continue
        candidates: list[tuple[str, ast.expr]] = []
        for keyword in call.keywords:
            if keyword.arg in {"batch_size", "train_batch_size", "per_device_train_batch_size"}:
                candidates.append((f"keyword:{keyword.arg}", keyword.value))
        if not candidates:
            call_leaf = _call_name(call).rsplit(".", 1)[-1]
            factory_parameter = loader_factory_batch_parameters.get(call_leaf)
            if factory_parameter is not None:
                parameter_name, parameter_index = factory_parameter
                if len(call.args) > parameter_index:
                    candidates.append((f"positional:{parameter_index}", call.args[parameter_index]))
                else:
                    keyword_expr = next(
                        (keyword.value for keyword in call.keywords if keyword.arg == parameter_name),
                        None,
                    )
                    if keyword_expr is not None:
                        candidates.append((f"keyword:{parameter_name}", keyword_expr))
            elif "dataloader" in _call_name(call) and len(call.args) >= 2:
                candidates.append(("positional:1", call.args[1]))
        for argument, expr in candidates:
            value = _resolve_static_int(expr, assignments)
            parameter_values: list[tuple[int, ast.expr]] = []
            enclosing = enclosing_functions.get(id(call))
            if value is None and enclosing is not None and isinstance(expr, ast.Name):
                parameters = [*enclosing.args.posonlyargs, *enclosing.args.args]
                parameter_names = [parameter.arg for parameter in parameters]
                if expr.id in parameter_names:
                    parameter_index = parameter_names.index(expr.id)
                    for caller in (node for node in ast.walk(module) if isinstance(node, ast.Call)):
                        caller_name = _call_name(caller).rsplit(".", 1)[-1]
                        if caller_name != enclosing.name.lower():
                            continue
                        caller_expr = caller.args[parameter_index] if len(caller.args) > parameter_index else None
                        if caller_expr is None:
                            caller_expr = next(
                                (keyword.value for keyword in caller.keywords if keyword.arg == expr.id),
                                None,
                            )
                        caller_value = _resolve_static_int(caller_expr, assignments)
                        if caller_value is not None and caller_expr is not None:
                            parameter_values.append((caller_value, caller_expr))
                    if parameter_values:
                        value = max(item[0] for item in parameter_values)
                        symbols.add(expr.id)
                        for _, caller_expr in parameter_values:
                            symbols.update(name for name in _expr_names(caller_expr) if _name_looks_like_train_batch(name))
                        if len({item[0] for item in parameter_values}) > 1:
                            diagnostics.append(
                                f"line {getattr(call, 'lineno', 0)} uses static fallback batch values; probing starts from {value}"
                            )
            if value is None:
                diagnostics.append(f"line {getattr(call, 'lineno', 0)} training batch expression is not statically resolvable")
                continue
            site_values.append(value)
            symbols.update(name for name in _expr_names(expr) if _name_looks_like_train_batch(name))
            sites.append(
                TrainingBatchSite(
                    lineno=int(getattr(call, "lineno", 0)),
                    col_offset=int(getattr(call, "col_offset", 0)),
                    argument=argument,
                    expression=ast.unparse(expr),
                )
            )

    if not sites:
        reason = "no statically proven training DataLoader batch argument"
        if diagnostics:
            reason = diagnostics[0]
        return TrainingBatchContract(unsupported_reason=reason, diagnostics=tuple(diagnostics))

    unique_values = set(site_values)
    if len(unique_values) != 1:
        return TrainingBatchContract(
            batch_symbols=tuple(sorted(symbols)),
            train_sites=tuple(sites),
            unsupported_reason="training loader sites resolve to different initial batch sizes",
            diagnostics=tuple(diagnostics),
        )

    minimum = 1
    for node in ast.walk(module):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1 or len(node.comparators) != 1:
            continue
        left_names = _expr_names(node.left)
        right_value = _resolve_static_int(node.comparators[0], assignments)
        if left_names & symbols and right_value is not None and isinstance(node.ops[0], (ast.GtE, ast.Gt)):
            minimum = max(minimum, right_value + (1 if isinstance(node.ops[0], ast.Gt) else 0))

    return TrainingBatchContract(
        initial_batch_size=site_values[0],
        minimum_batch_size=minimum,
        batch_symbols=tuple(sorted(symbols)),
        train_sites=tuple(sites),
        confidence="high",
        diagnostics=tuple(diagnostics),
    )


def normalized_mlevolve_script_signature(code: str) -> str:
    """Return a stable signature for generated code while ignoring batch-size edits."""
    normalized = code or ""
    for pattern in _BATCH_PROBE_NORMALIZE_PATTERNS:
        normalized = re.sub(pattern, r"\1<BS>", normalized)
    return sha1(normalized.encode("utf-8")).hexdigest()


def code_supports_batch_probe(code: str) -> bool:
    return analyze_training_batch_contract(code).supported


def detect_initial_batch_size(code: str) -> int | None:
    contract = analyze_training_batch_contract(code)
    if contract.initial_batch_size is not None:
        return contract.initial_batch_size
    match = _BATCH_PARAM_PATTERN.search(code or "")
    return _safe_int(match.group(1)) if match else None


def detect_epoch_count(code: str) -> int | None:
    match = _EPOCH_PARAM_PATTERN.search(code or "")
    if not match:
        return None
    return _safe_int(match.group(1))


def detect_input_resolution(code: str) -> int | str | None:
    code = code or ""
    tuple_match = _RESOLUTION_TUPLE_PATTERN.search(code)
    if tuple_match:
        height = _safe_int(tuple_match.group(1))
        width = _safe_int(tuple_match.group(2))
        if height is not None and width is not None:
            return height if height == width else f"{height}x{width}"
    single_match = _RESOLUTION_SINGLE_PATTERN.search(code)
    if not single_match:
        return None
    return _safe_int(single_match.group(1))


def detect_fold_count(code: str) -> int | None:
    match = _FOLD_PARAM_PATTERN.search(code or "")
    if not match:
        return None
    return _safe_int(match.group(1))


def detect_ensemble_count(code: str) -> int | None:
    match = _ENSEMBLE_PARAM_PATTERN.search(code or "")
    if not match:
        return None
    return _safe_int(match.group(1))


def detect_tta_count(code: str) -> int | None:
    match = _TTA_PARAM_PATTERN.search(code or "")
    if match:
        return _safe_int(match.group(1))
    if _TTA_BOOL_PATTERN.search(code or ""):
        return 1
    return None


def detect_model_key(code: str) -> str | None:
    match = _MODEL_PARAM_PATTERN.search(code or "")
    if match:
        return _clean_model_key(match.group(1))

    timm_match = re.search(r"timm\.create_model\(\s*['\"]([^'\"]+)['\"]", code or "")
    if timm_match:
        return _clean_model_key(timm_match.group(1))

    hf_match = re.search(r"\.from_pretrained\(\s*['\"]([^'\"]+)['\"]", code or "")
    if hf_match:
        return _clean_model_key(hf_match.group(1))

    torch_hub_match = re.search(r"torch\.hub\.load\([^,]+,\s*['\"]([^'\"]+)['\"]", code or "")
    if torch_hub_match:
        return _clean_model_key(torch_hub_match.group(1))

    return None


def detect_model_family(code: str) -> str | None:
    match = _MODEL_FAMILY_PARAM_PATTERN.search(code or "")
    if not match:
        return None
    return _clean_model_key(match.group(1))


def detect_model_branch(code: str) -> str | None:
    match = _MODEL_BRANCH_PARAM_PATTERN.search(code or "")
    if not match:
        return None
    return _canonical_branch_name(match.group(1))


def detect_framework(code: str) -> str:
    code_lower = (code or "").lower()
    if "import torch" in code_lower or "from torch" in code_lower or "torch." in code_lower:
        return "pytorch"
    if "tensorflow" in code_lower or "keras" in code_lower:
        return "tensorflow"
    if "xgboost" in code_lower:
        return "xgboost"
    if "lightgbm" in code_lower:
        return "lightgbm"
    if "sklearn" in code_lower or "scikit" in code_lower:
        return "sklearn"
    return "pytorch"


def detect_uses_amp(code: str) -> bool:
    code_lower = (code or "").lower()
    return any(
        token in code_lower
        for token in (
            "autocast",
            "gradscaler",
            "torch.amp",
            "cuda.amp",
            "bfloat16",
            "float16",
            "transformer_engine",
            "te.autocast",
        )
    )


def detect_precision_mode(code: str) -> str | None:
    code_text = code or ""
    for pattern in (_PRECISION_STRING_PATTERN, _PRECISION_TORCH_DTYPE_PATTERN, _AUTOCAST_DTYPE_PATTERN):
        match = pattern.search(code_text)
        if match:
            precision = _normalize_precision_mode(match.group(1))
            if precision:
                return precision
    code_lower = code_text.lower()
    if "nvfp4" in code_lower or "nvfp4blockscaling" in code_lower:
        return "nvfp4_te"
    if "mxfp8" in code_lower or "mxfp8blockscaling" in code_lower:
        return "mxfp8_te"
    if _uses_transformer_engine(code_text) and (
        "fp8" in code_lower
        or "delayedscaling" in code_lower
        or "float8" in code_lower
        or "format.hybrid" in code_lower
    ):
        return "fp8_te"
    if "torch.bfloat16" in code_lower or "bfloat16" in code_lower or "bf16" in code_lower:
        return "bf16"
    if "torch.float16" in code_lower or "float16" in code_lower or "fp16" in code_lower:
        return "fp16"
    if "allow_tf32" in code_lower or "set_float32_matmul_precision" in code_lower or "tf32" in code_lower:
        return "tf32"
    if "torch.float32" in code_lower or "float32" in code_lower or "fp32" in code_lower:
        return "fp32"
    if "autocast" in code_lower:
        return "mixed"
    return None


def detect_precision_backend(code: str) -> str | None:
    if _uses_transformer_engine(code):
        return "transformer_engine"
    return None


def detect_precision_model_adaptation(code: str) -> str | None:
    code_lower = (code or "").lower()
    if not _uses_transformer_engine(code):
        return None
    adaptation_tokens = (
        "te.linear",
        "te.layernormlinear",
        "te.transformerlayer",
        "transformer_engine.pytorch.linear",
        "transformer_engine.pytorch.layernormlinear",
        "transformer_engine.pytorch.transformerlayer",
        "replace_layers",
        "replace_linear",
        "convert_to_transformer_engine",
        "te_module",
    )
    if any(token in code_lower for token in adaptation_tokens):
        return "te_module_replacement"
    return None


def detect_learning_rate(code: str) -> float | None:
    match = _LEARNING_RATE_PARAM_PATTERN.search(code or "")
    if not match:
        return None
    return _safe_float(match.group(1))


def detect_weight_decay(code: str) -> float | None:
    match = _WEIGHT_DECAY_PARAM_PATTERN.search(code or "")
    if not match:
        return None
    return _safe_float(match.group(1))


def detect_gradient_accumulation_steps(code: str) -> int | None:
    match = _GRADIENT_ACCUMULATION_PARAM_PATTERN.search(code or "")
    if not match:
        return None
    return _safe_int(match.group(1))


def detect_num_workers(code: str) -> int | None:
    match = _NUM_WORKERS_PARAM_PATTERN.search(code or "")
    if not match:
        return None
    return _safe_int(match.group(1))


def detect_requires_gpu(code: str) -> bool:
    code_lower = (code or "").lower()
    if not code_lower:
        return True
    if any(token in code_lower for token in ("cuda", "torch.cuda", ".to(device)", ".cuda(")):
        return True
    if detect_framework(code) == "pytorch":
        return True
    return False


def _canonical_branch_name(value: str | None) -> str | None:
    cleaned = _clean_model_key(value)
    if not cleaned:
        return None
    normalized = cleaned.lower().replace(":", "-").replace("/", "-")
    normalized = re.sub(r"[_-]+", "-", normalized)
    compact = normalized.replace("-", "_")
    resnet = re.search(r"(?:^|_)resnet_?(\d+)(?:d|v\d+|_[a-z0-9]+)?(?:_|$)", compact)
    if resnet:
        return f"resnet{resnet.group(1)}"
    efficientnet = re.search(r"(?:^|_)efficientnet[_-]?([a-z]\d+)", compact)
    if efficientnet:
        return f"efficientnet-{efficientnet.group(1)}"
    convnext = re.search(r"(?:^|_)convnext[_-]?([a-z]+)", compact)
    if convnext:
        return f"convnext-{convnext.group(1)}"
    swin = re.search(r"(?:^|_)swin[_-]?([a-z0-9]+)", compact)
    if swin:
        return f"swin-{swin.group(1)}"
    vit = re.search(r"(?:^|_)(?:vit|vision-transformer)[_-]?([a-z0-9]+)?", compact)
    if vit:
        suffix = vit.group(1)
        return f"vit-{suffix}" if suffix else "vit"
    return normalized or None


def infer_branch_name(model_key: str | None, code: str = "") -> str | None:
    text = f"{model_key or ''} {code or ''}"
    branch = _canonical_branch_name(model_key)
    if branch:
        return branch
    for pattern in (
        r"timm\.create_model\(\s*['\"]([^'\"]+)['\"]",
        r"\.from_pretrained\(\s*['\"]([^'\"]+)['\"]",
        r"torch\.hub\.load\([^,]+,\s*['\"]([^'\"]+)['\"]",
        r"\b(resnet\d+[a-z]?(?:[_-]v\d+|[_-][a-z0-9]+)?)\b",
        r"\b(efficientnet[_-]?[a-z]\d+)\b",
        r"\b(convnext[_-]?[a-z]+)\b",
        r"\b(swin[_-]?[a-z0-9]+)\b",
        r"\b(vit[_-]?[a-z0-9]+)\b",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            branch = _canonical_branch_name(match.group(1))
            if branch:
                return branch
    return None


def infer_model_family(model_key: str | None, code: str = "") -> str | None:
    branch = infer_branch_name(model_key, code)
    if branch:
        return branch
    text = f"{model_key or ''} {code or ''}".lower()
    if any(token in text for token in ("vit", "swin", "deit", "transformer", "bert", "roberta", "llama", "gpt")):
        return "transformer"
    if any(token in text for token in ("resnet", "efficientnet", "convnext", "cnn", "densenet", "mobilenet")):
        return "cnn"
    if any(token in text for token in ("unet", "segformer", "deeplab")):
        return "segmentation"
    if any(token in text for token in ("diffusion", "stable-diffusion", "vae")):
        return "diffusion"
    if any(token in text for token in ("xgboost", "lightgbm", "catboost", "randomforest")):
        return "gbdt"
    return None


def introspect_training_script(code: str) -> dict[str, Any]:
    """Extract scheduler/MCP candidate hints from generated code."""
    code = code or ""
    explicit_branch_name = detect_model_branch(code)
    explicit_model_family = detect_model_family(code)
    model_key = detect_model_key(code)
    if (explicit_branch_name or explicit_model_family) and not model_key:
        model_key = explicit_branch_name or explicit_model_family
    inferred_branch_name = infer_branch_name(model_key, code)
    branch_name = explicit_branch_name or _canonical_branch_name(explicit_model_family) or inferred_branch_name
    branch_name_source = (
        "explicit_model_branch"
        if explicit_branch_name
        else ("model_family_alias" if explicit_model_family else ("inferred" if inferred_branch_name else None))
    )
    model_family = branch_name
    model_family_source = branch_name_source
    if not explicit_branch_name and not explicit_model_family and inferred_branch_name:
        logger.warning(
            "Generated script is missing MODEL_BRANCH; inferred branch_name=%s from model key/code.",
            inferred_branch_name,
        )
    batch_contract = analyze_training_batch_contract(code)
    candidate: dict[str, Any] = {
        "model_key": model_key,
        "branch_name": branch_name,
        "branch_name_source": branch_name_source,
        "model_family": model_family,
        "model_family_source": model_family_source,
        "proposed_batch_size": detect_initial_batch_size(code),
        "minimum_batch_size": batch_contract.minimum_batch_size if batch_contract.supported else None,
        "batch_probe_supported": batch_contract.supported,
        "batch_contract": batch_contract.to_dict(),
        "proposed_epochs": detect_epoch_count(code),
        "input_resolution": detect_input_resolution(code),
        "fold_count": detect_fold_count(code),
        "ensemble_count": detect_ensemble_count(code),
        "tta_count": detect_tta_count(code),
        "requires_gpu": detect_requires_gpu(code),
        "script_signature": normalized_mlevolve_script_signature(code) if code.strip() else None,
        "uses_amp": detect_uses_amp(code),
        "precision_mode": detect_precision_mode(code),
        "precision_backend": detect_precision_backend(code),
        "precision_model_adaptation": detect_precision_model_adaptation(code),
        "learning_rate": detect_learning_rate(code),
        "weight_decay": detect_weight_decay(code),
        "gradient_accumulation_steps": detect_gradient_accumulation_steps(code),
        "num_workers": detect_num_workers(code),
        "framework": detect_framework(code),
    }
    return {key: value for key, value in candidate.items() if value is not None}


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_precision_mode(value: str) -> str | None:
    normalized = str(value or "").strip().lower().replace("torch.", "")
    normalized = normalized.replace("-", "_")
    if normalized in {"bf16", "bfloat16"}:
        return "bf16"
    if normalized in {"fp16", "float16", "half"}:
        return "fp16"
    if normalized in {"fp32", "float32"}:
        return "fp32"
    if normalized == "tf32":
        return "tf32"
    if normalized in {"fp8", "float8", "fp8_te", "te_fp8"}:
        return "fp8_te"
    if normalized in {"mxfp8", "mx_fp8", "mxfp8_te", "te_mxfp8"}:
        return "mxfp8_te"
    if normalized in {"nvfp4", "fp4", "nvfp4_te", "te_nvfp4"}:
        return "nvfp4_te"
    if normalized in {"amp", "mixed", "mixed_precision"}:
        return "mixed"
    return None


def _uses_transformer_engine(code: str) -> bool:
    code_lower = (code or "").lower()
    return any(
        token in code_lower
        for token in (
            "import transformer_engine",
            "from transformer_engine",
            "transformer_engine.",
            "te.autocast",
            "te.linear",
            "te.layernormlinear",
            "te.transformerlayer",
        )
    )


def _clean_model_key(value: str) -> str | None:
    value = str(value or "").strip()
    if not value:
        return None
    return value

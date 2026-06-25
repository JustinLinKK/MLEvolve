"""Lightweight introspection for generated MLEvolve training scripts."""

from __future__ import annotations

import re
import logging
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


def normalized_mlevolve_script_signature(code: str) -> str:
    """Return a stable signature for generated code while ignoring batch-size edits."""
    normalized = code or ""
    for pattern in _BATCH_PROBE_NORMALIZE_PATTERNS:
        normalized = re.sub(pattern, r"\1<BS>", normalized)
    return sha1(normalized.encode("utf-8")).hexdigest()


def code_supports_batch_probe(code: str) -> bool:
    return _BATCH_PROBE_ENABLE_PATTERN.search(code or "") is not None


def detect_initial_batch_size(code: str) -> int | None:
    match = _BATCH_PARAM_PATTERN.search(code or "")
    if not match:
        return None
    return _safe_int(match.group(1))


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


def infer_model_family(model_key: str | None, code: str = "") -> str | None:
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
    explicit_model_family = detect_model_family(code)
    model_key = detect_model_key(code)
    if explicit_model_family and not model_key:
        model_key = explicit_model_family
    inferred_model_family = infer_model_family(model_key, code)
    model_family = explicit_model_family or inferred_model_family
    model_family_source = "explicit" if explicit_model_family else ("inferred" if inferred_model_family else None)
    if not explicit_model_family and inferred_model_family:
        logger.warning(
            "Generated script is missing MODEL_FAMILY; inferred legacy model_family=%s from model key/code.",
            inferred_model_family,
        )
    candidate: dict[str, Any] = {
        "model_key": model_key,
        "model_family": model_family,
        "model_family_source": model_family_source,
        "proposed_batch_size": detect_initial_batch_size(code),
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

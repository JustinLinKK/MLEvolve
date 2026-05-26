"""Lightweight introspection for generated MLEvolve training scripts."""

from __future__ import annotations

import re
from hashlib import sha1
from typing import Any


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

    return None


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
        )
    )


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
    model_key = detect_model_key(code)
    candidate: dict[str, Any] = {
        "model_key": model_key,
        "model_family": infer_model_family(model_key, code),
        "proposed_batch_size": detect_initial_batch_size(code),
        "proposed_epochs": detect_epoch_count(code),
        "input_resolution": detect_input_resolution(code),
        "fold_count": detect_fold_count(code),
        "ensemble_count": detect_ensemble_count(code),
        "tta_count": detect_tta_count(code),
        "requires_gpu": detect_requires_gpu(code),
        "script_signature": normalized_mlevolve_script_signature(code) if code.strip() else None,
        "uses_amp": detect_uses_amp(code),
        "framework": detect_framework(code),
    }
    return {key: value for key, value in candidate.items() if value is not None}


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_model_key(value: str) -> str | None:
    value = str(value or "").strip()
    if not value:
        return None
    return value

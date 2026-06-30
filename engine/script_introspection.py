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


def _assignment_pattern(names: tuple[str, ...], value_pattern: str) -> re.Pattern[str]:
    joined = "|".join(re.escape(name) for name in names)
    return re.compile(rf"\b(?:{joined})\b\s*=\s*{value_pattern}")


_BATCH_PARAM_PATTERN = _assignment_pattern(BATCH_PARAM_NAMES, r"(\d+)")
_EPOCH_PARAM_PATTERN = _assignment_pattern(EPOCH_PARAM_NAMES, r"(\d+)")
_RESOLUTION_TUPLE_PATTERN = _assignment_pattern(RESOLUTION_PARAM_NAMES, r"\(?\s*(\d+)\s*,\s*(\d+)")
_RESOLUTION_SINGLE_PATTERN = _assignment_pattern(RESOLUTION_PARAM_NAMES, r"(\d+)")
_FOLD_PARAM_PATTERN = _assignment_pattern(FOLD_PARAM_NAMES, r"(\d+)")
_ENSEMBLE_PARAM_PATTERN = _assignment_pattern(ENSEMBLE_PARAM_NAMES, r"(\d+)")
_TTA_PARAM_PATTERN = _assignment_pattern(TTA_PARAM_NAMES, r"(\d+)")
_TTA_BOOL_PATTERN = re.compile(r"\b(?:USE_TTA|use_tta|tta)\b\s*=\s*True\b")
_MODEL_PARAM_PATTERN = _assignment_pattern(MODEL_PARAM_NAMES, r"['\"]([^'\"]+)['\"]")
_BATCH_PROBE_NORMALIZE_PATTERN = re.compile(
    rf"(\b(?:{'|'.join(re.escape(name) for name in BATCH_PARAM_NAMES)})\b\s*=\s*)([^,\n\)]*)"
)


def normalized_mlevolve_script_signature(code: str) -> str:
    """Return a stable signature while ignoring direct batch-size edits."""
    normalized = code or ""
    normalized = _BATCH_PROBE_NORMALIZE_PATTERN.sub(r"\1<BS>", normalized)
    return sha1(normalized.encode("utf-8")).hexdigest()


def detect_initial_batch_size(code: str) -> int | None:
    match = _BATCH_PARAM_PATTERN.search(code or "")
    return _safe_int(match.group(1)) if match else None


def detect_epoch_count(code: str) -> int | None:
    match = _EPOCH_PARAM_PATTERN.search(code or "")
    return _safe_int(match.group(1)) if match else None


def detect_input_resolution(code: str) -> int | str | None:
    code = code or ""
    tuple_match = _RESOLUTION_TUPLE_PATTERN.search(code)
    if tuple_match:
        height = _safe_int(tuple_match.group(1))
        width = _safe_int(tuple_match.group(2))
        if height is not None and width is not None:
            return height if height == width else f"{height}x{width}"
    single_match = _RESOLUTION_SINGLE_PATTERN.search(code)
    return _safe_int(single_match.group(1)) if single_match else None


def detect_fold_count(code: str) -> int | None:
    match = _FOLD_PARAM_PATTERN.search(code or "")
    return _safe_int(match.group(1)) if match else None


def detect_ensemble_count(code: str) -> int | None:
    match = _ENSEMBLE_PARAM_PATTERN.search(code or "")
    return _safe_int(match.group(1)) if match else None


def detect_tta_count(code: str) -> int | None:
    match = _TTA_PARAM_PATTERN.search(code or "")
    if match:
        return _safe_int(match.group(1))
    if _TTA_BOOL_PATTERN.search(code or ""):
        return 1
    return None


def detect_model_key(code: str) -> str | None:
    code = code or ""
    match = _MODEL_PARAM_PATTERN.search(code)
    if match:
        return _clean_model_key(match.group(1))

    timm_match = re.search(r"timm\.create_model\(\s*['\"]([^'\"]+)['\"]", code)
    if timm_match:
        return _clean_model_key(timm_match.group(1))

    pretrained_match = re.search(r"\.from_pretrained\(\s*['\"]([^'\"]+)['\"]", code)
    if pretrained_match:
        return _clean_model_key(pretrained_match.group(1))

    return None


def introspect_training_script(code: str) -> dict[str, Any]:
    """Extract comparison-friendly training intent hints from generated code."""
    code = code or ""
    candidate: dict[str, Any] = {
        "model_key": detect_model_key(code),
        "proposed_batch_size": detect_initial_batch_size(code),
        "proposed_epochs": detect_epoch_count(code),
        "input_resolution": detect_input_resolution(code),
        "fold_count": detect_fold_count(code),
        "ensemble_count": detect_ensemble_count(code),
        "tta_count": detect_tta_count(code),
        "script_signature": normalized_mlevolve_script_signature(code) if code.strip() else None,
    }
    return {key: value for key, value in candidate.items() if value is not None}


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_model_key(value: str) -> str:
    return value.strip().replace("\\", "/")

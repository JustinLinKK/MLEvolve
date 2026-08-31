"""Strict canonical family/hardware identity generation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Mapping

from engine.script_introspection import introspect_training_script

from .models import PROFILE_SCHEMA_VERSION, ProfileIdentity


_FAMILY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(?:^|[/_.-])resnet\d*(?:$|[/_.-])", "resnet"),
    (r"(?:^|[/_.-])efficientnet", "efficientnet"),
    (r"(?:^|[/_.-])convnext", "convnext"),
    (r"(?:^|[/_.-])swin", "swin"),
    (r"(?:^|[/_.-])(?:vit|vision[-_]?transformer)", "vit"),
    (r"(?:^|[/_.-])densenet", "densenet"),
    (r"(?:^|[/_.-])mobilenet", "mobilenet"),
    (r"(?:^|[/_.-])unet", "unet"),
    (r"(?:^|[/_.-])bert", "bert"),
    (r"(?:^|[/_.-])(?:gpt|llama|mistral)", "decoder_transformer"),
    (r"(?:^|[/_.-])(?:xgboost|xgb)", "xgboost"),
    (r"(?:^|[/_.-])lightgbm", "lightgbm"),
)


def _slug(value: Any, *, default: str = "unknown") -> str:
    normalized = re.sub(r"[^a-z0-9.+-]+", "-", str(value or "").strip().lower()).strip("-")
    return normalized or default


def canonical_model_family(*values: Any) -> tuple[str | None, float]:
    """Resolve only sufficiently specific families.

    Explicit values are trusted more than model names, while broad labels such
    as ``cnn`` or ``transformer`` intentionally remain uncertain.
    """

    broad = {"cnn", "transformer", "model", "classifier", "neural-network", "unknown", "none"}
    resolved: list[tuple[str, float]] = []
    for index, value in enumerate(values):
        candidate = _slug(value, default="")
        if not candidate:
            continue
        padded = f"/{candidate}/"
        for pattern, family in _FAMILY_PATTERNS:
            if re.search(pattern, padded):
                resolved.append((family, 1.0 if index == 0 else 0.85))
                break
        else:
            if candidate not in broad and len(candidate) >= 3 and index == 0:
                resolved.append((candidate, 0.9))
    if not resolved:
        return None, 0.0
    families = {item[0] for item in resolved}
    if len(families) != 1:
        return None, 0.0
    family = resolved[0][0]
    return family, max(item[1] for item in resolved)


def _major(value: Any, *, prefix: str) -> str:
    match = re.search(r"(\d+)", str(value or ""))
    return f"{prefix}-{match.group(1)}" if match else f"{prefix}-unknown"


def _architecture(family: str, introspection: Mapping[str, Any]) -> str:
    explicit = introspection.get("architecture_family")
    if explicit:
        value = _slug(explicit)
        aliases = {"vision-transformer": "transformer", "decoder-transformer": "transformer"}
        return aliases.get(value, value)
    if family in {"resnet", "efficientnet", "convnext", "densenet", "mobilenet", "unet"}:
        return "cnn"
    if family in {"vit", "swin", "bert", "decoder_transformer"}:
        return "transformer"
    if family in {"xgboost", "lightgbm"}:
        return "tree"
    return "unknown"


def workload_bucket(*, task_description: str = "", introspection: Mapping[str, Any], family: str = "") -> str:
    description = task_description.lower()
    resolution = introspection.get("input_resolution") or "unknown"
    if any(token in description for token in ("image", "vision", "pixel", "classification")):
        modality = "image"
    elif any(token in description for token in ("text", "token", "language", "nlp")):
        modality = "text"
    elif any(token in description for token in ("audio", "speech", "sound")):
        modality = "audio"
    elif any(token in description for token in ("tabular", "csv", "regression")):
        modality = "tabular"
    elif family in {"resnet", "efficientnet", "convnext", "swin", "vit", "densenet", "mobilenet", "unet"}:
        modality = "image"
    elif family in {"bert", "decoder_transformer"}:
        modality = "text"
    elif family in {"xgboost", "lightgbm"}:
        modality = "tabular"
    else:
        modality = "unknown"
    if "regression" in description:
        task = "regression"
    elif "segmentation" in description:
        task = "segmentation"
    elif "detection" in description:
        task = "detection"
    elif "classification" in description:
        task = "classification"
    else:
        task = "generic"
    return _slug(f"{modality}-{task}-{resolution}")


def build_profile_identity(
    *,
    code: str,
    hardware: Mapping[str, Any],
    backend: str | None,
    task_description: str = "",
    model_family_hint: str | None = None,
    resource_slice: str | None = None,
    minimum_confidence: float = 0.75,
) -> ProfileIdentity | None:
    """Create the complete exact key, or return ``None`` for a cold start."""

    facts = introspect_training_script(code or "")
    family, confidence = canonical_model_family(
        model_family_hint,
        facts.get("model_family"),
        facts.get("model_key"),
        facts.get("architecture_family"),
    )
    if family is None or confidence < float(minimum_confidence):
        return None
    hardware_key = _slug(hardware.get("hardware_key"), default="")
    gpu_name = _slug(hardware.get("gpu_name"), default="")
    if not hardware_key or not gpu_name or gpu_name == "cuda-unavailable":
        return None
    framework = _slug(facts.get("framework"), default="unknown")
    framework_version = hardware.get("torch_version") if framework == "pytorch" else hardware.get("framework_version")
    framework_major = _major(framework_version, prefix=framework)
    cuda_major = _major(hardware.get("cuda_runtime"), prefix="cuda")
    runtime_class = _slug(f"{framework_major}_{cuda_major}")
    accelerator_key = _slug(
        f"{gpu_name}-cc{hardware.get('compute_capability') or 'unknown'}-vram{hardware.get('total_vram_mb') or 'unknown'}"
    )
    if resource_slice:
        slice_key = _slug(resource_slice)
    else:
        visible = os.getenv("CUDA_VISIBLE_DEVICES", "unspecified")
        slice_key = _slug(f"full-or-visible-{visible}-{hardware.get('total_vram_mb') or 'unknown'}")
    backend_class = _slug(backend, default="")
    if not backend_class:
        return None
    bucket = workload_bucket(task_description=task_description, introspection=facts, family=family)
    if bucket.startswith("unknown-") or framework_major.endswith("-unknown") or cuda_major.endswith("-unknown"):
        return None
    identity_payload = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "model_family": family,
        "architecture_type": _architecture(family, facts),
        "hardware_key": hardware_key,
        "accelerator_key": accelerator_key,
        "resource_slice_key": slice_key,
        "runtime_class": runtime_class,
        "framework_major": framework_major,
        "cuda_major": cuda_major,
        "backend_class": backend_class,
        "workload_bucket": bucket,
    }
    digest = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ProfileIdentity(**identity_payload, family_confidence=confidence, profile_key=digest)


def identities_compatible(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    required = (
        "model_family",
        "architecture_type",
        "accelerator_key",
        "resource_slice_key",
        "backend_class",
        "workload_bucket",
        "framework_major",
        "cuda_major",
    )
    return all(str(left.get(key) or "") == str(right.get(key) or "") for key in required)

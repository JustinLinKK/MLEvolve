"""Architecture-aware training precision policy shared across MLEvolve.

The policy intentionally distinguishes native, supported training formats from
capability-only integer formats and from storage/inference-only formats.  A
format is eligible for recommendation only when the repository has an
end-to-end training policy for it; low-level CUDA type availability alone is
not sufficient.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


PRECISION_MODE_NORMAL = "normal"
PRECISION_MODE_AGGRESSIVE = "aggressive"
PRECISION_OPTIMIZATION_MODES = frozenset(
    {PRECISION_MODE_NORMAL, PRECISION_MODE_AGGRESSIVE}
)

_BASE_POLICIES = ("fp32", "disabled")
_POLICY_TO_FEATURES: dict[str, tuple[str, ...]] = {
    "fp16_amp": ("amp", "fp16"),
    "bf16_amp": ("amp", "bf16"),
    "tf32": ("tf32",),
    "fp8_te": ("fp8", "fp8_e4m3", "fp8_e5m2"),
    "mxfp8_te": ("mxfp8",),
    "nvfp4_te": ("nvfp4",),
}
_INTEGER_PREFIXES = ("int", "uint", "sint", "u4", "s4", "b1")
_KNOWN_HIDDEN_PRECISION_FEATURES = frozenset(
    {"fp4", "fp64", "fp6", "mxfp4"}
)


def normalize_precision_optimization_mode(value: Any) -> str:
    mode = str(value or PRECISION_MODE_NORMAL).strip().lower().replace("-", "_")
    if mode not in PRECISION_OPTIMIZATION_MODES:
        expected = ", ".join(sorted(PRECISION_OPTIMIZATION_MODES))
        raise ValueError(
            f"Unsupported agent.precision_optimization_mode: {value}. "
            f"Expected one of: {expected}"
        )
    return mode


def normalize_precision_policy_name(value: Any) -> str | None:
    name = str(value or "").strip().lower().replace("-", "_").replace("torch.", "")
    aliases = {
        "fp16": "fp16_amp",
        "float16": "fp16_amp",
        "half": "fp16_amp",
        "fp16_amp": "fp16_amp",
        "bf16": "bf16_amp",
        "bfloat16": "bf16_amp",
        "bf16_amp": "bf16_amp",
        "fp32": "fp32",
        "float32": "fp32",
        "tf32": "tf32",
        "fp8": "fp8_te",
        "float8": "fp8_te",
        "te_fp8": "fp8_te",
        "fp8_te": "fp8_te",
        "mxfp8": "mxfp8_te",
        "mx_fp8": "mxfp8_te",
        "te_mxfp8": "mxfp8_te",
        "mxfp8_te": "mxfp8_te",
        "nvfp4": "nvfp4_te",
        "te_nvfp4": "nvfp4_te",
        "nvfp4_te": "nvfp4_te",
        "disabled": "disabled",
        "none": "disabled",
    }
    return aliases.get(name)


@dataclass(frozen=True, slots=True)
class PrecisionPolicy:
    mode: str
    architecture: str
    compute_capability: str | None
    allowed_policies: tuple[str, ...]
    permitted_features: tuple[str, ...]
    recommended_features: tuple[str, ...]
    integer_capability_indicators: tuple[str, ...]
    hidden_features: tuple[str, ...]
    requires_transformer_engine: tuple[str, ...]
    fallback_policy: str = "fp32"

    def allows(self, value: Any) -> bool:
        normalized = normalize_precision_policy_name(value)
        return bool(normalized and normalized in self.allowed_policies)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_precision_policy(
    hardware: Mapping[str, Any] | None = None,
    *,
    mode: Any = PRECISION_MODE_NORMAL,
    architecture: Any = None,
    compute_capability: Any = None,
    datatypes: Any = None,
) -> PrecisionPolicy:
    """Return the training precision allowlist for one hardware target."""
    normalized_mode = normalize_precision_optimization_mode(mode)
    source = dict(hardware or {})
    nested = source.get("hardware")
    if isinstance(nested, Mapping):
        source = {**source, **dict(nested)}

    architecture_value = architecture
    if architecture_value is None:
        architecture_value = source.get("architecture") or source.get("architectures")
    capability_value = compute_capability
    if capability_value is None:
        capability_value = source.get("compute_capability") or source.get("compute_capabilities")
    datatype_values = datatypes if datatypes is not None else source.get("datatypes")

    capability = _first_text(capability_value)
    normalized_architecture = _normalize_architecture(architecture_value, capability)
    allowed = list(_BASE_POLICIES)
    if normalized_architecture in {"volta", "turing"}:
        allowed.append("fp16_amp")
    elif normalized_architecture in {"ampere", "ada_lovelace", "hopper", "blackwell"}:
        allowed.extend(("tf32", "bf16_amp", "fp16_amp"))

    if normalized_mode == PRECISION_MODE_AGGRESSIVE:
        if normalized_architecture in {"ada_lovelace", "hopper", "blackwell"}:
            allowed.append("fp8_te")
        if normalized_architecture == "blackwell":
            allowed.extend(("mxfp8_te", "nvfp4_te"))

    permitted_features: list[str] = []
    for policy in allowed:
        for feature in _POLICY_TO_FEATURES.get(policy, ()):
            if feature not in permitted_features:
                permitted_features.append(feature)

    # Aggressive mode expands the deterministic allowlist, but these lower
    # precision recipes remain opt-in experiments rather than default advice.
    recommended_policies = [
        policy
        for policy in allowed
        if policy not in {"fp8_te", "mxfp8_te", "nvfp4_te"}
    ]
    recommended_features: list[str] = []
    for policy in recommended_policies:
        for feature in _POLICY_TO_FEATURES.get(policy, ()):
            if feature not in recommended_features:
                recommended_features.append(feature)

    integer_indicators = tuple(
        value
        for value in _normalize_string_list(datatype_values)
        if value.startswith(_INTEGER_PREFIXES)
    )
    visible = set(permitted_features) | set(integer_indicators)
    known_datatypes = set(_normalize_string_list(datatype_values))
    hidden = sorted((known_datatypes | set(_KNOWN_HIDDEN_PRECISION_FEATURES)) - visible)

    return PrecisionPolicy(
        mode=normalized_mode,
        architecture=normalized_architecture,
        compute_capability=capability,
        allowed_policies=tuple(dict.fromkeys(allowed)),
        permitted_features=tuple(permitted_features),
        recommended_features=tuple(recommended_features),
        integer_capability_indicators=integer_indicators,
        hidden_features=tuple(hidden),
        requires_transformer_engine=tuple(
            policy for policy in allowed if policy in {"fp8_te", "mxfp8_te", "nvfp4_te"}
        ),
    )


def precision_feature_visibility(feature_id: Any, policy: PrecisionPolicy) -> str:
    """Classify a precision feature for datatype-optimization prompts."""
    feature = str(feature_id or "").strip().lower().replace("-", "_")
    if feature in policy.recommended_features:
        return "recommendation"
    if feature in policy.permitted_features:
        return "permitted"
    if feature.startswith("fp8_") and "fp8_te" in policy.allowed_policies:
        return "permitted"
    if feature in policy.integer_capability_indicators or feature.startswith(_INTEGER_PREFIXES):
        return "integer_indicator"
    return "hidden"


def _normalize_architecture(value: Any, capability: str | None) -> str:
    values = _normalize_string_list(value)
    joined = " ".join(values).lower().replace("-", "_").replace(" ", "_")
    for token, normalized in (
        ("blackwell", "blackwell"),
        ("hopper", "hopper"),
        ("ada", "ada_lovelace"),
        ("ampere", "ampere"),
        ("turing", "turing"),
        ("volta", "volta"),
    ):
        if token in joined:
            return normalized

    parsed = _parse_compute_capability(capability)
    if parsed is None:
        return "unknown"
    major, minor = parsed
    if major >= 10:
        return "blackwell"
    if major == 9:
        return "hopper"
    if major == 8 and minor == 9:
        return "ada_lovelace"
    if major == 8:
        return "ampere"
    if major == 7 and minor >= 5:
        return "turing"
    if major == 7:
        return "volta"
    return "unknown"


def _parse_compute_capability(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    text = str(value).strip().lower().removeprefix("sm_")
    try:
        if "." in text:
            major, minor = text.split(".", 1)
            return int(major), int("".join(ch for ch in minor if ch.isdigit()) or 0)
        digits = "".join(ch for ch in text if ch.isdigit())
        if len(digits) >= 2:
            return int(digits[:-1]), int(digits[-1])
    except ValueError:
        return None
    return None


def _first_text(value: Any) -> str | None:
    values = _normalize_string_list(value)
    return values[0] if values else None


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return [str(item).strip().lower().replace("-", "_") for item in values if str(item).strip()]

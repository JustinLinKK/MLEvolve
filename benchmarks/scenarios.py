"""Required context-cache benchmark matrix and observation contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from itertools import product
from typing import Any, Mapping

BENCHMARK_MODES = ("baseline", "local-only", "provider-only", "both")
CONTEXT_LENGTHS = ("actual", "4k", "16k", "64k")
REUSE_MODES = ("same-agent", "cross-agent-common-root")
CONCURRENCY_LEVELS = (1, 4)
IDLE_GAPS_MINUTES = (0, 6, 12, 31, 61)
OPENROUTER_ROUTING_MODES = ("pinned", "production")
VLLM_CACHE_STATES = ("cold", "warm", "disabled")
VLLM_DEPLOYMENTS = ("single-server", "fleet")


@dataclass(frozen=True)
class BenchmarkScenario:
    mode: str
    context_length: str
    reuse: str
    concurrency: int
    idle_gap_minutes: int
    routing: str
    trials: int = 20

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def required_scenarios(*, trials: int = 20) -> list[BenchmarkScenario]:
    return [
        BenchmarkScenario(*values, trials=trials)
        for values in product(
            BENCHMARK_MODES,
            CONTEXT_LENGTHS,
            REUSE_MODES,
            CONCURRENCY_LEVELS,
            IDLE_GAPS_MINUTES,
            OPENROUTER_ROUTING_MODES,
        )
    ]


@dataclass(frozen=True)
class VLLMBenchmarkScenario:
    cache_state: str
    concurrency: int
    deployment: str
    trials: int = 20

    @property
    def execution(self) -> str:
        return "sequential" if self.concurrency == 1 else "concurrent"

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "benchmark_mode": (
                "baseline" if self.cache_state == "disabled" else "both"
            ),
            "execution": self.execution,
            "provider": "vllm",
        }


def required_vllm_scenarios(*, trials: int = 20) -> list[VLLMBenchmarkScenario]:
    """Cold/warm/disabled APC matrix for one server and a routed fleet."""

    return [
        VLLMBenchmarkScenario(*values, trials=trials)
        for values in product(
            VLLM_CACHE_STATES,
            CONCURRENCY_LEVELS,
            VLLM_DEPLOYMENTS,
        )
    ]


def cold_trial_marker(trial_id: str) -> str:
    """Return a neutral marker used only to force a cold benchmark prefix."""

    digest = hashlib.sha256(str(trial_id).encode("utf-8")).hexdigest()[:20]
    return f"[MLEVOLVE_CONTEXT_CACHE_COLD_TRIAL:{digest}]"


def inject_cold_marker(
    messages: list[Mapping[str, Any]], trial_id: str
) -> list[dict[str, Any]]:
    """Place the marker before the would-be stable prefix; never use in production."""

    return [
        {"role": "system", "content": cold_trial_marker(trial_id)},
        *[dict(message) for message in messages],
    ]

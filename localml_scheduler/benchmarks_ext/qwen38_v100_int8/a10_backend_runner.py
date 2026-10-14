"""Commands and result selection for the four-A10 serving comparison."""

from __future__ import annotations

import math
from collections.abc import Mapping
import json
from pathlib import Path


TP_SIZE = 4


def vllm_command(model_path: str, *, port: int) -> list[str]:
    """Build the vLLM server command for the exact W8A16 MTP checkpoint."""
    return [
        "vllm",
        "serve",
        model_path,
        "--host=127.0.0.1",
        f"--port={port}",
        "--served-model-name=qwen3.8-27b-int8-w8a16-a10-tp4",
        f"--tensor-parallel-size={TP_SIZE}",
        "--gpu-memory-utilization=0.90",
        "--max-model-len=4096",
        "--language-model-only",
        "--skip-mm-profiling",
        "--mamba-cache-mode=align",
        "--speculative-config={\"method\":\"mtp\",\"num_speculative_tokens\":3}",
    ]


def trtllm_command(model_path: str, *, port: int) -> list[str]:
    """Build the TensorRT-LLM PyTorch-backend command for the same checkpoint."""
    return [
        "trtllm-serve",
        model_path,
        "--backend=pytorch",
        "--host=127.0.0.1",
        f"--port={port}",
        f"--tp_size={TP_SIZE}",
        "--max_seq_len=4096",
        "--trust_remote_code",
        "--reasoning_parser=qwen3",
    ]


def select_fastest(results: Mapping[str, Mapping[str, object]]) -> str:
    """Return the highest-throughput successful backend, rejecting bad metrics."""
    candidates: list[tuple[float, str]] = []
    for backend, result in results.items():
        throughput = result.get("tokens_per_second")
        if result.get("ok") is True and isinstance(throughput, (int, float)):
            if math.isfinite(throughput) and throughput > 0:
                candidates.append((float(throughput), backend))
    if not candidates:
        raise RuntimeError("Neither vLLM nor TensorRT-LLM produced a valid benchmark.")
    return max(candidates)[1]


def load_status_results(directory: Path) -> dict[str, dict[str, object]]:
    """Load per-backend status records using their backend names as keys."""
    return {
        path.name.removesuffix(".status.json"): json.loads(path.read_text())
        for path in directory.glob("*.status.json")
    }

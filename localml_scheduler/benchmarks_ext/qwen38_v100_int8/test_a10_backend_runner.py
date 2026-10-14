"""Contract tests for the four-A10 backend comparison commands."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_vllm_command_uses_all_four_a10s_and_mtp() -> None:
    from a10_backend_runner import vllm_command

    command = vllm_command("/models/qwen", port=8000)

    assert command[:3] == ["vllm", "serve", "/models/qwen"]
    assert "--tensor-parallel-size=4" in command
    assert "--speculative-config={\"method\":\"mtp\",\"num_speculative_tokens\":3}" in command
    assert "--port=8000" in command


def test_tensorrt_llm_command_uses_same_model_and_parallelism() -> None:
    from a10_backend_runner import trtllm_command

    command = trtllm_command("/models/qwen", port=8001)

    assert command[:2] == ["trtllm-serve", "/models/qwen"]
    assert "--tp_size=4" in command
    assert "--port=8001" in command
    assert "--backend=pytorch" in command


def test_select_fastest_ignores_failed_backend() -> None:
    from a10_backend_runner import select_fastest

    winner = select_fastest(
        {
            "vllm": {"ok": True, "tokens_per_second": 42.0},
            "tensorrt_llm": {"ok": False, "tokens_per_second": 999.0},
        }
    )

    assert winner == "vllm"


def test_select_fastest_uses_higher_successful_throughput() -> None:
    from a10_backend_runner import select_fastest

    winner = select_fastest(
        {
            "vllm": {"ok": True, "tokens_per_second": 42.0},
            "tensorrt_llm": {"ok": True, "tokens_per_second": 49.0},
        }
    )

    assert winner == "tensorrt_llm"


def test_load_status_results_removes_status_suffix(tmp_path) -> None:
    import json

    from a10_backend_runner import load_status_results

    (tmp_path / "vllm.status.json").write_text(
        json.dumps({"ok": True, "tokens_per_second": 42.0})
    )

    assert load_status_results(tmp_path) == {
        "vllm": {"ok": True, "tokens_per_second": 42.0}
    }

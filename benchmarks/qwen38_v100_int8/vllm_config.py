"""V100-safe vLLM command construction."""

from __future__ import annotations


EXACT_INT8_MTP_MODEL = "lued/Qwen3.8-27B-INT8-W8A16-MTP"


def build_vllm_server_command(model_path: str) -> list[str]:
    """Return command for two NVLink-connected V100 GPUs, reserving GPU 2 and 3."""
    return [
        "CUDA_VISIBLE_DEVICES=0,1",
        "vllm",
        "serve",
        model_path,
        "--tensor-parallel-size=2",
        "--max-model-len=4096",
        "--gpu-memory-utilization=0.92",
        "--enable-prefix-caching",
    ]


def build_exact_int8_mtp_command() -> list[str]:
    """Return the exact three-V100 Qwen3.8 27B INT8 MTP vLLM command."""
    return [
        "CUDA_VISIBLE_DEVICES=0,1,2",
        "vllm",
        "serve",
        EXACT_INT8_MTP_MODEL,
        "--served-model-name=qwen3.8-27b-int8-w8a16",
        "--tensor-parallel-size=1",
        "--pipeline-parallel-size=3",
        "--max-model-len=4096",
        "--gpu-memory-utilization=0.92",
        "--mamba-cache-mode=align",
        "--speculative-config={\"method\":\"mtp\",\"num_speculative_tokens\":3}",
        "--default-chat-template-kwargs={\"enable_thinking\":true,\"preserve_thinking\":true}",
    ]


def visible_device_index(visible_devices: str) -> int:
    """Map a CUDA_VISIBLE_DEVICES selection to its local first-device index."""
    if not visible_devices.strip():
        raise ValueError("CUDA_VISIBLE_DEVICES must select at least one device")
    return 0

"""Contract tests for V100 vLLM serving configuration."""


def test_vllm_command_uses_two_nvlink_v100_cards_and_31gib_safe_limit() -> None:
    from benchmarks.qwen38_v100_int8.vllm_config import build_vllm_server_command

    command = build_vllm_server_command("/models/Qwen3.8-27B")

    assert command[:2] == ["CUDA_VISIBLE_DEVICES=0,1", "vllm"]
    assert "--tensor-parallel-size=2" in command
    assert "--max-model-len=4096" in command
    assert "--gpu-memory-utilization=0.92" in command
    assert "--enable-prefix-caching" in command


def test_single_visible_gpu_is_addressed_as_zero() -> None:
    from benchmarks.qwen38_v100_int8.vllm_config import visible_device_index

    assert visible_device_index("3") == 0


def test_exact_int8_mtp_command_uses_three_v100_devices() -> None:
    from benchmarks.qwen38_v100_int8.vllm_config import build_exact_int8_mtp_command

    command = build_exact_int8_mtp_command()

    assert command[:2] == ["CUDA_VISIBLE_DEVICES=0,1,3", "vllm"]
    assert "lued/Qwen3.8-27B-INT8-W8A16-MTP" in command
    assert "--tensor-parallel-size=1" in command
    assert "--pipeline-parallel-size=3" in command
    assert "--mamba-cache-mode=align" in command
    assert "--speculative-config={\"method\":\"mtp\",\"num_speculative_tokens\":3}" in command

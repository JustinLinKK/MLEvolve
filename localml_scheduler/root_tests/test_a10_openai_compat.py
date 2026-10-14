import json
import types
from functools import wraps

from deployments.serve_qwen38_a10_transformers import (
    build_max_memory,
    encode_sse,
    extract_tool_calls,
    unwrap_compressed_tensors_forward,
)


def test_extract_tool_calls_converts_qwen_markup_to_openai_shape():
    content, calls = extract_tool_calls(
        'analysis before<tool_call>{"name":"submit","arguments":{"score":17.2}}</tool_call>'
    )

    assert content == "analysis before"
    assert calls == [
        {
            "id": "call_0",
            "type": "function",
            "function": {
                "name": "submit",
                "arguments": '{"score": 17.2}',
            },
        }
    ]


def test_extract_tool_calls_leaves_plain_text_unchanged():
    content, calls = extract_tool_calls("plain answer")

    assert content == "plain answer"
    assert calls == []


def test_encode_sse_emits_openai_data_frame():
    payload = {"choices": [{"delta": {"content": "ok"}}]}

    assert encode_sse(payload) == f"data: {json.dumps(payload)}\n\n"


def test_three_gpu_memory_map_leaves_headroom_on_each_a10():
    assert build_max_memory(3) == {
        0: "21GiB",
        1: "21GiB",
        2: "21GiB",
        "cpu": "6GiB",
    }


def test_unwrap_compressed_forward_hidden_behind_accelerate_hook():
    class Module:
        def forward(self, value):
            return value

    module = Module()

    @wraps(Module.forward)
    def compressed_forward(self, value):
        return self.forward(value)

    compressed_forward.__module__ = "compressed_tensors.quantization.lifecycle.forward"
    module._old_forward = types.MethodType(compressed_forward, module)
    module.forward = types.MethodType(compressed_forward, module)
    unwrap_compressed_tensors_forward(module)

    assert module._old_forward.__func__ is Module.forward
    assert module.forward.__func__ is Module.forward

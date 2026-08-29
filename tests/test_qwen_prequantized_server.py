"""Regression tests for the pre-quantized Qwen serving path."""

from __future__ import annotations

import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path


def load_server_module():
    fastapi = types.ModuleType("fastapi")

    class FastAPI:
        def on_event(self, _event):
            return lambda function: function

        def get(self, _path):
            return lambda function: function

        def post(self, _path):
            return lambda function: function

    fastapi.FastAPI = FastAPI
    responses = types.ModuleType("fastapi.responses")
    responses.JSONResponse = object
    responses.StreamingResponse = object
    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = object
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = object
    transformers.AutoTokenizer = object
    transformers.BitsAndBytesConfig = object
    transformers.TextIteratorStreamer = object
    monkey_modules = {
        "fastapi": fastapi,
        "fastapi.responses": responses,
        "pydantic": pydantic,
        "transformers": transformers,
    }
    previous_modules = {name: sys.modules.get(name) for name in monkey_modules}
    sys.modules.update(monkey_modules)
    server_path = (
        Path(__file__).parents[1] / "benchmarks/qwen38_v100_int8/server.py"
    )
    spec = importlib.util.spec_from_file_location("qwen_server", server_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                del sys.modules[name]
            else:
                sys.modules[name] = previous
    return module


def test_prequantized_path_releases_unused_cache_on_each_gpu(monkeypatch):
    """First generation must not retain allocator fragments on either A10."""
    module = load_server_module()
    released: list[int] = []
    current: list[int] = []

    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "device_count", lambda: 2)

    @contextmanager
    def device(index: int):
        current.append(index)
        yield

    monkeypatch.setattr(module.torch.cuda, "device", device)
    monkeypatch.setattr(module.torch.cuda, "empty_cache", lambda: released.append(current[-1]))

    module.release_unused_cuda_cache()

    assert released == [0, 1]


def test_prequantized_model_reserves_gpu_zero_for_unpacking(monkeypatch):
    """Compressed-tensor first use needs spare device-zero memory to unpack."""
    module = load_server_module()
    captured = {}

    class Tokenizer:
        @staticmethod
        def from_pretrained(_path):
            return object()

    class Model:
        def eval(self):
            return self

    class ModelLoader:
        @staticmethod
        def from_pretrained(_path, **kwargs):
            captured.update(kwargs)
            return Model()

    monkeypatch.setattr(module, "AutoTokenizer", Tokenizer)
    monkeypatch.setattr(module, "AutoModelForCausalLM", ModelLoader)
    monkeypatch.setattr(module, "QUANTIZATION", "prequantized")
    monkeypatch.setattr(module, "GPU_COUNT", 2)

    module.load_model()

    assert captured["device_map"] == "balanced_low_0"

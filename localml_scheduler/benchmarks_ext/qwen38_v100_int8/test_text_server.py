"""Regression tests for the text-only local Qwen service."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from types import ModuleType

import torch


SERVER_PATH = Path(__file__).with_name("server.py")


def load_server_module():
    fastapi = ModuleType("fastapi")

    class FakeFastAPI:
        def on_event(self, _name):
            return lambda function: function

        get = on_event
        post = on_event

    responses = ModuleType("fastapi.responses")
    responses.JSONResponse = object
    responses.StreamingResponse = object
    fastapi.FastAPI = FakeFastAPI
    sys.modules["fastapi"] = fastapi
    sys.modules["fastapi.responses"] = responses

    transformers = ModuleType("transformers")

    class VisionProcessor:
        @classmethod
        def from_pretrained(cls, _model_path: str):
            return "vision-processor"

    class VisionModel:
        @classmethod
        def from_pretrained(cls, _model_path: str, **_kwargs):
            return SimpleNamespace(eval=lambda: "vision-model")

    class TextTokenizer:
        @classmethod
        def from_pretrained(cls, _model_path: str):
            return "unpatched-text-tokenizer"

    class TextModel:
        @classmethod
        def from_pretrained(cls, _model_path: str, **_kwargs):
            return SimpleNamespace(eval=lambda: "unpatched-text-model")

    transformers.AutoProcessor = VisionProcessor
    transformers.AutoModelForImageTextToText = VisionModel
    transformers.AutoTokenizer = TextTokenizer
    transformers.AutoModelForCausalLM = TextModel
    transformers.BitsAndBytesConfig = lambda **_kwargs: object()
    transformers.TextIteratorStreamer = object
    sys.modules["transformers"] = transformers
    spec = importlib.util.spec_from_file_location("qwen_text_server", SERVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_startup_loads_text_tokenizer_and_causal_language_model(monkeypatch) -> None:
    """Replacing the causal-Language-Model loader with a vision loader must fail."""
    server = load_server_module()
    loaded: dict[str, object] = {}

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_path: str):
            loaded["tokenizer_path"] = model_path
            return "text-tokenizer"

    class FakeCausalLanguageModel:
        @classmethod
        def from_pretrained(cls, model_path: str, **kwargs):
            loaded["model_path"] = model_path
            loaded["model_kwargs"] = kwargs
            return SimpleNamespace(eval=lambda: "text-model")

    monkeypatch.setattr(server, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(server, "AutoModelForCausalLM", FakeCausalLanguageModel)
    monkeypatch.setattr(server, "QUANTIZATION", "fp16")

    server.load_model()

    assert server.tokenizer == "text-tokenizer"
    assert server.model == "text-model"
    assert loaded["tokenizer_path"] == server.MODEL_PATH
    assert loaded["model_path"] == server.MODEL_PATH


def test_prompt_inputs_applies_text_chat_template_without_a_processor() -> None:
    """A text-only request must be tokenized directly from its chat template."""
    server = load_server_module()
    seen: dict[str, object] = {}

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            seen["messages"] = messages
            seen["template_kwargs"] = kwargs
            return "user: say hi\nassistant:"

        def __call__(self, text, **kwargs):
            seen["text"] = text
            seen["tokenize_kwargs"] = kwargs
            return {"input_ids": torch.tensor([[1, 2, 3]])}

    server.tokenizer = FakeTokenizer()
    server.model = SimpleNamespace(parameters=lambda: iter([torch.empty(1)]))

    inputs = server.prompt_inputs([{"role": "user", "content": "say hi"}])

    assert seen["messages"] == [{"role": "user", "content": "say hi"}]
    assert seen["template_kwargs"] == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    assert seen["text"] == "user: say hi\nassistant:"
    assert seen["tokenize_kwargs"] == {"return_tensors": "pt"}
    assert inputs["input_ids"].tolist() == [[1, 2, 3]]

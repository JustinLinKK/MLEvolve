"""Regression tests for embedding-device isolation."""

from __future__ import annotations

import sys
import types

from agents.memory.embedding_models import EmbeddingModel


def test_local_embedding_constructor_receives_requested_cpu_device(monkeypatch):
    """CPU retrieval must not first create a CUDA context in SentenceTransformer."""

    received: dict[str, object] = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, *, device: str) -> None:
            received["model_name"] = model_name
            received["device"] = device

        def to(self, device: str):
            received["to_device"] = device
            return self

        def get_sentence_embedding_dimension(self) -> int:
            return 768

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    embedding = EmbeddingModel(
        model_type="local", model_name="BAAI/bge-base-en-v1.5", device="cpu"
    )

    assert embedding.dimension == 768
    assert received == {
        "model_name": "BAAI/bge-base-en-v1.5",
        "device": "cpu",
        "to_device": "cpu",
    }

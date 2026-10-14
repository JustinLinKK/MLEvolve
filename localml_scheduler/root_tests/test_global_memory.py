from __future__ import annotations

import importlib
import sys
import types


def test_global_memory_defaults_embeddings_to_cpu(monkeypatch, tmp_path) -> None:
    received: dict[str, object] = {}

    class FakeEmbeddingModel:
        def __init__(self, **kwargs) -> None:
            received.update(kwargs)

    class FakeRetriever:
        def __init__(self, embedding_model) -> None:
            self.embedding_model = embedding_model

    monkeypatch.setitem(
        sys.modules,
        "agents.memory.embedding_models",
        types.SimpleNamespace(EmbeddingModel=FakeEmbeddingModel),
    )
    monkeypatch.setitem(
        sys.modules,
        "agents.memory.retriever",
        types.SimpleNamespace(HybridRetriever=FakeRetriever),
    )
    sys.modules.pop("agents.memory.global_memory", None)
    global_memory = importlib.import_module("agents.memory.global_memory")

    global_memory.GlobalMemoryLayer(str(tmp_path), embedding_model_path="bge")

    assert received["device"] == "cpu"

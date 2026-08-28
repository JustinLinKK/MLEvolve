from hardware_knowledge_graph.client import HardwareKnowledgeClient
from hardware_knowledge_graph.config import HardwareKnowledgeSettings


def test_hardware_knowledge_client_initializes_without_redis_cache(tmp_path) -> None:
    """HWKD must initialize its prompt/evidence layer when Redis is disabled."""
    settings = HardwareKnowledgeSettings.from_dict(
        {
            "runtime_root": str(tmp_path / "hardware-knowledge"),
            "redis_cache": {"enabled": False},
        }
    )

    client = HardwareKnowledgeClient(settings)

    assert client.knowledge.store is client.store

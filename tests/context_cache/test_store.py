from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import pytest

from context_cache.compiler import KnowledgePackCompiler, PackManifest
from context_cache.models import PackBuild
from context_cache.store import KnowledgePackStore, PackStoreError


def _compile_in_subprocess(cache_dir: str, counter_path: str) -> None:
    def resolver(source):
        del source
        with Path(counter_path).open("a", encoding="utf-8") as handle:
            handle.write("built\n")
        return [{"stable_id": "hardware-a", "value": 1}]

    store = KnowledgePackStore(cache_dir)
    compiler = KnowledgePackCompiler(store, source_resolvers={"qdrant": resolver})
    compiler.compile(
        PackManifest(
            role="reviewer",
            schema_version="1",
            knowledge_version="concurrent-k1",
            content={"sections": []},
            sources=({"kind": "qdrant", "snapshot": "hardware-k1", "required": True},),
        )
    )


def _manifest(*, version: str = "k1", value: str = "stable") -> PackManifest:
    return PackManifest(
        role="reviewer",
        schema_version="1",
        knowledge_version=version,
        content={"sections": [{"stable_id": "rules", "content": value}]},
        sources=({"kind": "manifest", "snapshot": version},),
    )


def test_repeated_load_does_not_query_source_again(tmp_path: Path) -> None:
    calls = 0

    def resolver(source):
        nonlocal calls
        calls += 1
        return [{"stable_id": source["snapshot"], "value": 1}]

    store = KnowledgePackStore(tmp_path)
    compiler = KnowledgePackCompiler(store, source_resolvers={"neo4j": resolver})
    manifest = PackManifest(
        role="reviewer",
        schema_version="1",
        knowledge_version="k1",
        content={"sections": []},
        sources=({"kind": "neo4j", "snapshot": "profiles-k1", "required": True},),
    )

    first = compiler.compile(manifest)
    second = compiler.compile(manifest)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert calls == 1
    assert first.ref.content_sha256 == second.ref.content_sha256
    assert "compiled_at" in json.loads(Path(first.ref.path).read_text(encoding="utf-8"))


def test_reordered_source_results_produce_the_same_semantic_hash(
    tmp_path: Path,
) -> None:
    records = [
        {"stable_id": "b", "value": 2, "similarity_score": 0.8},
        {"stable_id": "a", "value": 1, "similarity_score": 0.9},
    ]

    def resolver(source):
        return list(reversed(records)) if source["snapshot"] == "reverse" else records

    compiler = KnowledgePackCompiler(
        KnowledgePackStore(tmp_path), source_resolvers={"qdrant": resolver}
    )
    first = compiler.compile(
        PackManifest(
            "reviewer",
            "1",
            "ordered-k1",
            {"sections": []},
            (
                {
                    "kind": "qdrant",
                    "snapshot": "ordered",
                    "stable_id": "hardware-records",
                    "required": True,
                },
            ),
        )
    )
    second = compiler.compile(
        PackManifest(
            "reviewer",
            "1",
            "reverse-k1",
            {"sections": []},
            (
                {
                    "kind": "qdrant",
                    "snapshot": "reverse",
                    "stable_id": "hardware-records",
                    "required": True,
                },
            ),
        )
    )

    assert first.ref.content_sha256 == second.ref.content_sha256


def test_missing_or_corrupt_object_is_rebuilt_atomically(tmp_path: Path) -> None:
    first_store = KnowledgePackStore(tmp_path)
    first = KnowledgePackCompiler(first_store).compile(_manifest())
    Path(first.ref.path).write_text("{corrupt", encoding="utf-8")

    second_store = KnowledgePackStore(tmp_path)
    rebuilt = KnowledgePackCompiler(second_store).compile(_manifest())

    assert rebuilt.cache_hit is False
    assert second_store.verify()[0]["valid"] is True
    assert (
        json.loads(Path(rebuilt.ref.path).read_text(encoding="utf-8"))["content_sha256"]
        == rebuilt.ref.content_sha256
    )


def test_knowledge_version_alias_is_immutable(tmp_path: Path) -> None:
    compiler = KnowledgePackCompiler(KnowledgePackStore(tmp_path))
    compiler.compile(_manifest(value="first"))

    # Existing aliases are returned without rebuilding, preventing mutation in place.
    existing = compiler.compile(_manifest(value="second"))

    assert existing.envelope["content"]["sections"][0]["content"] == "first"


def test_run_freezes_role_reference_across_alias_versions(tmp_path: Path) -> None:
    store = KnowledgePackStore(tmp_path)
    compiler = KnowledgePackCompiler(store)
    first = compiler.compile(_manifest(version="k1", value="one"))
    second = compiler.compile(_manifest(version="k2", value="two"))

    assert store.freeze("run-a", first.ref) == first.ref
    assert store.freeze("run-a", second.ref).content_sha256 == first.ref.content_sha256


def test_multiple_subprocesses_compile_one_object_and_invoke_source_once(
    tmp_path: Path,
) -> None:
    counter = tmp_path / "resolver-calls.txt"
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_compile_in_subprocess, args=(str(tmp_path / "cache"), str(counter))
        )
        for _ in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(30)

    assert [process.exitcode for process in processes] == [0, 0, 0, 0]
    assert counter.read_text(encoding="utf-8").splitlines() == ["built"]
    objects = list((tmp_path / "cache" / "objects").glob("*.json"))
    assert len(objects) == 1
    assert not list((tmp_path / "cache" / "objects").glob("*.tmp"))


def test_cleanup_is_dry_run_by_default(tmp_path: Path) -> None:
    store = KnowledgePackStore(tmp_path)
    orphan = store.object_path("a" * 64)
    orphan.write_text("{}", encoding="utf-8")
    with store._connect() as connection:
        connection.execute(
            """INSERT INTO pack_objects VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("a" * 64, "reviewer", "1", "old", str(orphan), "[]", 2, "now", "now"),
        )

    assert store.cleanup() == [str(orphan)]
    assert orphan.exists()
    assert store.cleanup(dry_run=False) == [str(orphan)]
    assert not orphan.exists()


def test_sensitive_fields_are_rejected(tmp_path: Path) -> None:
    compiler = KnowledgePackCompiler(KnowledgePackStore(tmp_path))
    manifest = PackManifest(
        role="reviewer",
        schema_version="1",
        knowledge_version="secret-k1",
        content={
            "sections": [{"stable_id": "bad", "content": {"api_key": "do-not-store"}}]
        },
    )

    with pytest.raises(PackStoreError, match="sensitive fields"):
        compiler.compile(manifest)

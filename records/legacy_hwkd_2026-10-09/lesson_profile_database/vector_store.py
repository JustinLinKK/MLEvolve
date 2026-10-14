"""Derived Qdrant index for active lesson-profile records."""

from __future__ import annotations

import os
import uuid
from typing import Any, Mapping

import numpy as np

from .config import LessonProfileSettings


_PAYLOAD_INDEXES: dict[str, str] = {
    "profile_key": "keyword",
    "model_family": "keyword",
    "architecture_type": "keyword",
    "hardware_key": "keyword",
    "accelerator_key": "keyword",
    "resource_slice_key": "keyword",
    "runtime_class": "keyword",
    "framework_major": "keyword",
    "cuda_major": "keyword",
    "backend_class": "keyword",
    "workload_bucket": "keyword",
    "record_kind": "keyword",
    "lesson_type": "keyword",
    "agent_audiences": "keyword",
    "layer_type": "keyword",
    "change_action": "keyword",
    "status": "keyword",
    "maturity": "keyword",
    "confidence_band": "keyword",
    "active": "bool",
    "revision": "integer",
    "confidence": "float",
}


class LessonVectorStore:
    def __init__(
        self,
        settings: LessonProfileSettings,
        *,
        qdrant_client: Any | None = None,
        qdrant_models: Any | None = None,
        embedding_model: Any | None = None,
    ):
        self.settings = settings
        self.config = settings.qdrant
        self._client = qdrant_client
        self._models = qdrant_models
        self._embedding_model = embedding_model

    @property
    def enabled(self) -> bool:
        return bool(self.settings.enabled and self.config.enabled)

    def _qdrant_models(self) -> Any:
        if self._models is None:
            from qdrant_client import models

            self._models = models
        return self._models

    def _qdrant_client(self) -> Any:
        if self._client is None:
            from qdrant_client import QdrantClient

            api_key = os.getenv(self.config.api_key_env) if self.config.api_key_env else None
            self._client = QdrantClient(url=self.config.url, api_key=api_key, timeout=3)
        return self._client

    def _embedder(self) -> Any:
        if self._embedding_model is None:
            from agents.memory.embedding_models import EmbeddingModel

            self._embedding_model = EmbeddingModel(
                model_type=self.config.embedding_model_type,
                model_name=self.config.embedding_model_name,
                dimension=self.config.embedding_dimension,
                device=self.config.embedding_device,
            )
        return self._embedding_model

    def _encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._embedder().encode(texts, show_progress_bar=False), dtype=np.float32)

    def _dimension(self) -> int:
        if self.config.embedding_dimension:
            return int(self.config.embedding_dimension)
        return int(getattr(self._embedder(), "dimension"))

    def _distance(self) -> Any:
        models = self._qdrant_models()
        value = str(self.config.distance or "Cosine").upper()
        return getattr(models.Distance, value, models.Distance.COSINE)

    def _collection_exists(self) -> bool:
        client = self._qdrant_client()
        if hasattr(client, "collection_exists"):
            return bool(client.collection_exists(self.config.collection_name))
        try:
            client.get_collection(self.config.collection_name)
            return True
        except Exception:
            return False

    def ensure_collection(self, *, recreate: bool = False) -> dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "reason": "Qdrant disabled"}
        client = self._qdrant_client()
        models = self._qdrant_models()
        if recreate and self._collection_exists():
            client.delete_collection(self.config.collection_name)
        created = False
        if recreate or not self._collection_exists():
            client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=models.VectorParams(size=self._dimension(), distance=self._distance()),
            )
            created = True
        indexed: list[str] = []
        for field_name, kind in _PAYLOAD_INDEXES.items():
            schema = getattr(models.PayloadSchemaType, kind.upper())
            try:
                client.create_payload_index(
                    collection_name=self.config.collection_name,
                    field_name=field_name,
                    field_schema=schema,
                    wait=True,
                )
                indexed.append(field_name)
            except Exception as exc:
                # Qdrant reports an existing index as an error in some versions.
                if "already" not in str(exc).lower() and "exists" not in str(exc).lower():
                    raise
        return {
            "ok": True,
            "collection": self.config.collection_name,
            "created": created,
            "dimension": self._dimension(),
            "payload_indexes": indexed,
        }

    @staticmethod
    def point_id(profile_key: str, revision: int, record_kind: str, record_id: str) -> str:
        return str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"mlevolve-lesson-profile:{profile_key}:{revision}:{record_kind}:{record_id}",
        ))

    @staticmethod
    def _confidence_band(confidence: float) -> str:
        if confidence >= 0.8:
            return "high"
        if confidence >= 0.5:
            return "medium"
        return "low"

    @staticmethod
    def _search_text(payload: Mapping[str, Any]) -> str:
        content = payload.get("content") or payload.get("baseline") or {}
        parts = [
            str(payload.get("lesson_type") or payload.get("record_kind") or ""),
            str(content.get("summary") if isinstance(content, Mapping) else content),
            str(content.get("lesson") if isinstance(content, Mapping) else ""),
            " ".join(str(item) for item in payload.get("warnings") or []),
            str(payload.get("change_signature") or ""),
        ]
        return "\n".join(part for part in parts if part and part != "None")[:6000]

    def publication_records(self, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
        identity = dict(payload["identity"])
        profile_key = str(payload["profile_key"])
        revision = int(payload["revision_number"])
        trust = dict(payload.get("trust") or {})
        confidence = float(trust.get("confidence") or 0.0)
        common = {
            **identity,
            "profile_key": profile_key,
            "revision": revision,
            "maturity": str(payload.get("maturity") or "provisional"),
            "status": "active",
            "active": True,
        }
        records = [{
            **common,
            "record_id": f"baseline:{profile_key}:{revision}",
            "record_kind": "baseline",
            "lesson_type": "family_baseline",
            "agent_audiences": ["draft", "improve", "evolution", "fusion", "aggregation", "review"],
            "baseline": dict(payload.get("baseline") or {}),
            "confidence": confidence,
            "confidence_band": self._confidence_band(confidence),
            "evidence_refs": list(trust.get("evidence_refs") or []),
            "warnings": list((payload.get("baseline") or {}).get("warnings") or []),
        }]
        for lesson in payload.get("lessons") or []:
            lesson = dict(lesson)
            lesson_confidence = float(lesson.get("confidence") or 0.0)
            records.append({
                **common,
                "record_id": str(lesson["lesson_id"]),
                "record_kind": "lesson",
                "lesson_type": str(lesson.get("lesson_type") or "modification"),
                "agent_audiences": list(lesson.get("agent_audiences") or []),
                "content": dict(lesson.get("content") or {}),
                "change_signature": str(lesson.get("change_signature") or ""),
                "change_scope": str(lesson.get("change_scope") or "training_only"),
                "change_action": str(lesson.get("change_action") or "other"),
                "layer_type": str(lesson.get("layer_type") or "other"),
                "confidence": lesson_confidence,
                "confidence_band": self._confidence_band(lesson_confidence),
                "evidence_refs": list(lesson.get("evidence_refs") or []),
                "warnings": list(lesson.get("warnings") or []),
            })
        for record in records:
            record["search_text"] = self._search_text(record)
        return records

    def upsert_publication(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return {"ok": True, "skipped": True, "reason": "Qdrant disabled"}
        self.ensure_collection()
        records = self.publication_records(payload)
        vectors = self._encode([str(record["search_text"]) for record in records])
        models = self._qdrant_models()
        client = self._qdrant_client()
        if hasattr(client, "set_payload"):
            client.set_payload(
                collection_name=self.config.collection_name,
                payload={"active": False, "status": "superseded"},
                points=models.Filter(must=[self._condition("profile_key", str(payload["profile_key"]))]),
                wait=True,
            )
        points = []
        for index, record in enumerate(records):
            points.append(models.PointStruct(
                id=self.point_id(
                    str(record["profile_key"]),
                    int(record["revision"]),
                    str(record["record_kind"]),
                    str(record["record_id"]),
                ),
                vector=vectors[index].tolist(),
                payload=record,
            ))
        client.upsert(
            collection_name=self.config.collection_name,
            points=points,
            wait=True,
        )
        return {"ok": True, "record_count": len(records), "collection": self.config.collection_name}

    def _condition(self, key: str, value: Any) -> Any:
        models = self._qdrant_models()
        if isinstance(value, bool):
            return models.FieldCondition(key=key, match=models.MatchValue(value=value))
        if isinstance(value, (list, tuple, set)):
            if hasattr(models, "MatchAny"):
                return models.FieldCondition(key=key, match=models.MatchAny(any=list(value)))
            value = next(iter(value), "")
        return models.FieldCondition(key=key, match=models.MatchValue(value=value))

    def _filter(self, filters: Mapping[str, Any]) -> Any:
        must = [self._condition(key, value) for key, value in filters.items() if value is not None]
        return self._qdrant_models().Filter(must=must)

    def search(
        self,
        *,
        query: str,
        filters: Mapping[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        if not self.enabled or not self._collection_exists():
            return []
        vector = self._encode([query])[0].tolist()
        client = self._qdrant_client()
        query_filter = self._filter(filters)
        if hasattr(client, "query_points"):
            response = client.query_points(
                collection_name=self.config.collection_name,
                query=vector,
                query_filter=query_filter,
                limit=max(1, int(limit)),
                with_payload=True,
            )
            points = list(getattr(response, "points", response) or [])
        else:
            points = list(client.search(
                collection_name=self.config.collection_name,
                query_vector=vector,
                query_filter=query_filter,
                limit=max(1, int(limit)),
                with_payload=True,
            ) or [])
        results = []
        for point in points:
            item = dict(getattr(point, "payload", None) or (point.get("payload") if isinstance(point, dict) else {}) or {})
            score = getattr(point, "score", None)
            if score is None and isinstance(point, dict):
                score = point.get("score")
            item["score"] = float(score or 0.0)
            item.pop("search_text", None)
            results.append(item)
        return results

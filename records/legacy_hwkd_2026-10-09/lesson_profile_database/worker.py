"""Durable, non-blocking lesson profile builder worker."""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Callable

from .builder import LessonBuilder
from .config import LessonProfileSettings
from .registry import LessonProfileRegistry
from .vector_store import LessonVectorStore


LOGGER = logging.getLogger("MLEvolve")


class LessonBuilderWorker:
    def __init__(
        self,
        settings: LessonProfileSettings,
        registry: LessonProfileRegistry,
        builder: LessonBuilder,
        vector_store: LessonVectorStore,
        *,
        invalidator: Callable[[str], None] | None = None,
        worker_id: str | None = None,
    ):
        self.settings = settings
        self.registry = registry
        self.builder = builder
        self.vector_store = vector_store
        self.invalidator = invalidator
        self.worker_id = worker_id or f"lesson-builder-{uuid.uuid4().hex[:12]}"
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    @property
    def running(self) -> bool:
        return any(thread.is_alive() for thread in self._threads)

    def start(self) -> "LessonBuilderWorker":
        if self.running or not self.settings.builder.enabled:
            return self
        self.registry.initialize()
        self._stop.clear()
        concurrency = max(1, int(self.settings.builder.concurrency))
        for index in range(concurrency):
            thread = threading.Thread(
                target=self._run,
                name=f"{self.worker_id}-{index}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
        return self

    def stop(self, *, timeout: float = 2.0) -> None:
        """Stop without draining; queued and leased work remains durable."""

        self._stop.set()
        for thread in list(self._threads):
            thread.join(timeout=max(0.0, float(timeout)))
        self._threads = [thread for thread in self._threads if thread.is_alive()]

    def run_foreground(self, *, once: bool = False) -> None:
        self.registry.initialize()
        while not self._stop.is_set():
            worked = self.process_once()
            if once:
                return
            if not worked:
                self._stop.wait(max(0.05, float(self.settings.builder.poll_interval_seconds)))

    def _run(self) -> None:
        while not self._stop.is_set():
            worked = self.process_once()
            if not worked:
                self._stop.wait(max(0.05, float(self.settings.builder.poll_interval_seconds)))

    def process_once(self) -> bool:
        job = self.registry.lease_next_job(
            worker_id=self.worker_id,
            lease_seconds=self.settings.builder.lease_seconds,
        )
        if job is None:
            return False
        observation_id = str(job["observation_id"])
        try:
            observation = self.registry.observation(observation_id)
            if observation is None:
                raise RuntimeError(f"Missing observation {observation_id}")
            publication = self.registry.pending_publication(observation_id)
            if publication is None:
                built = self.builder.build(observation)
                if built is None:
                    self.registry.complete_job(str(job["job_id"]), observation_id=observation_id, ignored=True)
                    return True
                feedback = getattr(getattr(getattr(self.builder, "cfg", None), "agent", None), "feedback", None)
                builder_model = self.settings.builder.model or getattr(feedback, "model", None)
                publication = self.registry.prepare_revision(
                    identity=built["identity"],
                    observation_id=observation_id,
                    baseline=built["baseline"],
                    trust=built["trust"],
                    maturity=built["maturity"],
                    lessons=built["lessons"],
                    builder_model=builder_model,
                    builder_prompt_version=self.settings.builder.prompt_version,
                    extractor_version=self.settings.builder.extractor_version,
                )
            self.vector_store.upsert_publication(publication["payload"])
            activated = self.registry.activate_publication(str(publication["outbox_id"]))
            self.registry.complete_job(str(job["job_id"]), observation_id=observation_id)
            if self.invalidator is not None:
                self.invalidator(str(activated["profile_key"]))
        except Exception as exc:
            pending = self.registry.pending_publication(observation_id)
            if pending is not None:
                self.registry.record_outbox_failure(str(pending["outbox_id"]), str(exc))
            self.registry.fail_job(
                str(job["job_id"]),
                error=str(exc),
                max_retries=self.settings.builder.max_retries,
                retry_delay_seconds=self.settings.builder.retry_delay_seconds,
            )
            LOGGER.warning("Lesson profile builder job %s failed open: %s", job["job_id"], exc)
        return True

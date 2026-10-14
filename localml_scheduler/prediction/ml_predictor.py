"""Scheduler adapter for the CPU-only PerfSeer student predictor."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
import json
import logging
import math
import multiprocessing
from pathlib import Path
import sys
import time
from typing import Any, Sequence

from ..config import PredictionSettings
from ..domain import TrainingJob
from ..hardware import HardwareProfile
from .source_worker import convert_source_options_worker, convert_source_worker
from .v32_source_worker import predict_v32_source_options_worker


LOGGER = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PERFSEER_SRC = REPOSITORY_ROOT / "PerfSeer-predictor" / "src"
DEFAULT_REGISTRY = REPOSITORY_ROOT / "PerfSeer-predictor" / "models" / "registry.json"


class JobPredictionError(RuntimeError):
    """One job could not be converted or predicted and must use branch profiling."""


class JobPredictionPending(JobPredictionError):
    """Source capture is running; the scheduler must retry on a later tick."""


@dataclass
class _PendingV32Capture:
    receiver: Any
    process: Any
    keys: tuple[str, ...]
    deadline: float


@dataclass(frozen=True)
class ModelSpecification:
    source_path: Path
    working_directory: Path
    entry: str
    input_shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[str, ...]
    precision: str
    constructor_args: tuple[Any, ...]
    constructor_kwargs: dict[str, Any]
    steps_per_epoch: int | None = None

    def worker_request(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "working_directory": str(self.working_directory),
            "entry": self.entry,
            "input_shapes": [list(shape) for shape in self.input_shapes],
            "input_dtypes": list(self.input_dtypes),
            "precision": self.precision,
            "constructor_args": list(self.constructor_args),
            "constructor_kwargs": self.constructor_kwargs,
            "steps_per_epoch": self.steps_per_epoch,
        }


def _normalize_precision(value: object) -> str:
    aliases = {
        "fp32": "fp32_ieee",
        "float32": "fp32_ieee",
        "fp32_ieee": "fp32_ieee",
        "tf32": "tf32",
        "bf16": "bf16_amp",
        "bfloat16": "bf16_amp",
        "bf16_amp": "bf16_amp",
        "fp16": "fp16_amp",
        "float16": "fp16_amp",
        "fp16_amp": "fp16_amp",
    }
    normalized = aliases.get(str(value or "fp32_ieee").strip().lower())
    if normalized is None:
        raise JobPredictionError(f"unsupported predictor precision {value!r}")
    return normalized


def _resolve_shapes(raw_shapes: Sequence[Sequence[Any]], batch_size: int) -> tuple[tuple[int, ...], ...]:
    resolved: list[tuple[int, ...]] = []
    for raw_shape in raw_shapes:
        dimensions: list[int] = []
        for dimension in raw_shape:
            if isinstance(dimension, str) and dimension.strip().lower() == "$batch":
                dimensions.append(int(batch_size))
            else:
                dimensions.append(int(dimension))
        if not dimensions or any(dimension <= 0 for dimension in dimensions):
            raise JobPredictionError(f"invalid predictor input shape {raw_shape!r}")
        resolved.append(tuple(dimensions))
    if not resolved:
        raise JobPredictionError("predictor input_shapes is empty")
    return tuple(resolved)


def model_specification_for_job(job: TrainingJob, batch_size: int) -> ModelSpecification:
    metadata = dict(job.metadata.get("perfseer_model") or {})
    source_value = (
        metadata.get("source_path")
        or job.metadata.get("architecture_source")
        or job.metadata.get("source_path")
        or job.baseline_model_path
    )
    if not source_value:
        raise JobPredictionError("no model source path is available")
    source_path = Path(str(source_value)).expanduser()
    if not source_path.is_absolute():
        source_path = (Path.cwd() / source_path).resolve()
    if not source_path.is_file() or source_path.suffix != ".py":
        raise JobPredictionError(f"model source is not a Python file: {source_path}")
    raw_working_directory = job.config.runner_kwargs.get("working_dir")
    working_directory = (
        Path(str(raw_working_directory)).expanduser()
        if raw_working_directory
        else source_path.parent
    )
    if not working_directory.is_absolute():
        working_directory = (Path.cwd() / working_directory).resolve()
    if not working_directory.is_dir():
        raise JobPredictionError(
            f"model execution directory is not a directory: {working_directory}"
        )

    raw_shapes = metadata.get("input_shapes")
    if raw_shapes is None:
        input_shape = metadata.get("input_shape") or job.metadata.get("input_shape")
        if input_shape is None:
            raise JobPredictionError("perfseer_model.input_shapes or metadata.input_shape is required")
        raw_shapes = [["$batch", *list(input_shape)]]
    input_shapes = _resolve_shapes(raw_shapes, batch_size)

    raw_dtypes = metadata.get("input_dtypes") or ["float32"]
    input_dtypes = tuple(str(value) for value in raw_dtypes)
    if len(input_dtypes) == 1 and len(input_shapes) > 1:
        input_dtypes = tuple(input_dtypes[0] for _ in input_shapes)
    if len(input_dtypes) != len(input_shapes):
        raise JobPredictionError("input_dtypes must contain one value or match input_shapes")

    raw_steps_per_epoch = metadata.get("steps_per_epoch") or job.metadata.get(
        "steps_per_epoch"
    )
    try:
        steps_per_epoch = int(raw_steps_per_epoch)
    except (TypeError, ValueError):
        steps_per_epoch = 0
    if steps_per_epoch <= 0:
        raw_dataset_size = metadata.get("dataset_size") or job.metadata.get(
            "dataset_size"
        )
        try:
            dataset_size = int(raw_dataset_size)
        except (TypeError, ValueError):
            dataset_size = 0
        steps_per_epoch = (
            int(math.ceil(dataset_size / max(1, int(batch_size))))
            if dataset_size > 0
            else 0
        )

    return ModelSpecification(
        source_path=source_path,
        working_directory=working_directory,
        entry=str(metadata.get("entry") or "build_model"),
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        precision=_normalize_precision(metadata.get("precision") or job.metadata.get("precision")),
        constructor_args=tuple(metadata.get("constructor_args") or ()),
        constructor_kwargs=dict(metadata.get("constructor_kwargs") or {}),
        steps_per_epoch=steps_per_epoch or None,
    )


class MLVramPredictor:
    """Hardware-selected student runtime with per-job conversion fallback signals."""

    def __init__(self, settings: PredictionSettings, hardware: HardwareProfile) -> None:
        self.settings = settings
        self.hardware = hardware
        self.available = False
        self.unavailable_reason: str | None = None
        self.model_id: str | None = None
        self.artifact_hash: str | None = None
        self._runtime: Any | None = None
        self._encoded_type: Any | None = None
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._job_values: dict[str, float] = {}
        self._job_epoch_seconds: dict[str, float] = {}
        self._v32_predictions: dict[str, dict[str, float]] = {}
        self._v32_pending_capture: _PendingV32Capture | None = None
        self._job_failures: dict[str, str] = {}
        self._target_names: tuple[str, ...] = ()
        self.last_sources: dict[str, str] = {}
        self.last_errors: dict[str, str] = {}
        self._initialize()

    def close(self) -> None:
        """Release an unfinished source-capture worker without delaying shutdown."""
        pending = self._v32_pending_capture
        self._v32_pending_capture = None
        if pending is None:
            return
        try:
            pending.receiver.close()
        finally:
            if pending.process.is_alive():
                pending.process.terminate()
            pending.process.join(timeout=0.0)

    def _initialize(self) -> None:
        try:
            if self.settings.v32_source_root:
                self._initialize_v32()
                self.available = True
                return
            if not PERFSEER_SRC.is_dir():
                raise RuntimeError(f"PerfSeer submodule source is missing: {PERFSEER_SRC}")
            if str(PERFSEER_SRC) not in sys.path:
                sys.path.insert(0, str(PERFSEER_SRC))
            from perfseer_student import (
                EncodedGraph,
                HardwareInfo,
                ModelRegistry,
                StudentRuntime,
            )

            train_mem_index = 1
            if self.settings.test_override_enabled:
                if not self.settings.test_model_path:
                    raise RuntimeError("test_override_enabled requires test_model_path")
                artifact_path = Path(self.settings.test_model_path).expanduser().resolve()
                self.model_id = "test_override"
            else:
                registry_path = Path(self.settings.registry_path).expanduser() if self.settings.registry_path else DEFAULT_REGISTRY
                record = ModelRegistry(registry_path).select(
                    HardwareInfo(
                        self.hardware.gpu_name,
                        str(self.hardware.compute_capability or ""),
                        int(self.hardware.total_vram_mb or 0),
                    )
                )
                artifact_path = record.artifact_path
                train_mem_index = record.train_mem_index
                self.model_id = record.model_id
                self._target_names = tuple(record.targets)
            if not artifact_path.is_file():
                raise RuntimeError(f"predictor artifact is missing: {artifact_path}")
            self.artifact_hash = sha256(artifact_path.read_bytes()).hexdigest()
            self._runtime = StudentRuntime(artifact_path, train_mem_index=train_mem_index)
            self._encoded_type = EncodedGraph
            self._smoke_test_runtime()
            self.available = True
        except Exception as exc:
            self.unavailable_reason = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("ML predictor unavailable; using branch profiles: %s", self.unavailable_reason)

    def _initialize_v32(self) -> None:
        """Validate the external v3.2 package before any source capture runs."""
        source_root = Path(str(self.settings.v32_source_root)).expanduser().resolve()
        artifact_path = Path(str(self.settings.v32_artifact_path)).expanduser().resolve()
        if not source_root.is_dir():
            raise RuntimeError(f"PerfSeer v3.2 source is missing: {source_root}")
        if not artifact_path.is_file():
            raise RuntimeError(f"PerfSeer v3.2 artifact is missing: {artifact_path}")
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        import torch

        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        target_names = tuple(str(name) for name in artifact.get("target_names") or ())
        required_targets = {"train_epoch_ms", "train_peak_vram_mib"}
        if artifact.get("version") != "perfseer_v32_checkpoint_v2":
            raise RuntimeError("predictor artifact is not a PerfSeer v3.2 checkpoint")
        if not required_targets.issubset(target_names):
            raise RuntimeError("v3.2 artifact lacks scheduler time or peak-VRAM targets")
        self.model_id = "perfseer_v32_student"
        self.artifact_hash = sha256(artifact_path.read_bytes()).hexdigest()
        self._target_names = target_names
        self._runtime = "v32"

    def _predict_v32_options(
        self,
        job: TrainingJob,
        specifications: list[ModelSpecification],
        keys: list[str],
    ) -> None:
        missing = [index for index, key in enumerate(keys) if key not in self._v32_predictions]
        if not missing:
            return
        if self.settings.v32_async_source_capture:
            self._poll_v32_capture()
            missing = [index for index, key in enumerate(keys) if key not in self._v32_predictions]
            if not missing:
                return
        # Once the baseline batch has an estimate, let the planner admit that
        # unchanged training configuration.  Capturing every optional batch
        # variant serially would keep the entire queue cold.
        if any(key in self._v32_predictions for key in keys):
            return
        missing = self._v32_primary_capture_indices(job, specifications, missing)
        if self.settings.v32_async_source_capture:
            if self._v32_pending_capture is not None:
                raise JobPredictionPending("v3.2 source capture is still running")
            self._start_v32_capture(specifications, keys, missing)
            raise JobPredictionPending("v3.2 source capture started")
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=predict_v32_source_options_worker,
            args=(
                sender,
                str(self.settings.v32_source_root),
                str(self.settings.v32_artifact_path),
                [specifications[index].worker_request() for index in missing],
                self.settings.v32_capture_device,
            ),
            # Candidate modules may construct DataLoader workers at import
            # time.  A daemon process cannot create those children, whereas
            # this bounded worker is joined/terminated explicitly below.
            daemon=False,
        )
        process.start()
        sender.close()
        timeout_seconds = self.settings.conversion_timeout_seconds * max(1, len(missing))
        try:
            if not receiver.poll(timeout_seconds):
                process.terminate()
                process.join(timeout=2.0)
                raise JobPredictionError(f"v3.2 source capture exceeded {timeout_seconds:.1f}s")
            response = receiver.recv()
        except EOFError as exc:
            raise JobPredictionError("v3.2 source worker exited without a result") from exc
        finally:
            receiver.close()
            if process.is_alive():
                process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        if not response.get("ok"):
            raise JobPredictionError(str(response.get("error") or "v3.2 source capture failed"))
        predictions = list(response.get("predictions") or [])
        if len(predictions) != len(missing):
            raise JobPredictionError("v3.2 source worker returned an incomplete option set")
        for index, prediction in zip(missing, predictions, strict=True):
            peak_vram = float(prediction["train_peak_vram_mib"])
            epoch_ms = float(prediction["train_epoch_ms"])
            if not (0.0 < peak_vram < float("inf") and 0.0 < epoch_ms < float("inf")):
                raise JobPredictionError("v3.2 prediction is non-positive or non-finite")
            self._v32_predictions[keys[index]] = {
                "train_peak_vram_mib": peak_vram,
                "train_epoch_ms": epoch_ms,
            }

    @staticmethod
    def _v32_primary_capture_indices(
        job: TrainingJob,
        specifications: list[ModelSpecification],
        missing: list[int],
    ) -> list[int]:
        """Capture only the candidate's existing training batch at cold start."""
        configured_batch = job.config.runner_kwargs.get(
            "batch_size", job.requested_batch_size
        )
        try:
            configured_batch = int(configured_batch)
        except (TypeError, ValueError):
            configured_batch = None
        if configured_batch is not None:
            for index in missing:
                shapes = specifications[index].input_shapes
                if shapes and int(shapes[0][0]) == configured_batch:
                    return [index]
        return [missing[0]]

    def _start_v32_capture(
        self,
        specifications: list[ModelSpecification],
        keys: list[str],
        missing: list[int],
    ) -> None:
        """Start one CPU-side V3.2 capture and return immediately to the service."""
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=predict_v32_source_options_worker,
            args=(
                sender,
                str(self.settings.v32_source_root),
                str(self.settings.v32_artifact_path),
                [specifications[index].worker_request() for index in missing],
                self.settings.v32_capture_device,
            ),
            daemon=False,
        )
        process.start()
        sender.close()
        timeout_seconds = self.settings.conversion_timeout_seconds * max(1, len(missing))
        self._v32_pending_capture = _PendingV32Capture(
            receiver=receiver,
            process=process,
            keys=tuple(keys[index] for index in missing),
            deadline=time.monotonic() + timeout_seconds,
        )

    def _poll_v32_capture(self) -> None:
        """Consume a ready capture result without blocking the scheduler tick."""
        pending = self._v32_pending_capture
        if pending is None:
            return
        if not pending.receiver.poll(0.0):
            # A child can die before it writes to the pipe (for example while
            # importing a generated source).  Waiting for the nominal timeout
            # leaves Scheduler status permanently "pending" even though no
            # capture is running.  Surface that condition immediately so the
            # caller can record a fallback and continue admission.
            if not pending.process.is_alive():
                self._v32_pending_capture = None
                pending.receiver.close()
                pending.process.join(timeout=0.0)
                raise JobPredictionError("v3.2 source worker exited without a result")
            if time.monotonic() < pending.deadline:
                return
            self._v32_pending_capture = None
            pending.receiver.close()
            if pending.process.is_alive():
                pending.process.terminate()
            pending.process.join(timeout=0.0)
            raise JobPredictionError("v3.2 source capture exceeded its configured deadline")
        self._v32_pending_capture = None
        try:
            response = pending.receiver.recv()
        except EOFError as exc:
            raise JobPredictionError("v3.2 source worker exited without a result") from exc
        finally:
            pending.receiver.close()
            if pending.process.is_alive():
                pending.process.terminate()
            pending.process.join(timeout=0.0)
        if not response.get("ok"):
            raise JobPredictionError(str(response.get("error") or "v3.2 source capture failed"))
        predictions = list(response.get("predictions") or [])
        if len(predictions) != len(pending.keys):
            raise JobPredictionError("v3.2 source worker returned an incomplete option set")
        for key, prediction in zip(pending.keys, predictions, strict=True):
            peak_vram = float(prediction["train_peak_vram_mib"])
            epoch_ms = float(prediction["train_epoch_ms"])
            if not (0.0 < peak_vram < float("inf") and 0.0 < epoch_ms < float("inf")):
                raise JobPredictionError("v3.2 prediction is non-positive or non-finite")
            self._v32_predictions[key] = {
                "train_peak_vram_mib": peak_vram,
                "train_epoch_ms": epoch_ms,
            }

    def _smoke_test_runtime(self) -> None:
        import torch

        assert self._encoded_type is not None
        assert self._runtime is not None
        encoded = self._encoded_type(
            x=torch.zeros((2, 53), dtype=torch.float32),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            edge_attr=torch.zeros((1, 3), dtype=torch.float32),
            u=torch.zeros((1, 40), dtype=torch.float32),
            batch=torch.zeros(2, dtype=torch.long),
        )
        self._runtime.predict(encoded)

    def _cache_key(self, specification: ModelSpecification) -> str:
        source_digest = sha256(specification.source_path.read_bytes()).hexdigest()
        payload = {
            "source_sha256": source_digest,
            "working_directory": str(specification.working_directory),
            "entry": specification.entry,
            "input_shapes": specification.input_shapes,
            "input_dtypes": specification.input_dtypes,
            "precision": specification.precision,
            "constructor_args": specification.constructor_args,
            "constructor_kwargs": specification.constructor_kwargs,
            "steps_per_epoch": specification.steps_per_epoch,
            "artifact_sha256": self.artifact_hash,
        }
        return sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def _convert(self, specification: ModelSpecification) -> Any:
        key = self._cache_key(specification)
        cached = self._cache.pop(key, None)
        if cached is not None:
            self._cache[key] = cached
            return cached

        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=convert_source_worker,
            args=(sender, str(PERFSEER_SRC), specification.worker_request()),
            daemon=True,
        )
        process.start()
        sender.close()
        try:
            if not receiver.poll(self.settings.conversion_timeout_seconds):
                process.terminate()
                process.join(timeout=2.0)
                raise JobPredictionError(
                    f"source conversion exceeded {self.settings.conversion_timeout_seconds:.1f}s"
                )
            response = receiver.recv()
        except EOFError as exc:
            raise JobPredictionError("source conversion subprocess exited without a result") from exc
        finally:
            receiver.close()
            if process.is_alive():
                process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        if not response.get("ok"):
            raise JobPredictionError(str(response.get("error") or "source conversion failed"))

        import torch

        arrays = response["tensors"]
        assert self._encoded_type is not None
        encoded = self._encoded_type(
            x=torch.from_numpy(arrays[0]),
            edge_index=torch.from_numpy(arrays[1]),
            edge_attr=torch.from_numpy(arrays[2]),
            u=torch.from_numpy(arrays[3]),
            batch=torch.from_numpy(arrays[4]),
        )
        self._cache[key] = encoded
        while len(self._cache) > self.settings.cache_size:
            self._cache.popitem(last=False)
        return encoded

    def _convert_many(self, specifications: list[ModelSpecification]) -> list[Any]:
        """Convert batch variants in one worker while reusing model tracing."""
        if not specifications:
            return []
        encoded_by_index: dict[int, Any] = {}
        missing: list[tuple[int, ModelSpecification, str]] = []
        for index, specification in enumerate(specifications):
            key = self._cache_key(specification)
            cached = self._cache.pop(key, None)
            if cached is not None:
                self._cache[key] = cached
                encoded_by_index[index] = cached
            else:
                missing.append((index, specification, key))
        if missing:
            context = multiprocessing.get_context("spawn")
            receiver, sender = context.Pipe(duplex=False)
            process = context.Process(
                target=convert_source_options_worker,
                args=(sender, str(PERFSEER_SRC), [spec.worker_request() for _, spec, _ in missing]),
                daemon=True,
            )
            process.start()
            sender.close()
            timeout_seconds = self.settings.conversion_timeout_seconds * max(1, len(missing))
            try:
                if not receiver.poll(timeout_seconds):
                    process.terminate()
                    process.join(timeout=2.0)
                    raise JobPredictionError(f"batch source conversion exceeded {timeout_seconds:.1f}s")
                response = receiver.recv()
            except EOFError as exc:
                raise JobPredictionError("batch source conversion subprocess exited without a result") from exc
            finally:
                receiver.close()
                if process.is_alive():
                    process.join(timeout=2.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2.0)
            if not response.get("ok"):
                raise JobPredictionError(str(response.get("error") or "batch source conversion failed"))
            raw_options = list(response.get("tensors_by_option") or [])
            if len(raw_options) != len(missing):
                raise JobPredictionError("batch source conversion returned an incomplete option set")
            import torch

            assert self._encoded_type is not None
            for (index, _specification, key), arrays in zip(missing, raw_options, strict=True):
                encoded = self._encoded_type(
                    x=torch.from_numpy(arrays[0]),
                    edge_index=torch.from_numpy(arrays[1]),
                    edge_attr=torch.from_numpy(arrays[2]),
                    u=torch.from_numpy(arrays[3]),
                    batch=torch.from_numpy(arrays[4]),
                )
                encoded_by_index[index] = encoded
                self._cache[key] = encoded
                while len(self._cache) > self.settings.cache_size:
                    self._cache.popitem(last=False)
        return [encoded_by_index[index] for index in range(len(specifications))]

    def _prediction_key(self, job: TrainingJob, batch_size: int, specification: ModelSpecification) -> str:
        payload = {
            "job_id": job.job_id,
            "batch_size": int(batch_size),
            "feature_key": self._cache_key(specification),
            "hardware": {
                "gpu_name": self.hardware.gpu_name,
                "compute_capability": self.hardware.compute_capability,
                "total_vram_mb": self.hardware.total_vram_mb,
            },
            "model_id": self.model_id,
            "artifact_sha256": self.artifact_hash,
        }
        return sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def predict_avg_vram_options(self, job: TrainingJob, batch_sizes: Sequence[int]) -> dict[int, float]:
        """Predict all requested batches with one trace/conversion bundle."""
        if not self.available:
            unavailable_error = self.unavailable_reason or "ML predictor unavailable"
            self.last_sources[job.job_id] = "branch_profile"
            self.last_errors[job.job_id] = unavailable_error
            raise JobPredictionError(unavailable_error)
        ordered_sizes = [int(value) for value in dict.fromkeys(batch_sizes)]
        try:
            specifications = [
                model_specification_for_job(job, batch_size)
                for batch_size in ordered_sizes
            ]
        except Exception as exc:
            prediction_error = (
                exc
                if isinstance(exc, JobPredictionError)
                else JobPredictionError(f"{type(exc).__name__}: {exc}")
            )
            self.last_sources[job.job_id] = "branch_profile"
            self.last_errors[job.job_id] = str(prediction_error)
            raise prediction_error
        keys = [self._prediction_key(job, batch_size, spec) for batch_size, spec in zip(ordered_sizes, specifications, strict=True)]
        failed = next((self._job_failures[key] for key in keys if key in self._job_failures), None)
        if failed is not None:
            self.last_sources[job.job_id] = "branch_profile"
            self.last_errors[job.job_id] = failed
            raise JobPredictionError(failed)
        if self._runtime == "v32":
            try:
                self._predict_v32_options(job, specifications, keys)
            except JobPredictionPending as exc:
                self.last_sources[job.job_id] = "branch_profile"
                self.last_errors[job.job_id] = str(exc)
                raise
            except Exception as exc:
                prediction_error = exc if isinstance(exc, JobPredictionError) else JobPredictionError(f"{type(exc).__name__}: {exc}")
                for key in keys:
                    self._job_failures[key] = str(prediction_error)
                self.last_sources[job.job_id] = "branch_profile"
                self.last_errors[job.job_id] = str(prediction_error)
                raise prediction_error
            self.last_sources[job.job_id] = "ml_predictor"
            self.last_errors.pop(job.job_id, None)
            return {
                batch_size: self._v32_predictions[key]["train_peak_vram_mib"]
                for batch_size, key in zip(ordered_sizes, keys, strict=True)
                if key in self._v32_predictions
            }
        missing_indices = [index for index, key in enumerate(keys) if key not in self._job_values]
        try:
            missing_encoded = (
                self._convert_many([specifications[index] for index in missing_indices])
                if missing_indices
                else []
            )
            assert self._runtime is not None
            for index, encoded in zip(missing_indices, missing_encoded, strict=True):
                value = float(self._runtime.predict_train_mem_mb(encoded))
                if not (0.0 < value < float("inf")):
                    raise JobPredictionError(f"invalid train_mem prediction {value}")
                self._job_values[keys[index]] = value
        except Exception as exc:
            prediction_error = exc if isinstance(exc, JobPredictionError) else JobPredictionError(f"{type(exc).__name__}: {exc}")
            for key in keys:
                self._job_failures[key] = str(prediction_error)
            self.last_sources[job.job_id] = "branch_profile"
            self.last_errors[job.job_id] = str(prediction_error)
            raise prediction_error
        self.last_sources[job.job_id] = "ml_predictor"
        self.last_errors.pop(job.job_id, None)
        return {batch_size: self._job_values[key] for batch_size, key in zip(ordered_sizes, keys, strict=True)}

    def predict_avg_vram_mb(self, job: TrainingJob, batch_size: int) -> float:
        return self.predict_avg_vram_options(job, [batch_size])[int(batch_size)]

    def predict_seconds_per_epoch_options(self, job: TrainingJob, batch_sizes: Sequence[int]) -> dict[int, float]:
        """Return scheduler-grade epoch predictions when the artifact explicitly declares them.

        The unsupported ``train_time`` target is deliberately not accepted because
        it is not a measured full-epoch label.
        """
        if "train_epoch_ms" not in self._target_names:
            raise JobPredictionError("predictor artifact does not expose train_epoch_ms")
        if not self.available:
            raise JobPredictionError(self.unavailable_reason or "ML predictor unavailable")
        epoch_index = self._target_names.index("train_epoch_ms")
        ordered_sizes = [int(value) for value in dict.fromkeys(batch_sizes)]
        try:
            specifications = [
                model_specification_for_job(job, batch_size)
                for batch_size in ordered_sizes
            ]
        except Exception as exc:
            prediction_error = (
                exc
                if isinstance(exc, JobPredictionError)
                else JobPredictionError(f"{type(exc).__name__}: {exc}")
            )
            self.last_sources[job.job_id] = "branch_profile"
            self.last_errors[job.job_id] = str(prediction_error)
            raise prediction_error
        keys = [self._prediction_key(job, batch_size, spec) for batch_size, spec in zip(ordered_sizes, specifications, strict=True)]
        if self._runtime == "v32":
            try:
                self._predict_v32_options(job, specifications, keys)
            except Exception as exc:
                if isinstance(exc, JobPredictionError):
                    raise
                raise JobPredictionError(f"{type(exc).__name__}: {exc}") from exc
            return {
                batch_size: self._v32_predictions[key]["train_epoch_ms"] / 1000.0
                for batch_size, key in zip(ordered_sizes, keys, strict=True)
                if key in self._v32_predictions
            }
        missing = [index for index, key in enumerate(keys) if key not in self._job_epoch_seconds]
        try:
            encoded_options = self._convert_many([specifications[index] for index in missing]) if missing else []
            assert self._runtime is not None
            for index, encoded in zip(missing, encoded_options, strict=True):
                milliseconds = float(self._runtime.predict(encoded)[epoch_index].item())
                seconds = milliseconds / 1000.0
                if not (0.0 < seconds < float("inf")):
                    raise JobPredictionError(f"invalid train_epoch_ms prediction {milliseconds}")
                self._job_epoch_seconds[keys[index]] = seconds
        except Exception as exc:
            if isinstance(exc, JobPredictionError):
                raise
            raise JobPredictionError(f"{type(exc).__name__}: {exc}") from exc
        return {batch_size: self._job_epoch_seconds[key] for batch_size, key in zip(ordered_sizes, keys, strict=True)}

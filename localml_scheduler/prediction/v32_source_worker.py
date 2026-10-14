"""Isolated capture worker for the PerfSeer v3.2 paired-graph student."""

from __future__ import annotations

import os
import re
import sys
from contextlib import contextmanager
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any


def _training_settings(request: dict[str, Any]) -> dict[str, Any]:
    try:
        steps_per_epoch = int(request.get("steps_per_epoch"))
    except (TypeError, ValueError):
        steps_per_epoch = 1
    return {
        "microbatch_size": int(
            request.get("training_microbatch_size")
            or request["input_shapes"][0][0]
        ),
        "gradient_accumulation_steps": 1,
        "precision": str(request["precision"]),
        "backend": "cuda_eager",
        "optimizer": {"name": "adam", "learning_rate": 1e-3, "weight_decay": 0.0},
        "scheduler": {"name": "none"},
        "steps_per_epoch": max(1, steps_per_epoch),
    }


def _capture_request_with_small_batch(request: dict[str, Any]) -> dict[str, Any]:
    """Bound source-graph capture without changing the inferred train batch.

    PerfSeer needs the model topology and the intended training microbatch,
    but its CPU graph capture need not materialize all 32 high-resolution
    images.  A batch of two preserves normal batch-shaped operations while
    keeping capture RAM bounded; ``training_microbatch_size`` retains the
    scheduler-facing physical batch for inference.
    """
    copied = dict(request)
    shapes: list[list[Any]] = []
    actual_batch: int | None = None
    for raw_shape in request["input_shapes"]:
        shape = list(raw_shape)
        if shape:
            try:
                current_batch = int(shape[0])
            except (TypeError, ValueError):
                current_batch = 1
            if actual_batch is None:
                actual_batch = current_batch
            shape[0] = max(1, min(2, current_batch))
        shapes.append(shape)
    copied["input_shapes"] = shapes
    if actual_batch is not None:
        copied["training_microbatch_size"] = actual_batch
    return copied


def _move_tensors(value: Any, device: Any) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tensors(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    return value


def _align_last_tensor_feature_dim(model: Any, inputs: Any, device: Any) -> Any:
    """Repair a statically inferred tabular width from the model's own error.

    MLEvolve candidates often build the tabular branch from the real CSV while
    exposing only a source-level column list.  Engineered features therefore
    make the static width too small.  A single no-grad forward is safe here:
    it changes no model state and only replaces the final tensor when PyTorch
    identifies its required linear input width unambiguously.
    """
    if not isinstance(inputs, tuple) or len(inputs) < 2:
        return inputs
    last = inputs[-1]
    if not hasattr(last, "ndim") or int(last.ndim) != 2:
        return inputs
    try:
        import torch

        with torch.no_grad():
            model(*inputs)
        return inputs
    except RuntimeError as exc:
        match = re.search(
            r"mat1 and mat2 shapes cannot be multiplied \(\d+x(\d+) and (\d+)x\d+\)",
            str(exc),
        )
        if match is None:
            return inputs
        observed, required = (int(value) for value in match.groups())
        if observed != int(last.shape[1]) or required <= 0 or required == observed:
            return inputs
        replacement = torch.zeros(
            (int(last.shape[0]), required),
            dtype=last.dtype,
            device=device,
        )
        return (*inputs[:-1], replacement)


@contextmanager
def _execution_directory(request: dict[str, Any]):
    """Replay a candidate import from the same directory as its execution."""
    source_path = Path(str(request["source_path"])).expanduser().resolve()
    raw_directory = request.get("working_directory")
    directory = (
        Path(str(raw_directory)).expanduser().resolve()
        if raw_directory
        else source_path.parent
    )
    if not directory.is_dir():
        raise FileNotFoundError(f"source execution directory not found: {directory}")
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


def predict_v32_source_options_worker(
    connection: Connection,
    source_root: str,
    artifact_path: str,
    requests: list[dict[str, Any]],
    capture_device: str = "cpu",
) -> None:
    """Capture source variants and return only scheduler-safe v3.2 outputs."""
    capture_device = str(capture_device or "cpu").strip().lower()
    if capture_device not in {"cpu", "cuda"}:
        connection.send({"ok": False, "error": f"unsupported v3.2 capture device: {capture_device}"})
        connection.close()
        return
    if capture_device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["NVIDIA_VISIBLE_DEVICES"] = "void"
    try:
        root = str(Path(source_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        import torch

        if capture_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("v3.2 CUDA capture requested but no CUDA device is available")
        device = torch.device(capture_device)

        from perfseer_v3.inputs import build_inputs
        from perfseer_v3.load import load_model_entry, source_sha256
        from perfseer_v3.source import SourceModelSpecV3
        from perfseer_v32.capture import capture
        from perfseer_v32.inference import predict

        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        predictions: list[dict[str, float]] = []
        for request in requests:
            capture_request = _capture_request_with_small_batch(request)
            specification = SourceModelSpecV3.from_shapes(
                capture_request["source_path"],
                capture_request["entry"],
                tuple(tuple(shape) for shape in capture_request["input_shapes"]),
                input_dtypes=tuple(capture_request["input_dtypes"]),
                constructor_args=tuple(capture_request["constructor_args"]),
                constructor_kwargs=dict(capture_request["constructor_kwargs"]),
            )
            with _execution_directory(request):
                model = load_model_entry(
                    specification.source_path,
                    specification.entry,
                    constructor_args=specification.constructor_args,
                    constructor_kwargs=dict(specification.constructor_kwargs),
                ).to(device)
                inputs, _ = build_inputs(
                    specification.positional_inputs,
                    specification.keyword_inputs,
                    seed=0,
                )
                inputs = _move_tensors(inputs, device)
                inputs = _align_last_tensor_feature_dim(model, inputs, device)
                design, _ = capture(
                    model,
                    inputs,
                    None,
                    None,
                    _training_settings(capture_request),
                    architecture_key=source_sha256(specification.source_path),
                )
            row = predict(artifact, [design], device="cpu", amp=False, microbatch=1)[0]
            predictions.append(
                {
                    "train_epoch_ms": float(row["train_epoch_ms"]),
                    "train_peak_vram_mib": float(row["train_peak_vram_mib"]),
                }
            )
        connection.send(
            {"ok": True, "predictions": predictions, "capture_device": capture_device}
        )
    except BaseException as exc:
        connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()

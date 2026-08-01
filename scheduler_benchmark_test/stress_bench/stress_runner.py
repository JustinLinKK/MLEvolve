"""Scheduler-compatible training runner for Stress Test Data v1.0 models."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from localml_scheduler.domain import SafePointType
from localml_scheduler.execution.runner_protocol import RunnerContext
from localml_scheduler.profiling.runtime_probe import estimate_total_runtime_from_epoch_1

_BUILD_MODEL_CACHE: dict[str, Any] = {}


def _build_model_fn(source_path: str):
    fn = _BUILD_MODEL_CACHE.get(source_path)
    if fn is None:
        spec = importlib.util.spec_from_file_location("stress_model_source", source_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = module.build_model
        _BUILD_MODEL_CACHE[source_path] = fn
    return fn


def _autocast(precision: str, device: torch.device):
    if precision == "bf16_amp":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    if precision == "fp16_amp":
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return torch.autocast(device_type=device.type, enabled=False)


def _apply_tf32(precision: str) -> None:
    enabled = precision == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = enabled
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = enabled


def train_stress_model(
    *,
    source_path: str,
    constructor_kwargs: dict[str, Any],
    input_shape: list[int],
    precision: str,
    epochs: int,
    batches_per_epoch: int,
    device: torch.device,
    step_callback=None,
    stream_data: bool = False,
) -> dict[str, Any]:
    """Shared training body. stream_data=True refreshes the input each step so every epoch
    is a full pass over batches_per_epoch freshly-drawn synthetic samples."""
    _apply_tf32(precision)
    build_model = _build_model_fn(source_path)
    model = build_model(**constructor_kwargs).to(device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=1e-3) if params else None
    scaler = torch.amp.GradScaler(device.type, enabled=(precision == "fp16_amp" and device.type == "cuda"))
    x = torch.randn(tuple(int(v) for v in input_shape), device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    epoch_seconds: list[float] = []
    global_step = 0
    last_loss = 0.0
    for epoch in range(int(epochs)):
        epoch_started = time.perf_counter()
        for _ in range(int(batches_per_epoch)):
            if stream_data:
                x.normal_()
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with _autocast(precision, device):
                out = model(x)
                loss = F.mse_loss(out.float(), torch.zeros_like(out, dtype=torch.float32))
            if loss.requires_grad:
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            if optimizer is not None:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
            global_step += 1
            last_loss = float(loss.detach().float().item()) if global_step % 1000 == 0 else last_loss
            if step_callback is not None:
                step_callback(epoch, global_step, last_loss)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_seconds.append(time.perf_counter() - epoch_started)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_seconds = time.perf_counter() - started
    peak_mib = float(torch.cuda.max_memory_reserved(device) / (1024 * 1024)) if device.type == "cuda" else 0.0
    peak_alloc_mib = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if device.type == "cuda" else 0.0
    del model, x, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "training_seconds": total_seconds, "epoch_seconds": epoch_seconds,
        "global_steps": global_step, "final_loss": last_loss,
        "peak_reserved_mib": peak_mib, "peak_allocated_mib": peak_alloc_mib,
    }


def run_stress_job(context: RunnerContext) -> dict[str, Any]:
    params = dict(context.job.config.runner_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend_name = str(context.job.metadata.get("placement_backend") or "exclusive")
    epochs = int(context.job.max_epochs or params.get("epochs") or 1)
    batches_per_epoch = int(params.get("batches_per_epoch") or 100)
    run_started = time.perf_counter()
    context.control_hook.safe_point(
        SafePointType.BEFORE_TRAIN, epoch=0, global_step=0, metrics={}, message="before training",
        state_factory=lambda: {"job_config": context.job.to_dict()},
    )
    steps_per_epoch = batches_per_epoch
    state: dict[str, Any] = {"profiled": False}

    def on_step(epoch: int, global_step: int, loss: float) -> None:
        if global_step % steps_per_epoch:
            return
        if not state["profiled"] and epoch == 0:
            epoch_1_seconds = time.perf_counter() - run_started
            estimated = estimate_total_runtime_from_epoch_1(
                startup_seconds=0.0, epoch_1_seconds=epoch_1_seconds, total_epochs=epochs)
            context.upsert_runtime_profile(
                backend_name=backend_name, strategy="epoch_1", startup_seconds=0.0,
                epoch_1_seconds=epoch_1_seconds, steps_per_epoch=steps_per_epoch,
                avg_step_time_ms=epoch_1_seconds * 1000.0 / max(1, steps_per_epoch),
                estimated_total_runtime_seconds=estimated, confidence=0.90, source="observed",
                observations=1, metadata={"runner": "stress_runner"})
            state["profiled"] = True
        context.control_hook.safe_point(
            SafePointType.EPOCH, epoch=epoch, global_step=global_step, metrics={"loss": loss},
            state_factory=lambda: {"job_config": context.job.to_dict()})

    result = train_stress_model(
        source_path=str(params["source_path"]), constructor_kwargs=dict(params.get("constructor_kwargs") or {}),
        input_shape=list(params["input_shape"]), precision=str(params.get("precision") or "fp32_ieee"),
        epochs=epochs, batches_per_epoch=batches_per_epoch, device=device, step_callback=on_step,
        stream_data=bool(params.get("stream_data")))

    context.control_hook.safe_point(
        SafePointType.EXPLICIT, epoch=epochs, global_step=result["global_steps"],
        metrics={"loss": result["final_loss"]}, message="after training",
        state_factory=lambda: {"job_config": context.job.to_dict()})
    result["wall_seconds"] = time.perf_counter() - run_started
    result["backend"] = backend_name
    result["job_id"] = context.job.job_id
    result["step_idx"] = (context.job.metadata or {}).get("step_idx")
    result["finished_at"] = time.time()
    result_dir = params.get("result_dir")
    if result_dir:
        path = Path(result_dir) / f"{context.job.job_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
    return result


def make_baseline_checkpoint(path: str | Path, payload: dict[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return str(path)


__all__ = ["run_stress_job", "train_stress_model", "make_baseline_checkpoint"]

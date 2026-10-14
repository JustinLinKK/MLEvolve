"""Scheduler-compatible training runner for Stress Test Data v1.0 models."""

from __future__ import annotations

import importlib.util
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from localml_scheduler.domain import SafePointType
from localml_scheduler.execution.runner_protocol import RunnerContext
from localml_scheduler.profiling.runtime_probe import estimate_total_runtime_from_epoch_1

_BUILD_MODEL_CACHE: dict[str, Any] = {}
_MODEL_BUILD_LOCK = threading.Lock()
_CUDA_GLOBAL_LOCK = threading.Lock()


def _build_model_fn(source_path: str):
    with _MODEL_BUILD_LOCK:
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
    memory_ballast_mib: int = 0,
    compute_repeats: int = 1,
    bandwidth_mib: int = 0,
    step_delay_ms: float = 0.0,
    random_seed: int = 0,
    manage_tf32: bool = True,
) -> dict[str, Any]:
    """Shared training body. stream_data=True refreshes the input each step so every epoch
    is a full pass over batches_per_epoch freshly-drawn synthetic samples."""
    # TF32 flags and the CUDA caching allocator are process-global.  Stream
    # benchmark jobs use one common precision and disable per-job mutation.
    if manage_tf32:
        with _CUDA_GLOBAL_LOCK:
            _apply_tf32(precision)
    build_model = _build_model_fn(source_path)
    with _MODEL_BUILD_LOCK:
        torch.manual_seed(int(random_seed))
        model = build_model(**constructor_kwargs).to(device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=1e-3) if params else None
    scaler = torch.amp.GradScaler(device.type, enabled=(precision == "fp16_amp" and device.type == "cuda"))
    generator = None
    if device.type == "cuda":
        generator = torch.Generator(device=device).manual_seed(int(random_seed))
    x = torch.randn(
        tuple(int(v) for v in input_shape), device=device, generator=generator
    )
    ballast: list[torch.Tensor] = []
    remaining_bytes = max(0, int(memory_ballast_mib)) * 1024 * 1024
    chunk_bytes = 256 * 1024 * 1024
    while remaining_bytes:
        size = min(chunk_bytes, remaining_bytes)
        chunk = torch.empty(size, dtype=torch.uint8, device=device)
        chunk.zero_()
        ballast.append(chunk)
        remaining_bytes -= size
    bandwidth_buffer = (
        torch.ones(max(1, int(bandwidth_mib)) * 1024 * 1024 // 4, device=device)
        if int(bandwidth_mib) > 0
        else None
    )
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()
        start_allocated = int(torch.cuda.memory_allocated(device))
        start_reserved = int(torch.cuda.memory_reserved(device))
    else:
        start_allocated = start_reserved = 0
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
            loss = None
            for _repeat in range(max(1, int(compute_repeats))):
                with _autocast(precision, device):
                    out = model(x)
                    loss = F.mse_loss(
                        out.float(), torch.zeros_like(out, dtype=torch.float32)
                    )
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
            if bandwidth_buffer is not None:
                with torch.no_grad():
                    bandwidth_buffer.mul_(1.000001).add_(0.000001)
            if step_delay_ms > 0:
                time.sleep(float(step_delay_ms) / 1000.0)
            global_step += 1
            if loss is not None and (global_step % 1000 == 0 or global_step == 1):
                last_loss = float(loss.detach().float().item())
            if step_callback is not None:
                step_callback(epoch, global_step, last_loss)
        if device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()
        epoch_seconds.append(time.perf_counter() - epoch_started)

    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()
    total_seconds = time.perf_counter() - started
    # Report this process's live allocation plus externally sampled NVML peaks.
    peak_mib = (
        float(max(start_reserved, torch.cuda.memory_reserved(device)) / (1024 * 1024))
        if device.type == "cuda"
        else 0.0
    )
    peak_alloc_mib = (
        float(max(start_allocated, torch.cuda.memory_allocated(device)) / (1024 * 1024))
        if device.type == "cuda"
        else 0.0
    )
    del model, x, optimizer, ballast, bandwidth_buffer
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()
        with _CUDA_GLOBAL_LOCK:
            torch.cuda.empty_cache()
    return {
        "training_seconds": total_seconds, "epoch_seconds": epoch_seconds,
        "global_steps": global_step, "final_loss": last_loss,
        "peak_reserved_mib": peak_mib, "peak_allocated_mib": peak_alloc_mib,
        "memory_ballast_mib": int(memory_ballast_mib),
        "worker_pid": os.getpid(),
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
        if not state["profiled"] and epoch in {0, 1}:
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

    completed_epochs = max(0, int(context.job.metadata.get("last_completed_epoch") or 0))
    remaining_epochs = max(0, epochs - completed_epochs)

    def resumed_step_callback(epoch: int, global_step: int, loss: float) -> None:
        reported_epoch = completed_epochs + epoch + 1
        reported_step = completed_epochs * batches_per_epoch + global_step
        on_step(reported_epoch, reported_step, loss)

    try:
        result = train_stress_model(
            source_path=str(params["source_path"]), constructor_kwargs=dict(params.get("constructor_kwargs") or {}),
            input_shape=list(params["input_shape"]), precision=str(params.get("precision") or "fp32_ieee"),
            epochs=remaining_epochs, batches_per_epoch=batches_per_epoch, device=device,
            step_callback=resumed_step_callback,
            stream_data=bool(params.get("stream_data")),
            memory_ballast_mib=int(params.get("memory_ballast_mib") or 0),
            compute_repeats=int(params.get("compute_repeats") or 1),
            bandwidth_mib=int(params.get("bandwidth_mib") or 0),
            step_delay_ms=float(params.get("step_delay_ms") or 0.0),
            random_seed=int(params.get("random_seed") or 0),
            manage_tf32=bool(params.get("manage_tf32", True)))
    except BaseException:
        if device.type == "cuda":
            try:
                torch.cuda.current_stream(device).synchronize()
            finally:
                with _CUDA_GLOBAL_LOCK:
                    torch.cuda.empty_cache()
        raise
    result["global_steps"] = completed_epochs * batches_per_epoch + int(result["global_steps"])

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

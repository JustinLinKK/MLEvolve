"""Shared full-dataset training runtime for generated histopathology jobs."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable
import csv
import json
import math
import os
import random
import time

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

from . import A10_VRAM_CAP_MIB, DATASET_SIZE, INPUT_SIZE


class HistopathologyDataset(Dataset):
    def __init__(self, data_root: str | Path):
        self.root = Path(data_root).expanduser().resolve()
        labels_path = self.root / "train_labels.csv"
        image_root = self.root / "train"
        if not labels_path.is_file() or not image_root.is_dir():
            raise FileNotFoundError(
                f"Histopathology dataset not found under {self.root}; expected train_labels.csv and train/."
            )
        with labels_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != DATASET_SIZE:
            raise ValueError(f"Expected {DATASET_SIZE} labeled images, found {len(rows)} in {labels_path}")
        partial_limit = os.environ.get("STANDARD_BENCH_MAX_SAMPLES")
        if partial_limit is not None:
            if os.environ.get("STANDARD_BENCH_ALLOW_PARTIAL") != "1":
                raise RuntimeError("STANDARD_BENCH_MAX_SAMPLES requires STANDARD_BENCH_ALLOW_PARTIAL=1")
            rows = rows[: max(1, int(partial_limit))]
        self.samples = [(image_root / f"{row['id']}.tif", int(row["label"])) for row in rows]
        self.transform = transforms.Compose(
            [
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.7, 0.5, 0.65), (0.15, 0.18, 0.14)),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label


def _configure_precision(precision: str) -> tuple[torch.dtype | None, bool]:
    if not torch.cuda.is_available():
        return None, False
    torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
    torch.backends.cudnn.allow_tf32 = precision == "tf32"
    if precision == "fp16":
        return torch.float16, True
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 job requires an Ampere-or-newer CUDA GPU")
        return torch.bfloat16, False
    return None, False


def _apply_allocator_cap(device: torch.device) -> tuple[float | None, int | None]:
    if device.type != "cuda":
        return None, None
    total_mib = int(torch.cuda.get_device_properties(device).total_memory / (1024**2))
    requested_mib = int(os.environ.get("STANDARD_BENCH_VRAM_CAP_MIB", str(A10_VRAM_CAP_MIB)))
    effective_mib = min(requested_mib, int(total_mib * 0.95))
    fraction = min(0.95, max(0.01, effective_mib / float(total_mib)))
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)
    return fraction, effective_mib


def _autocast_context(device: torch.device, dtype: torch.dtype | None):
    if device.type == "cuda" and dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def run_generated_job(
    *,
    spec: dict[str, Any],
    build_model: Callable[[], nn.Module],
    build_loader: Callable[..., Any],
    register_training_state: Callable[..., None],
    restore_training_state: Callable[..., dict[str, int]],
    optimizer_step_completed: Callable[..., None],
    session: Any,
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    job_id = str(spec["job_id"])
    seed = int(spec["seed"])
    precision = str(spec["precision"])
    data_root = os.environ.get("HISTOPATH_DATA_ROOT")
    if not data_root:
        raise RuntimeError("Set HISTOPATH_DATA_ROOT to the prepared/public histopathology dataset directory")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    allocator_fraction, allocator_cap_mib = _apply_allocator_cap(device)
    autocast_dtype, use_scaler = _configure_precision(precision)

    dataset = HistopathologyDataset(data_root)
    loader = build_loader(session, dataset)
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec["learning_rate"]), weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler and device.type == "cuda")
    register_training_state(session, model, optimizer, scaler)
    progress = restore_training_state(session)
    batch_size = int(session.batch_size)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    samples_seen = 0
    global_step = int(progress["global_step"])
    last_loss = math.nan
    model.train()
    for epoch_index in range(int(progress["epoch"]), int(epochs)):
        epoch_started = time.perf_counter()
        epoch_samples = 0
        for batch_index, (features, labels) in enumerate(loader):
            features = features.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, autocast_dtype):
                logits = model(features)
                loss = criterion(logits, labels)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            last_loss = float(loss.detach().item())
            count = int(labels.shape[0])
            samples_seen += count
            epoch_samples += count
            global_step += 1
            optimizer_step_completed(
                session,
                count,
                epoch_index,
                batch_index,
                global_step,
                {"loss": last_loss},
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_started
        metric = {
            "epoch": epoch_index + 1,
            "global_step": global_step,
            "loss": last_loss,
            "epoch_seconds": epoch_seconds,
            "samples_seen": samples_seen,
            "epoch_samples": epoch_samples,
            "batch_size": int(batch_size),
        }
        print("MLEVOLVE_METRIC " + json.dumps(metric, sort_keys=True), flush=True)

    elapsed = time.perf_counter() - started
    peak_allocated_mib = None
    peak_reserved_mib = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024**2)
    output_root = Path(
        os.environ.get("STANDARD_BENCH_RESULT_DIR", f"working/benchmark_jobs/{job_id}")
    ).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "standard-histopath-job-result-v1",
        "job_id": job_id,
        "family": spec["family"],
        "architecture": spec["architecture"],
        "variant": spec["variant"],
        "precision": precision,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "dataset_size": len(dataset),
        "samples_seen": samples_seen,
        "global_steps": global_step,
        "loss": last_loss,
        "elapsed_seconds": elapsed,
        "throughput_images_per_second": samples_seen / elapsed if elapsed > 0 else None,
        "peak_allocated_mib": peak_allocated_mib,
        "peak_reserved_mib": peak_reserved_mib,
        "allocator_fraction": allocator_fraction,
        "allocator_cap_mib": allocator_cap_mib,
        "physical_device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    (output_root / "metric.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("STANDARD_BENCH_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    return result

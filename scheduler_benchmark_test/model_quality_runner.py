"""Deterministic classification runner used to audit scheduler model quality.

The pressure benchmark intentionally measures throughput with a synthetic loss.
This runner is separate: it trains a learnable classification problem, evaluates
an untouched validation set after every epoch, and checkpoints all state needed
for an exact scheduler pause/resume.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from localml_scheduler.domain import SafePointType
from localml_scheduler.execution.runner_protocol import RunnerContext


class QualityMLP(nn.Module):
    """Moderately compute-heavy MLP with no stochastic layers."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def create_quality_baseline(
    path: str | Path,
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    seed: int,
) -> str:
    """Create the exact initial weights shared by all execution modes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = QualityMLP(input_dim, hidden_dim, output_dim)
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": int(input_dim),
            "hidden_dim": int(hidden_dim),
            "output_dim": int(output_dim),
            "seed": int(seed),
        },
        path,
    )
    return str(path)


def _teacher(input_dim: int, output_dim: int, teacher_seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(teacher_seed))
    return torch.randn(input_dim, output_dim, generator=generator) / input_dim**0.5


def build_quality_dataset(
    *,
    num_samples: int,
    input_dim: int,
    output_dim: int,
    dataset_seed: int,
    teacher_seed: int,
    label_noise: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a reproducible nonlinear classification task on the CPU."""
    generator = torch.Generator().manual_seed(int(dataset_seed))
    features = torch.randn(num_samples, input_dim, generator=generator)
    weights = _teacher(input_dim, output_dim, teacher_seed)
    logits = features @ weights
    nonlinear_width = min(input_dim, output_dim * 2)
    nonlinear = torch.sin(features[:, :nonlinear_width]).reshape(
        num_samples, output_dim, -1
    ).mean(dim=2)
    logits = logits + 0.45 * nonlinear
    if label_noise > 0:
        logits = logits + float(label_noise) * torch.randn(
            logits.shape, generator=generator
        )
    labels = logits.argmax(dim=1)
    return features, labels


def _parameter_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _autocast(precision: str, device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    if precision == "bf16_amp":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16_amp":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    criterion: nn.Module,
    precision: str,
) -> tuple[float, float, list[int]]:
    model.eval()
    total_loss = 0.0
    correct = 0
    predictions: list[int] = []
    for start in range(0, len(labels), batch_size):
        batch_x = features[start : start + batch_size]
        batch_y = labels[start : start + batch_size]
        with _autocast(precision, features.device):
            logits = model(batch_x).float()
        total_loss += float(criterion(logits, batch_y).item()) * len(batch_y)
        predicted = logits.argmax(dim=1)
        correct += int((predicted == batch_y).sum().item())
        predictions.extend(int(value) for value in predicted.cpu().tolist())
    model.train()
    return correct / max(1, len(labels)), total_loss / max(1, len(labels)), predictions


def train_quality_model(
    *,
    baseline: dict[str, Any],
    params: dict[str, Any],
    device: torch.device,
    resume_state: dict[str, Any] | None = None,
    epoch_callback: Callable[[int, int, dict[str, float], Callable[[], dict[str, Any]]], None]
    | None = None,
) -> dict[str, Any]:
    """Train and evaluate one paired quality replicate."""
    input_dim = int(params.get("input_dim", baseline["input_dim"]))
    hidden_dim = int(params.get("hidden_dim", baseline["hidden_dim"]))
    output_dim = int(params.get("output_dim", baseline["output_dim"]))
    batch_size = int(params.get("batch_size", 256))
    epochs = int(params.get("epochs", 10))
    dataset_seed = int(params.get("dataset_seed", 3107))
    teacher_seed = int(params.get("teacher_seed", 991))
    train_samples = int(params.get("train_samples", 16384))
    validation_samples = int(params.get("validation_samples", 4096))
    label_noise = float(params.get("label_noise", 0.30))
    precision = str(params.get("precision", "bf16_amp"))
    step_delay_ms = float(params.get("step_delay_ms", 0.0))

    train_x, train_y = build_quality_dataset(
        num_samples=train_samples,
        input_dim=input_dim,
        output_dim=output_dim,
        dataset_seed=dataset_seed,
        teacher_seed=teacher_seed,
        label_noise=label_noise,
    )
    validation_x, validation_y = build_quality_dataset(
        num_samples=validation_samples,
        input_dim=input_dim,
        output_dim=output_dim,
        dataset_seed=dataset_seed + 1_000_003,
        teacher_seed=teacher_seed,
        label_noise=label_noise,
    )
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    validation_x = validation_x.to(device)
    validation_y = validation_y.to(device)

    model = QualityMLP(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(baseline["model_state"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params.get("learning_rate", 1.5e-3)),
        weight_decay=float(params.get("weight_decay", 1e-4)),
    )
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0
    global_step = 0
    history: list[dict[str, float | int]] = []
    if resume_state:
        model.load_state_dict(resume_state["model_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        start_epoch = int(resume_state.get("epoch", 0))
        global_step = int(resume_state.get("global_step", 0))
        history = list(resume_state.get("history") or [])

    def checkpoint_state(epoch: int) -> dict[str, Any]:
        return {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "history": list(history),
        }

    started = time.perf_counter()
    model.train()
    for epoch in range(start_epoch, epochs):
        epoch_started = time.perf_counter()
        permutation_generator = torch.Generator().manual_seed(dataset_seed + epoch * 104729)
        permutation = torch.randperm(train_samples, generator=permutation_generator).to(device)
        loss_sum = 0.0
        sample_count = 0
        for start in range(0, train_samples, batch_size):
            indices = permutation[start : start + batch_size]
            batch_x = train_x.index_select(0, indices)
            batch_y = train_y.index_select(0, indices)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(precision, device):
                logits = model(batch_x).float()
                loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().item()) * len(batch_y)
            sample_count += len(batch_y)
            global_step += 1
            if step_delay_ms > 0:
                time.sleep(step_delay_ms / 1000.0)
        validation_accuracy, validation_loss, _ = _evaluate(
            model,
            validation_x,
            validation_y,
            batch_size=batch_size,
            criterion=criterion,
            precision=precision,
        )
        row: dict[str, float | int] = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(1, sample_count),
            "validation_accuracy": validation_accuracy,
            "validation_loss": validation_loss,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        if epoch_callback is not None:
            epoch_callback(
                epoch + 1,
                global_step,
                {
                    "accuracy": validation_accuracy,
                    "validation_loss": validation_loss,
                    "loss": float(row["train_loss"]),
                },
                lambda epoch=epoch + 1: checkpoint_state(epoch),
            )

    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()
    final_accuracy, final_validation_loss, predictions = _evaluate(
        model,
        validation_x,
        validation_y,
        batch_size=batch_size,
        criterion=criterion,
        precision=precision,
    )
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()
    result = {
        "training_seconds": time.perf_counter() - started,
        "epochs_completed": epochs,
        "global_steps": global_step,
        "final_validation_accuracy": final_accuracy,
        "final_validation_loss": final_validation_loss,
        "history": history,
        "validation_predictions": predictions,
        "validation_label_sha256": hashlib.sha256(
            validation_y.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest(),
        "model_parameter_sha256": _parameter_sha256(model),
        "stream_host_pid": os.getpid(),
        "cuda_stream_id": (
            int(torch.cuda.current_stream(device).cuda_stream)
            if device.type == "cuda"
            else None
        ),
    }
    del model, optimizer, train_x, train_y, validation_x, validation_y
    return result


def run_quality_job(context: RunnerContext) -> dict[str, Any]:
    """Scheduler protocol entry point with complete pause/resume checkpoints."""
    params = dict(context.job.config.runner_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = context.load_baseline_object()
    resume_payload = context.load_resume_checkpoint()
    resume_state = resume_payload.get("state") if resume_payload else None
    if resume_state and "model_state" not in resume_state:
        resume_state = None
    start_epoch = int((resume_state or {}).get("epoch", 0))
    start_step = int((resume_state or {}).get("global_step", 0))

    context.control_hook.safe_point(
        SafePointType.BEFORE_TRAIN,
        epoch=start_epoch,
        global_step=start_step,
        metrics={},
        state_factory=lambda: resume_state
        or {"initial_checkpoint": True, "epoch": 0, "global_step": 0},
    )

    def epoch_callback(
        epoch: int,
        global_step: int,
        metrics: dict[str, float],
        state_factory: Callable[[], dict[str, Any]],
    ) -> None:
        context.control_hook.safe_point(
            SafePointType.EPOCH,
            epoch=epoch,
            global_step=global_step,
            metrics=metrics,
            state_factory=state_factory,
        )

    result = train_quality_model(
        baseline=baseline,
        params=params,
        device=device,
        resume_state=resume_state,
        epoch_callback=epoch_callback,
    )
    result.update(
        {
            "backend": str(context.job.metadata.get("placement_backend") or "exclusive"),
            "job_id": context.job.job_id,
            "logical_job_id": context.job.metadata.get("logical_job_id"),
            "finished_at": time.time(),
        }
    )
    result_dir = Path(str(params["result_dir"]))
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / f"{context.job.job_id}.json").write_text(json.dumps(result, indent=2))
    return result


def _standalone(spec_path: Path, result_path: Path) -> int:
    spec = json.loads(spec_path.read_text())
    baseline = torch.load(spec["baseline_model_path"], map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = train_quality_model(baseline=baseline, params=spec, device=device)
    result.update(
        {
            "job_id": spec["job_id"],
            "logical_job_id": spec["job_id"],
            "pid": os.getpid(),
            "finished_at": time.time(),
        }
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args(argv)
    return _standalone(Path(args.spec), Path(args.result))


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared helpers for scheduler benchmark trace generation and sweeps."""

from __future__ import annotations

import gc
import importlib.util
from pathlib import Path
import re

try:
    import torch
except ModuleNotFoundError:
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = Path(__file__).resolve().parent

DEFAULT_VRAM_BUDGET_GIB = 28.0
DEFAULT_CACHE_MEMORY_BUDGET_GIB = 6.0
DEFAULT_CACHE_WARM_TOP_K = 4
DEFAULT_CACHE_ENTRY_CAPACITY = 8
DEFAULT_CACHE_MAX_RAM_PERCENT = 0.20
DEFAULT_BINARY_RANGE_UP = 16
DEFAULT_BINARY_RANGE_DOWN = 8
DEFAULT_POWER_OF_TWO_RANGE_UP = 1
DEFAULT_POWER_OF_TWO_RANGE_DOWN = 1

REQUIRED_BENCHMARK_MODULES = (
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("timm", "timm"),
    ("pandas", "pandas"),
    ("PIL", "pillow"),
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib"),
    ("hydra", "hydra-core"),
    ("omegaconf", "omegaconf"),
    ("psutil", "psutil"),
)


def missing_benchmark_modules() -> list[tuple[str, str]]:
    missing: list[tuple[str, str]] = []
    for module_name, package_name in REQUIRED_BENCHMARK_MODULES:
        if importlib.util.find_spec(module_name) is None:
            missing.append((module_name, package_name))
    return missing


def benchmark_dependency_error_message(missing: list[tuple[str, str]]) -> str:
    package_names = ", ".join(sorted({package_name for _, package_name in missing}))
    return (
        "Missing benchmark Python dependencies: "
        f"{package_names}\n"
        "Install them with:\n"
        "  python -m pip install -r scheduler_benchmark_test/requirements.txt\n"
        "If this repo is not installed as a package yet, also run:\n"
        "  python -m pip install -e ."
    )


def assert_benchmark_python_deps() -> None:
    missing = missing_benchmark_modules()
    if missing:
        raise SystemExit(benchmark_dependency_error_message(missing))


def assert_cassava_root(data_root: str | Path) -> Path:
    root = Path(data_root).expanduser().resolve()
    train_csv = root / "train.csv"
    train_images = root / "train_images"
    if not train_csv.exists() or not train_images.exists():
        raise SystemExit(
            "Cassava benchmark dataset not found.\n"
            f"Expected at: {root}\n"
            f"Missing: {'train.csv' if not train_csv.exists() else ''}"
            f"{' and ' if (not train_csv.exists() and not train_images.exists()) else ''}"
            f"{'train_images/' if not train_images.exists() else ''}\n"
            "Set CASSAVA_ROOT to the prepared/public dataset directory before running the benchmark.\n"
            "Example:\n"
            "  export CASSAVA_ROOT=/path/to/cassava-leaf-disease-classification/prepared/public"
        )
    return root


def _sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def model_startpoint_id(model_name: str) -> str:
    return f"benchmark-startpoint:{model_name}"


def model_startpoint_path(startpoints_dir: str | Path, model_name: str) -> Path:
    return Path(startpoints_dir) / f"{_sanitize_name(model_name)}.pt"


def _compress_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    assert_benchmark_python_deps()
    assert torch is not None
    compressed: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        tensor = value.detach().cpu()
        if torch.is_floating_point(tensor):
            tensor = tensor.half()
        compressed[key] = tensor
    return compressed


def ensure_timm_startpoint(
    startpoints_dir: str | Path,
    *,
    model_name: str,
    num_classes: int = 5,
    seed: int = 42,
) -> str:
    """Create one reusable TIMM checkpoint per model if it does not exist yet."""
    assert_benchmark_python_deps()
    assert torch is not None
    path = model_startpoint_path(startpoints_dir, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return str(path.resolve())

    import timm

    torch.manual_seed(seed)
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    checkpoint = {
        "model_name": model_name,
        "num_classes": num_classes,
        "seed": seed,
        "state_dict": _compress_state_dict(model.state_dict()),
    }
    torch.save(checkpoint, path)
    del checkpoint
    del model
    gc.collect()
    return str(path.resolve())

"""Generate short pair traces for packing-composition sensitivity checks.

Creates three tiny traces:
- same_model_pair.jsonl: same exact model twice
- same_arch_pair.jsonl: same architecture family, different models
- cross_arch_pair.jsonl: different architecture families
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from benchmark_support import (
    assert_benchmark_python_deps,
    assert_cassava_root,
    ensure_timm_startpoint,
    model_startpoint_id,
)

OUT_DIR = Path(os.environ.get("OUT_DIR", os.path.dirname(__file__))) / "arch_sensitivity_inputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CODE_DIR = OUT_DIR / "codes"
CODE_DIR.mkdir(parents=True, exist_ok=True)
STARTPOINT_DIR = OUT_DIR / "startpoints"
STARTPOINT_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = os.environ.get(
    "CASSAVA_ROOT",
    "/home/downeyflyfan/Research_Projects/AI/Datasets/mle-bench-data/cassava-leaf-disease-classification/prepared/public",
)
SUBSET_SIZE = int(os.environ.get("ARCH_SENSITIVITY_SUBSET_SIZE", "1000"))
EPOCHS = int(os.environ.get("ARCH_SENSITIVITY_EPOCHS", "1"))
SEED = int(os.environ.get("ARCH_SENSITIVITY_SEED", "42"))

TRACE_DEFS = {
    "same_model_pair": [
        ("convnext_base", 16, "CNN", "draft"),
        ("convnext_base", 16, "CNN", "improve"),
    ],
    "same_arch_pair": [
        ("convnext_base", 16, "CNN", "draft"),
        ("efficientnet_b3", 16, "CNN", "improve"),
    ],
    "cross_arch_pair": [
        ("convnext_base", 16, "CNN", "draft"),
        ("vit_base_patch16_224", 16, "Transformer", "improve"),
    ],
}

VRAM_AT_MAX_BS = {
    "convnext_base": (48, 11763),
    "efficientnet_b3": (48, 8024),
    "vit_base_patch16_224": (32, 4721),
}

SCRIPT_TEMPLATE = '''import os, time, json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import timm

DATA_ROOT = "{data_root}"
IMG_DIR = f"{{DATA_ROOT}}/train_images"
SUBSET = {subset}
BS = {bs}
EPOCHS = {epochs}
MODEL_NAME = "{model}"
SEED = {seed}
STARTPOINT_PATH = r"{startpoint_path}"

torch.manual_seed(SEED)
df = pd.read_csv(f"{{DATA_ROOT}}/train.csv").sample(n=SUBSET, random_state=SEED).reset_index(drop=True)


class CassavaDS(Dataset):
    def __init__(self, df, img_dir, tfm):
        self.df = df; self.img_dir = img_dir; self.tfm = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(os.path.join(self.img_dir, row["image_id"])).convert("RGB")
        return self.tfm(img), int(row["label"])


tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dl = DataLoader(CassavaDS(df, IMG_DIR, tfm), batch_size=BS, shuffle=True, num_workers=0, pin_memory=True)
device = torch.device("cuda")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=5).to(device)
startpoint = torch.load(STARTPOINT_PATH, map_location="cpu")
state_dict = startpoint["state_dict"] if isinstance(startpoint, dict) and "state_dict" in startpoint else startpoint
model.load_state_dict(state_dict, strict=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

t0 = time.time()
last_loss = 0.0
for _epoch in range(EPOCHS):
    for x, y in dl:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
torch.cuda.synchronize()
elapsed = time.time() - t0
peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
Path("metric.json").write_text(json.dumps({{
    "metric": last_loss,
    "elapsed_s": round(elapsed, 2),
    "peak_vram_mib": round(peak_mb, 1),
    "model": MODEL_NAME,
    "bs": BS,
}}))
Path("submission").mkdir(exist_ok=True)
Path("submission/submission.csv").write_text("image_id,label\\nfoo,0\\n")
print(f"DONE {{MODEL_NAME}} t={{elapsed:.1f}}s vram={{peak_mb:.0f}}MB")
'''


def estimate_vram_mb(model_name: str, bs: int) -> int:
    base_bs, base_mb = VRAM_AT_MAX_BS[model_name]
    overhead = base_mb * 0.3
    activation = base_mb * 0.7
    return int(overhead + activation * (bs / base_bs))


def write_trace(trace_name: str, steps: list[tuple[str, int, str, str]]) -> None:
    entries = []
    trace_dir = CODE_DIR / trace_name
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = OUT_DIR / f"{trace_name}.jsonl"
    for idx, (model, bs, klass, stage) in enumerate(steps):
        entry_seed = SEED + idx
        startpoint_path = ensure_timm_startpoint(
            STARTPOINT_DIR,
            model_name=model,
            num_classes=5,
            seed=SEED,
        )
        code = SCRIPT_TEMPLATE.format(
            data_root=DATA_ROOT,
            subset=SUBSET_SIZE,
            bs=bs,
            epochs=EPOCHS,
            model=model,
            seed=entry_seed,
            startpoint_path=startpoint_path,
        )
        code_path = trace_dir / f"step_{idx:03d}.py"
        code_path.write_text(code, encoding="utf-8")
        entries.append(
            {
                "step_idx": idx,
                "agent_used": stage,
                "child_id": f"{trace_name}_node_{idx}",
                "branch_id": trace_name,
                "code": code,
                "exec_submit_at": 0.0,
                "estimated_vram_mb": estimate_vram_mb(model, bs),
                "model_class": klass,
                "model_name": model,
                "bs": bs,
                "max_bs": bs * 2,
                "subset": SUBSET_SIZE,
                "epochs": EPOCHS,
                "dataset_seed": entry_seed,
                "data_root": DATA_ROOT,
                "learning_rate": 1e-3,
                "startpoint_id": model_startpoint_id(model),
                "startpoint_path": startpoint_path,
                "code_path": str(code_path.resolve()),
                "exec_duration_s_estimate": 60.0,
                "deferred": True,
            }
        )
    with trace_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")
    print(f"Wrote {trace_path}")


def main() -> None:
    assert_benchmark_python_deps()
    assert_cassava_root(DATA_ROOT)
    for trace_name, steps in TRACE_DEFS.items():
        write_trace(trace_name, steps)


if __name__ == "__main__":
    main()

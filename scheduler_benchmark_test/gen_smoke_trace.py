"""Smoke trace: 4 entries with enough queued work to exercise pack-3/pack-4 scenarios.

Used by smoke_run.sh to verify scripts work before full sweep.
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

SMOKE_DIR = Path(os.environ.get("SMOKE_DIR", "/tmp/smoke_unbiased"))
CODE_DIR = SMOKE_DIR / "codes"
STARTPOINT_DIR = SMOKE_DIR / "startpoints"
SMOKE_DIR.mkdir(parents=True, exist_ok=True)
CODE_DIR.mkdir(parents=True, exist_ok=True)
STARTPOINT_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = os.environ.get(
    "CASSAVA_ROOT",
    "/home/downeyflyfan/Research_Projects/AI/Datasets/mle-bench-data/cassava-leaf-disease-classification/prepared/public",
)

SMOKE_STEPS = [
    ("resnet101", 16, "CNN", "draft"),
    ("resnet101", 16, "CNN", "improve"),
    ("vit_base_patch16_224", 16, "Transformer", "improve"),
    ("mixer_b16_224", 24, "Mixer", "improve"),
]

TPL = '''import os, time, json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import timm

DATA_ROOT = "{data_root}"
df = pd.read_csv(f"{{DATA_ROOT}}/train.csv").sample(n=500, random_state=42).reset_index(drop=True)


class DS(Dataset):
    def __init__(self, df, t):
        self.df = df; self.t = t

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(os.path.join(f"{{DATA_ROOT}}/train_images", r["image_id"])).convert("RGB")
        return self.t(img), int(r["label"])


tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

dl = DataLoader(DS(df, tfm), batch_size={bs}, shuffle=True, num_workers=0, pin_memory=True)
device = torch.device("cuda")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
model = timm.create_model("{model}", pretrained=False, num_classes=5).to(device)
startpoint = torch.load(r"{startpoint_path}", map_location="cpu")
state_dict = startpoint["state_dict"] if isinstance(startpoint, dict) and "state_dict" in startpoint else startpoint
model.load_state_dict(state_dict, strict=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

t0 = time.time()
last = 0.0
for x, y in dl:
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    opt.zero_grad()
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    opt.step()
    last = float(loss.item())
torch.cuda.synchronize()
elapsed = time.time() - t0
peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
Path("metric.json").write_text(json.dumps({{
    "metric": last, "elapsed_s": round(elapsed, 2), "peak_vram_mib": round(peak_mb, 1),
    "model": "{model}", "bs": {bs},
}}))
Path("submission").mkdir(exist_ok=True)
Path("submission/submission.csv").write_text("image_id,label\\nfoo,0\\n")
print(f"DONE {model} t={{elapsed:.1f}}s vram={{peak_mb:.0f}}MB")
'''


def main():
    assert_benchmark_python_deps()
    assert_cassava_root(DATA_ROOT)
    entries = []
    for idx, (model, bs, klass, stage) in enumerate(SMOKE_STEPS):
        startpoint_path = ensure_timm_startpoint(
            STARTPOINT_DIR,
            model_name=model,
            num_classes=5,
            seed=42,
        )
        code = TPL.format(data_root=DATA_ROOT, bs=bs, model=model, startpoint_path=startpoint_path)
        cp = CODE_DIR / f"step_{idx:03d}.py"
        cp.write_text(code, encoding="utf-8")
        entries.append({
            "step_idx": idx, "agent_used": stage, "child_id": f"n{idx}",
            "branch_id": f"b{idx}", "code": code, "exec_submit_at": idx * 2.0,
            "estimated_vram_mb": 3500, "model_class": klass, "model_name": model,
            "bs": bs, "max_bs": bs * 2, "deferred": True,
            "subset": 500, "epochs": 1, "dataset_seed": 42 + idx, "data_root": DATA_ROOT,
            "learning_rate": 1e-3, "startpoint_id": model_startpoint_id(model),
            "startpoint_path": startpoint_path, "code_path": str(cp.resolve()),
        })
    out = SMOKE_DIR / "smoke_trace.jsonl"
    with out.open("w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    print(f"Wrote {len(entries)} smoke entries to {out}")


if __name__ == "__main__":
    main()

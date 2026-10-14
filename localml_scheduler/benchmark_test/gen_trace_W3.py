"""Generate W3 trace: balanced arch + cassava + heavy models.

20 entries: 7 CNN + 7 Transformer + 6 Mixer.
Each entry has full Python training script using real cassava data.

Outputs:
  - workload_trace_W3.jsonl  (one JSON line per entry)
  - replay_codes_W3/step_NNN.py  (one runnable script per entry)

Configurable via env vars:
  - CASSAVA_ROOT (default = ours)
  - SUBSET_SIZE (default 4000) — bump or shrink to control per-job time
  - EPOCHS (default 2)
  - SEED (default 42)
  - OUT_DIR (default = same dir as this script)
"""
from __future__ import annotations
import json
import os
import random
from pathlib import Path

from benchmark_support import (
    assert_benchmark_python_deps,
    assert_cassava_root,
    ensure_timm_startpoint,
    model_startpoint_id,
)

OUT_DIR = Path(os.environ.get("OUT_DIR", os.path.dirname(__file__)))
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRACE_PATH = OUT_DIR / "workload_trace_W3.jsonl"
CODE_DIR = OUT_DIR / "replay_codes_W3"
CODE_DIR.mkdir(parents=True, exist_ok=True)
STARTPOINT_DIR = OUT_DIR / "startpoints"
STARTPOINT_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = os.environ.get(
    "CASSAVA_ROOT",
    "/home/downeyflyfan/Research_Projects/AI/Datasets/mle-bench-data/cassava-leaf-disease-classification/prepared/public",
)
SUBSET_SIZE = int(os.environ.get("SUBSET_SIZE", "4000"))
EPOCHS = int(os.environ.get("EPOCHS", "2"))
SEED = int(os.environ.get("SEED", "42"))

CNN_MODELS = [
    ("convnext_base", 48), ("convnext_base", 48), ("convnext_base", 48), ("convnext_base", 48),
    ("efficientnet_b3", 48), ("efficientnet_b3", 48), ("resnet101", 32),
]
TRANS_MODELS = [
    ("vit_base_patch16_224", 32), ("vit_base_patch16_224", 32),
    ("vit_base_patch16_224", 32), ("vit_base_patch16_224", 32),
    ("swin_small_patch4_window7_224", 32), ("swin_small_patch4_window7_224", 32),
    ("deit_base_patch16_224", 32),
]
MIXER_MODELS = [
    ("mixer_b16_224", 48), ("mixer_b16_224", 48), ("mixer_b16_224", 48),
    ("gmlp_s16_224", 48), ("gmlp_s16_224", 48), ("resmlp_24_224", 48),
]

STAGES = (["draft"] * 3 + ["improve"] * 10 + ["debug"] * 2 + ["evolution"] * 3 + ["fusion"] * 2)
BS_BUCKETS = [16, 24, 32, 48]

# Conservative per-model sizing fit from an older smoke run.
# The replay harness now defaults to a 28 GiB safe budget for RTX 5090,
# so treat these values as ordering hints rather than exact 5090 measurements.
VRAM_AT_MAX_BS = {
    "convnext_base": (48, 11763), "efficientnet_b3": (48, 8024),
    "resnet101": (32, 4442), "vit_base_patch16_224": (32, 4721),
    "swin_small_patch4_window7_224": (32, 5830), "deit_base_patch16_224": (32, 4721),
    "mixer_b16_224": (48, 6223), "gmlp_s16_224": (48, 6771), "resmlp_24_224": (48, 5199),
}


def estimate_vram_mb(model_name: str, bs: int) -> int:
    base_bs, base_mb = VRAM_AT_MAX_BS[model_name]
    overhead = base_mb * 0.3
    activation = base_mb * 0.7
    return int(overhead + activation * (bs / base_bs))


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

ds = CassavaDS(df, IMG_DIR, tfm)
# num_workers=0 (Python 3.14 + DataLoader IPC issue across subprocesses)
dl = DataLoader(ds, batch_size=BS, shuffle=True, num_workers=0, pin_memory=True)

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
n_samples = 0
last_loss = 0.0
for epoch in range(EPOCHS):
    for x, y in dl:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
        n_samples += x.shape[0]
torch.cuda.synchronize()
elapsed = time.time() - t0
peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

Path("metric.json").write_text(json.dumps({{
    "metric": last_loss,
    "elapsed_s": round(elapsed, 2),
    "peak_vram_mib": round(peak_mb, 1),
    "model": MODEL_NAME,
    "bs": BS,
    "epochs": EPOCHS,
    "subset": SUBSET,
    "throughput_sps": round(n_samples / elapsed, 1),
}}))
Path("submission").mkdir(exist_ok=True)
Path("submission/submission.csv").write_text("image_id,label\\nfoo,0\\n")
print(f"DONE {{MODEL_NAME}} bs={{BS}} sub={{SUBSET}} ep={{EPOCHS}} loss={{last_loss:.3f}} t={{elapsed:.1f}}s vram={{peak_mb:.0f}}MB")
'''


def gen_trace():
    assert_benchmark_python_deps()
    assert_cassava_root(DATA_ROOT)
    rng = random.Random(SEED)
    pool = []
    for m, b in CNN_MODELS:
        pool.append((m, b, "CNN"))
    for m, b in TRANS_MODELS:
        pool.append((m, b, "Transformer"))
    for m, b in MIXER_MODELS:
        pool.append((m, b, "Mixer"))
    rng.shuffle(pool)
    stages = list(STAGES)
    rng.shuffle(stages)

    entries = []
    for idx, ((model, max_bs, klass), stage) in enumerate(zip(pool, stages, strict=True)):
        bs_options = [b for b in BS_BUCKETS if b <= max_bs]
        bs = rng.choice(bs_options)
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
        code_path = CODE_DIR / f"step_{idx:03d}.py"
        code_path.write_text(code, encoding="utf-8")
        vram_est = estimate_vram_mb(model, bs)
        entries.append({
            "step_idx": idx,
            "agent_used": stage,
            "child_id": f"node_{idx}",
            "branch_id": f"b{idx % 3}",
            "code": code,
            "exec_submit_at": idx * 5.0,
            "estimated_vram_mb": vram_est,
            "model_class": klass,
            "model_name": model,
            "bs": bs,
            "max_bs": max_bs,
            "subset": SUBSET_SIZE,
            "epochs": EPOCHS,
            "dataset_seed": entry_seed,
            "data_root": DATA_ROOT,
            "learning_rate": 1e-3,
            "startpoint_id": model_startpoint_id(model),
            "startpoint_path": startpoint_path,
            "code_path": str(code_path.resolve()),
            "exec_duration_s_estimate": 100.0,
            "llm_calls": [{"phase": "code_gen", "duration_s": 5.0}],
            "deferred": True,
        })

    with TRACE_PATH.open("w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    from collections import Counter
    cls = Counter(e["model_class"] for e in entries)
    bs_d = Counter(e["bs"] for e in entries)
    md = Counter(e["model_name"] for e in entries)
    sg = Counter(e["agent_used"] for e in entries)
    print(f"Wrote {len(entries)} entries -> {TRACE_PATH}")
    print(f"Codes -> {CODE_DIR}/step_NNN.py")
    print(f"Class:  {dict(cls)}")
    print(f"Models: {dict(md)}")
    print(f"BS:     {dict(bs_d)}")
    print(f"Stage:  {dict(sg)}")


if __name__ == "__main__":
    gen_trace()

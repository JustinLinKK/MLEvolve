import os, time, json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import timm

DATA_ROOT = "/home/vscode/datasets/cassava-leaf-disease-classification/prepared/public"
IMG_DIR = f"{DATA_ROOT}/train_images"
SUBSET = 4000
BS = 16
EPOCHS = 2
MODEL_NAME = "vit_base_patch16_224"
SEED = 58
STARTPOINT_PATH = r"/workspaces/MLEvolve/scheduler_benchmark_test/startpoints/vit_base_patch16_224.pt"

torch.manual_seed(SEED)
df = pd.read_csv(f"{DATA_ROOT}/train.csv").sample(n=SUBSET, random_state=SEED).reset_index(drop=True)


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

Path("metric.json").write_text(json.dumps({
    "metric": last_loss,
    "elapsed_s": round(elapsed, 2),
    "peak_vram_mib": round(peak_mb, 1),
    "model": MODEL_NAME,
    "bs": BS,
    "epochs": EPOCHS,
    "subset": SUBSET,
    "throughput_sps": round(n_samples / elapsed, 1),
}))
Path("submission").mkdir(exist_ok=True)
Path("submission/submission.csv").write_text("image_id,label\nfoo,0\n")
print(f"DONE {MODEL_NAME} bs={BS} sub={SUBSET} ep={EPOCHS} loss={last_loss:.3f} t={elapsed:.1f}s vram={peak_mb:.0f}MB")

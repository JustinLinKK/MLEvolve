import os, sys, math, random, warnings, time

# Set before importing torch for better memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Data loading & split ----------------
train_df = pd.read_csv("input/train.csv")
test_df = pd.read_csv("input/test.csv")
FEATURE_COLS = [
    "Face",
    "Eyes",
    "Near",
    "Group",
    "Occular_Protection",
    "Accessory",
    "Action",
    "Collage",
    "Info",
    "Human",
    "Idiomatic",
    "Blur",
]
# Use actual columns present
FEATURE_COLS = [
    c
    for c in [
        "Face",
        "Eyes",
        "Near",
        "Group",
        "Occlusion",
        "Accessory",
        "Action",
        "Collage",
        "Info",
        "Human",
        "Blur",
        "Subject Focus",
    ]
    if c in train_df.columns
]
TARGET = "Pawpularity"

# Stratified decile split (reproducible)
train_df["decile"] = pd.qcut(train_df[TARGET], q=10, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
tr_idx, va_idx = next(iter(skf.split(train_df, train_df["decile"])))
val_mask = np.zeros(len(train_df), dtype=bool)
val_mask[va_idx] = True
train_split = train_df[~val_mask].reset_index(drop=True)
val_split = train_df[val_mask].reset_index(drop=True)
print(f"Train split: {len(train_split)}, Val split: {len(val_split)}")

IMG_SIZE = 384
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_transforms(augment=True):
    tfs = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if augment:
        tfs.insert(1, transforms.RandomHorizontalFlip(p=0.5))
        tfs.insert(
            2, transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
        )
    return transforms.Compose(tfs)


class PetDataset(Dataset):
    def __init__(self, df, img_dir, augment=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tf = make_transforms(augment)
        self.has_target = TARGET in df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_path = os.path.join(self.img_dir, f"{row['Id']}.jpg")
        from PIL import Image

        img = Image.open(img_path).convert("RGB")
        img = self.tf(img)
        feat_vals = np.asarray(row[FEATURE_COLS].values, dtype=np.float32)
        feats = torch.from_numpy(feat_vals)
        if self.has_target:
            tgt = torch.tensor(float(row[TARGET]), dtype=torch.float32)
            return img, feats, tgt
        return img, feats


def collate_fn(batch):
    imgs, feats, tgts = zip(*batch)
    return torch.stack(imgs), torch.stack(feats), torch.stack(tgts)


def collate_fn_no_tgt(batch):
    imgs, feats = zip(*batch)
    return torch.stack(imgs), torch.stack(feats)


train_ds = PetDataset(train_split, "input/train", augment=True)
val_ds = PetDataset(val_split, "input/train", augment=False)
test_ds = PetDataset(test_df, "input/test", augment=False)
BATCH = 4  # Reduced from 16 to avoid OOM
GRAD_ACCUM_STEPS = 4  # Effective batch size = 4 * 4 = 16
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True,
    collate_fn=collate_fn,
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True,
    collate_fn=collate_fn_no_tgt,
)


# ---------------- Model (ViT-Small/384 + meta MLP) ----------------
class PawpularityModel(nn.Module):
    def __init__(self, feat_dim=384, n_meta=None):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch16_384", pretrained=True, num_classes=0
        )
        self.meta_mlp = nn.Sequential(
            nn.Linear(n_meta, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(feat_dim + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, images, meta):
        x = self.backbone(images)
        m = self.meta_mlp(meta)
        z = torch.cat([x, m], dim=1)
        return self.head(z).squeeze(-1)


model = PawpularityModel(feat_dim=384, n_meta=len(FEATURE_COLS)).to(DEVICE)

# Param groups
backbone_params = list(model.backbone.parameters())
other_params = [
    p for n, p in model.named_parameters() if not any(p is bp for bp in backbone_params)
]
optimizer = torch.optim.AdamW(
    [{"params": backbone_params, "lr": 3e-5}, {"params": other_params, "lr": 5e-4}],
    weight_decay=1e-4,
)

EPOCHS = 20
total_steps = EPOCHS * len(train_loader) // GRAD_ACCUM_STEPS
warmup_steps = total_steps // 5


def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

loss_fn = nn.MSELoss()
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
best_rmse = float("inf")
patience = 10
bad_epochs = 0
global_step = 0


def evaluate(loader):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for imgs, feats, tgt in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            feats = feats.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                p = model(imgs, feats)
            preds.append(p.float().cpu().numpy())
            trues.append(tgt.cpu().numpy())
    preds = np.clip(np.concatenate(preds), 1, 100)
    trues = np.concatenate(trues)
    return mean_squared_error(trues, preds, squared=False)


for epoch in range(EPOCHS):
    model.train()
    running = 0.0
    optimizer.zero_grad()
    for b, (imgs, feats, tgt) in enumerate(train_loader):
        imgs = imgs.to(DEVICE, non_blocking=True)
        feats = feats.to(DEVICE, non_blocking=True)
        tgt = tgt.to(DEVICE, non_blocking=True)
        # mild mixup
        if random.random() < 0.5:
            lam = float(np.random.beta(0.1, 0.1))
            idx = torch.randperm(imgs.size(0), device=DEVICE)
            imgs = lam * imgs + (1.0 - lam) * imgs[idx]
            feats = lam * feats + (1.0 - lam) * feats[idx]
            tgt = lam * tgt + (1.0 - lam) * tgt[idx]

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            out = model(imgs, feats)
            loss = loss_fn(out, tgt) / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()
        running += loss.item() * GRAD_ACCUM_STEPS

        if (b + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

    train_loss = running / len(train_loader)
    val_rmse = evaluate(val_loader)
    print(
        f"Epoch {epoch+1}/{EPOCHS} | train_loss={train_loss:.4f} | val_rmse={val_rmse:.4f}"
    )
    if val_rmse < best_rmse - 1e-4:
        best_rmse = val_rmse
        bad_epochs = 0
        torch.save(model.state_dict(), "./working/vit_best.pth")
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print("Early stopping")
            break
    # Clear cache periodically to prevent fragmentation
    if (epoch + 1) % 5 == 0:
        torch.cuda.empty_cache()

model.load_state_dict(torch.load("./working/vit_best.pth"))
model.eval()
vp = []
vt = []
with torch.no_grad():
    for imgs, feats, tgt in val_loader:
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            p = model(imgs.to(DEVICE), feats.to(DEVICE))
        vp.append(p.float().cpu().numpy())
        vt.append(tgt.cpu().numpy())
val_pred_vit = np.clip(np.concatenate(vp), 1, 100)
val_true = np.concatenate(vt)
vit_val_rmse = mean_squared_error(val_true, val_pred_vit, squared=False)
print(f"ViT fine-tuned val RMSE: {vit_val_rmse:.4f}")

# Test inference for ViT
tp = []
with torch.no_grad():
    for imgs, feats in test_loader:
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            p = model(imgs.to(DEVICE), feats.to(DEVICE))
        tp.append(p.float().cpu().numpy())
test_pred_vit = np.clip(np.concatenate(tp), 1, 100)
torch.cuda.empty_cache()

# ---------------- Stage 2: LightGBM on frozen ViT-B/16 embeddings ----------------
EMB_SIZE = 224
emb_model = timm.create_model(
    "vit_base_patch16_224", pretrained=True, num_classes=0
).to(DEVICE)
emb_model.eval()
emb_tf = transforms.Compose(
    [
        transforms.Resize((EMB_SIZE, EMB_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


class EmbDataset(Dataset):
    def __init__(self, df, img_dir):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        from PIL import Image

        img = Image.open(os.path.join(self.img_dir, f"{row['Id']}.jpg")).convert("RGB")
        return emb_tf(img)


def extract_embeddings(df, img_dir):
    ds = EmbDataset(df, img_dir)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    embs = []
    with torch.no_grad():
        for imgs in dl:
            embs.append(emb_model(imgs.to(DEVICE)).cpu().numpy())
    return np.concatenate(embs)


print("Extracting embeddings...")
train_emb = extract_embeddings(train_split, "input/train")
torch.cuda.empty_cache()
val_emb = extract_embeddings(val_split, "input/train")
torch.cuda.empty_cache()
test_emb = extract_embeddings(test_df, "input/test")
torch.cuda.empty_cache()
pca = PCA(n_components=128, random_state=SEED)
train_emb_p = pca.fit_transform(train_emb)
val_emb_p = pca.transform(val_emb)
test_emb_p = pca.transform(test_emb)
X_tr = np.hstack([train_emb_p, train_split[FEATURE_COLS].values.astype(np.float32)])
X_va = np.hstack([val_emb_p, val_split[FEATURE_COLS].values.astype(np.float32)])
X_te = np.hstack([test_emb_p, test_df[FEATURE_COLS].values.astype(np.float32)])
y_tr = train_split[TARGET].values.astype(np.float32)
y_va = val_split[TARGET].values.astype(np.float32)

lgbm = lgb.LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=SEED,
    verbose=-1,
)
lgbm.fit(
    X_tr,
    y_tr,
    eval_set=[(X_va, y_va)],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
)
val_pred_lgb = np.clip(lgbm.predict(X_va), 1, 100)
test_pred_lgb = np.clip(lgbm.predict(X_te), 1, 100)
lgb_val_rmse = mean_squared_error(y_va, val_pred_lgb, squared=False)
print(f"LightGBM val RMSE: {lgb_val_rmse:.4f}")

# ---------------- Ensemble blending on validation RMSE ----------------
best_w, best_blend_rmse = 0.5, float("inf")
for w in np.arange(0.0, 1.01, 0.05):
    blend = w * val_pred_vit + (1 - w) * val_pred_lgb
    rmse = mean_squared_error(val_true, blend, squared=False)
    if rmse < best_blend_rmse:
        best_blend_rmse = rmse
        best_w = w
print(
    f"Best ensemble weight (ViT): {best_w:.2f}, blended val RMSE: {best_blend_rmse:.4f}"
)

final_val_pred = best_w * val_pred_vit + (1 - best_w) * val_pred_lgb
final_test_pred = best_w * test_pred_vit + (1 - best_w) * test_pred_lgb
final_val_rmse = mean_squared_error(val_true, final_val_pred, squared=False)

sub = pd.DataFrame({"Id": test_df["Id"].values, "Pawpularity": final_test_pred})
sub.to_csv("./submission/submission.csv", index=False)
print(f"Submission rows: {len(sub)}")
print(f"Final Validation Score: {final_val_rmse:.4f}")
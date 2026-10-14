import os

os.makedirs("submission", exist_ok=True)
os.makedirs("working", exist_ok=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2  # effective batch = 32
EPOCHS = 15
PATIENCE = 5
LR_BACKBONE = 1e-5
LR_HEAD = 5e-4
WEIGHT_DECAY = 1e-4
MIXUP_ALPHA = 0.3
NUM_WORKERS = 4

# Use CUDA if available, no strict memory threshold
def _get_device():
    if torch.cuda.is_available():
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            print(f"CUDA available. Free: {free_mem/1e9:.2f} GB / Total: {total_mem/1e9:.2f} GB")
            return torch.device("cuda")
        except Exception as e:
            print(f"CUDA check failed: {e}, falling back to CPU.")
            return torch.device("cpu")
    return torch.device("cpu")

DEVICE = _get_device()
USE_AMP = DEVICE.type == "cuda"
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
print(f"Using device: {DEVICE}, AMP: {USE_AMP}")
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Load data ───────────────────────────────────────────────────────────
train_df = pd.read_csv("input/train.csv")
test_df = pd.read_csv("input/test.csv")

META_COLS = [
    "Accessory",
    "Action",
    "Blur",
    "Collage",
    "Eyes",
    "Face",
    "Group",
    "Human",
    "Info",
    "Near",
    "Occlusion",
    "Subject Focus",
]

print(f"Train: {len(train_df)}, Test: {len(test_df)}")
print(
    f"Target stats: mean={train_df['Pawpularity'].mean():.2f}, std={train_df['Pawpularity'].std():.2f}"
)

# ── Reproducible stratified split ───────────────────────────────────────
# Use deciles of Pawpularity for stratification to keep target distribution
# balanced between the training and validation subsets.
train_df["decile"] = pd.qcut(
    train_df["Pawpularity"], q=10, labels=False, duplicates="drop"
)
train_split, val_split = train_test_split(
    train_df,
    test_size=0.2,
    random_state=SEED,
    stratify=train_df["decile"],
)
train_split = train_split.reset_index(drop=True)
val_split = val_split.reset_index(drop=True)

print(f"Train split: {len(train_split)}, Val split: {len(val_split)}")


# ── Dataset ─────────────────────────────────────────────────────────────
class PetDataset(Dataset):
    def __init__(self, df, img_dir, augment=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.augment = augment
        self.has_target = "Pawpularity" in self.df.columns
        base_t = [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        if augment:
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
                    ),
                    transforms.RandomAffine(
                        degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)
                    ),
                    transforms.RandomGrayscale(p=0.1),
                ]
                + base_t
            )
        else:
            self.transform = transforms.Compose(base_t)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_path = os.path.join(self.img_dir, f"{row['Id']}.jpg")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        meta = row[META_COLS].values.astype(np.float32)
        if self.has_target:
            target = float(row["Pawpularity"]) / 100.0  # normalize to [0,1]
        else:
            target = 0.0  # placeholder; not used during test inference
        return img, meta, target


train_ds = PetDataset(train_split, "input/train", augment=True)
val_ds = PetDataset(val_split, "input/train", augment=False)
test_ds = PetDataset(test_df, "input/test", augment=False)

VAL_BATCH_SIZE = 32  # larger batch for inference (no gradients needed)
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)
val_loader = DataLoader(
    val_ds,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)
test_loader = DataLoader(
    test_ds,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)


# ── Model ───────────────────────────────────────────────────────────────
class PawpularityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=0
        )
        dim = self.backbone.num_features  # 768

        self.img_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.meta_mlp = nn.Sequential(
            nn.Linear(len(META_COLS), 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        fused_dim = 256 + 64  # 320
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    @property
    def backbone_params(self):
        return list(self.backbone.parameters())

    @property
    def head_params(self):
        return (list(self.img_proj.parameters()) +
                list(self.meta_mlp.parameters()) +
                list(self.head.parameters()))

    def forward(self, images, meta):
        f = self.backbone(images)  # (B, 1280)
        f = self.img_proj(f)       # (B, 256)
        m = self.meta_mlp(meta)    # (B, 64)
        z = torch.cat([f, m], dim=1)  # (B, 320)
        out = self.head(z).squeeze(-1)  # (B,)
        return out


model = PawpularityModel().to(DEVICE)
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
    # Enable gradient checkpointing on backbone to save memory
    try:
        model.backbone.gradient_checkpointing = True
        print("Gradient checkpointing enabled on backbone.")
    except Exception:
        pass

    # If using CUDA, enable TF32 for faster matmul
    if DEVICE.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# Differential learning rates
backbone_params = model.backbone_params
head_params = model.head_params
optimizer = optim.AdamW(
    [
        {"params": backbone_params, "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY},
        {"params": head_params, "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY},
    ]
)

total_steps = EPOCHS * len(train_loader) // GRAD_ACCUM_STEPS
GRAD_CLIP_NORM = 1.0


def lr_lambda(step):
    # Warmup for first 5% then cosine decay
    warmup_steps = int(0.05 * total_steps)
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1 + np.cos(np.pi * progress))


scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Mixup helper ────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=MIXUP_ALPHA):
    # Use a plain Python float for lam so it does NOT upcast the float32
    # tensors to float64 (which breaks loss.backward with a dtype error).
    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(DEVICE)
    one_minus_lam = 1.0 - lam
    mixed_x = lam * x + one_minus_lam * x[index]
    mixed_y = lam * y + one_minus_lam * y[index]
    # Explicitly keep the exact input dtype (float32) to be safe.
    return mixed_x.to(x.dtype), mixed_y.to(y.dtype)


# ── Training loop ───────────────────────────────────────────────────────
best_rmse = float("inf")
best_state = None
patience_counter = 0
best_epoch = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()
    for batch_idx, (imgs, metas, targets) in enumerate(train_loader):
        imgs = imgs.to(DEVICE, non_blocking=True)
        metas = metas.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True).to(torch.float32)

        # Apply mixup
        imgs, targets = mixup_data(imgs, targets)

        # Use mixed precision for forward pass when on CUDA
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            preds = model(imgs, metas)
            loss = nn.MSELoss()(preds, targets) / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        running_loss += loss.item() * GRAD_ACCUM_STEPS
        n_batches += 1

        # Step optimizer every GRAD_ACCUM_STEPS batches
        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

    # Handle remaining batches if not divisible
    if n_batches % GRAD_ACCUM_STEPS != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

    avg_train_loss = running_loss / max(n_batches, 1)
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # Validation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, metas, targets in val_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            metas = metas.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                preds = model(imgs, metas)
            all_preds.append(preds.float().cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    val_rmse = np.sqrt(mean_squared_error(all_targets * 100, all_preds * 100))
    print(
        f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val RMSE: {val_rmse:.4f}"
    )

    if val_rmse < best_rmse:
        best_rmse = val_rmse
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epoch + 1
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

# ── TTA prediction function ─────────────────────────────────────────────
def tta_predict(model, df, img_dir, device, n_views=5):
    """Generate TTA-augmented predictions by averaging over multiple views."""
    model.eval()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    base_resize = transforms.Resize((IMG_SIZE, IMG_SIZE))
    to_tensor = transforms.ToTensor()

    # Pre-compute all images for each view to avoid re-reading files
    def load_all_images(transform_fn):
        images = []
        for i in range(len(df)):
            row = df.iloc[i]
            img_path = os.path.join(img_dir, f"{row['Id']}.jpg")
            img = Image.open(img_path).convert("RGB")
            img = transform_fn(img)
            images.append(img)
        return images

    all_preds = np.zeros(len(df))
    infer_batch_size = 8

    for view_idx in range(n_views):
        if view_idx == 0:
            tf = transforms.Compose([base_resize, to_tensor, normalize])
        elif view_idx == 1:
            tf = transforms.Compose([base_resize, transforms.Lambda(lambda x: __import__('torchvision.transforms.functional', fromlist=['hflip']).hflip(x)), to_tensor, normalize])
        elif view_idx == 2:
            tf = transforms.Compose([base_resize, transforms.Lambda(lambda x: __import__('torchvision.transforms.functional', fromlist=['rotate']).rotate(x, 5)), to_tensor, normalize])
        elif view_idx == 3:
            tf = transforms.Compose([base_resize, transforms.Lambda(lambda x: __import__('torchvision.transforms.functional', fromlist=['rotate']).rotate(x, -5)), to_tensor, normalize])
        else:
            tf = transforms.Compose([base_resize, to_tensor, normalize])

        # Load all images for this view
        batch_imgs_list = []
        for i in range(len(df)):
            row = df.iloc[i]
            img_path = os.path.join(img_dir, f"{row['Id']}.jpg")
            img = Image.open(img_path).convert("RGB")
            img = tf(img)
            batch_imgs_list.append(img)

        preds = []
        for start in range(0, len(df), infer_batch_size):
            end = min(start + infer_batch_size, len(df))
            imgs = torch.stack(batch_imgs_list[start:end]).to(device, non_blocking=True)
            metas = torch.tensor(
                np.stack([df.iloc[i][META_COLS].values.astype(np.float32) for i in range(start, end)])
            ).to(device, non_blocking=True)

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    out = model(imgs, metas)
            preds.append(out.float().cpu().numpy())

            if device.type == "cuda":
                del imgs, metas
                torch.cuda.empty_cache()

        view_preds = np.concatenate(preds)
        all_preds += view_preds

    all_preds /= n_views
    return all_preds


# ── Restore best model and compute TTA validation score ─────────────────
model.load_state_dict(best_state)
model.to(DEVICE)
model.eval()

# Use fewer TTA views for speed (3 views instead of 5)
val_preds_tta = tta_predict(model, val_split, "input/train", DEVICE, n_views=3)
val_targets_arr = np.array(val_split["Pawpularity"].values)
final_rmse = np.sqrt(mean_squared_error(val_targets_arr, val_preds_tta * 100))
print(f"Best Validation RMSE (TTA): {final_rmse:.4f}")

# ── Test inference using best validated model with TTA ──────────────────
test_preds = tta_predict(model, test_df, "input/test", DEVICE, n_views=3) * 100
test_preds = np.clip(test_preds, 1.0, 100.0)

# ── Write submission ────────────────────────────────────────────────────
submission = pd.DataFrame({"Id": test_df["Id"], "Pawpularity": test_preds})
submission.to_csv("submission/submission.csv", index=False)
print(f"Submission saved: {len(submission)} rows")
print(
    f"Prediction stats: mean={test_preds.mean():.2f}, min={test_preds.min():.2f}, max={test_preds.max():.2f}"
)

print(f"Final Validation Score: {final_rmse}")
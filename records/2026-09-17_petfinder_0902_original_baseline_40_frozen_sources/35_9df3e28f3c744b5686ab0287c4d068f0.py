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
BATCH_SIZE = 32
EPOCHS = 50
LR_BACKBONE = 5e-6
LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-4
MIXUP_ALPHA = 0.3
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)


# ── Model ───────────────────────────────────────────────────────────────
class PawpularityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_base.fb_in1k", pretrained=True, num_classes=0
        )

        # Dynamically detect actual channel dimensions from backbone stages
        # For convnext_base: stages have output channels [128, 256, 512, 1024]
        stage_channels = []
        for stage in self.backbone.stages:
            # Get the last conv layer's out_channels
            last_conv = None
            for m in stage.modules():
                if isinstance(m, nn.Conv2d):
                    last_conv = m
            if last_conv is not None:
                stage_channels.append(last_conv.out_channels)
            else:
                # Fallback: use a dummy forward pass to determine shape
                raise ValueError("Could not determine stage channel dimensions")

        print(f"Detected stage channels: {stage_channels}")

        # Scale projections to normalize each stage's features to 128-dim
        self.scale_projs = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(c),
                nn.Linear(c, 128),
                nn.GELU()
            ) for c in stage_channels
        ])

        self.meta_mlp = nn.Sequential(
            nn.Linear(len(META_COLS), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Gated fusion: 512 (4*128) + 128 (meta) -> 128
        self.gate_proj = nn.Linear(512 + 128, 128)

        # Head: 512 (scales) + 128 (gated meta) = 640
        self.head = nn.Sequential(
            nn.LayerNorm(640),
            nn.Linear(640, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

        # Forward hooks for capturing stage features
        self._stage_feats = []
        self._hooks = []
        for i, stage in enumerate(self.backbone.stages):
            hook = stage.register_forward_hook(self._capture_stage_feat)
            self._hooks.append(hook)

    def _capture_stage_feat(self, module, input, output):
        self._stage_feats.append(output)

    def forward(self, images, meta):
        # Clear previous stage features
        self._stage_feats = []

        # Forward pass through backbone (triggers hooks)
        x = self.backbone(images)

        # Pool each captured spatial map to get global features
        pooled_feats = []
        for feat in self._stage_feats:
            pooled = torch.nn.functional.adaptive_avg_pool2d(feat, 1).flatten(1)
            pooled_feats.append(pooled)

        # Project each scale to 128-dim
        projected_scales = []
        for i, pooled in enumerate(pooled_feats):
            proj = self.scale_projs[i](pooled)
            projected_scales.append(proj)

        # Concatenate all scale projections (4 * 128 = 512)
        cat_scales = torch.cat(projected_scales, dim=1)

        # Process metadata
        m = self.meta_mlp(meta)  # (B, 128)

        # Gated fusion
        gate_input = torch.cat([cat_scales, m], dim=1)  # (B, 640)
        gate = torch.sigmoid(self.gate_proj(gate_input))  # (B, 128)
        gated_m = m * gate  # (B, 128)

        # Final concatenation
        z = torch.cat([cat_scales, gated_m], dim=1)  # (B, 640)

        out = self.head(z).squeeze(-1)  # (B,)
        return out

    @property
    def backbone_params(self):
        return list(self.backbone.parameters())

    @property
    def head_params(self):
        return (list(self.scale_projs.parameters()) +
                list(self.meta_mlp.parameters()) +
                list(self.gate_proj.parameters()) +
                list(self.head.parameters()))


model = PawpularityModel().to(DEVICE)

# Differential learning rates
backbone_params = model.backbone_params
head_params = model.head_params
optimizer = optim.AdamW(
    [
        {"params": backbone_params, "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY},
        {"params": head_params, "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY},
    ]
)

total_steps = EPOCHS * len(train_loader)


def lr_lambda(step):
    # Warmup for first 5% then cosine decay
    warmup_steps = int(0.05 * total_steps)
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1 + np.cos(np.pi * progress))


scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Loss function - MSE directly optimizes RMSE
loss_fn = nn.MSELoss()


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
PATIENCE = 8

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    n_batches = 0
    for imgs, metas, targets in train_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        metas = metas.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True).to(torch.float32)

        # Apply mixup
        imgs, targets = mixup_data(imgs, targets)

        preds = model(imgs, metas)
        loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        n_batches += 1

    avg_train_loss = running_loss / max(n_batches, 1)

    # Validation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, metas, targets in val_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            metas = metas.to(DEVICE, non_blocking=True)
            preds = model(imgs, metas)
            all_preds.append(preds.cpu().numpy())
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
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

# ── TTA Prediction Function ─────────────────────────────────────────────
def tta_predict(model, loader, device, n_tta=5):
    """Generate predictions using Test-Time Augmentation."""
    model.eval()
    all_preds = []

    with torch.no_grad():
        for imgs, metas, *_ in loader:
            imgs = imgs.to(device, non_blocking=True)
            metas = metas.to(device, non_blocking=True)

            # Original
            pred_orig = model(imgs, metas)

            # Horizontal flip
            imgs_flip = torch.flip(imgs, dims=[3])
            pred_flip = model(imgs_flip, metas)

            # Rotation -5 degrees (tensor-native, supports batched input)
            imgs_rot_neg = transforms.functional.rotate(imgs, -5)
            pred_rot_neg = model(imgs_rot_neg, metas)

            # Rotation +5 degrees
            imgs_rot_pos = transforms.functional.rotate(imgs, 5)
            pred_rot_pos = model(imgs_rot_pos, metas)

            # Brightness adjustment (deterministic, tensor-native)
            imgs_bright = transforms.functional.adjust_brightness(imgs, 1.1)
            pred_bright = model(imgs_bright, metas)

            # Average all predictions
            avg_pred = (pred_orig + pred_flip + pred_rot_neg + pred_rot_pos + pred_bright) / 5.0
            all_preds.append(avg_pred.cpu().numpy())

    return np.concatenate(all_preds)

# ── Restore best model ──────────────────────────────────────────────────
model.load_state_dict(best_state)
model.to(DEVICE)
model.eval()

# ── Final validation inference with TTA ─────────────────────────────────
val_preds_tta = tta_predict(model, val_loader, DEVICE)
val_targets_arr = np.array([row["Pawpularity"] for _, row in val_split.iterrows()])
final_rmse = np.sqrt(mean_squared_error(val_targets_arr, val_preds_tta * 100))
print(f"Best Validation RMSE (with TTA): {final_rmse:.4f}")

# ── Test inference with TTA using the validated best model ──────────────
test_preds_tta = tta_predict(model, test_loader, DEVICE)
test_preds = test_preds_tta * 100
# Clip to valid range
test_preds = np.clip(test_preds, 1.0, 100.0)

# ── Write submission ────────────────────────────────────────────────────
submission = pd.DataFrame({"Id": test_df["Id"], "Pawpularity": test_preds})
submission.to_csv("submission/submission.csv", index=False)
print(f"Submission saved: {len(submission)} rows")
print(
    f"Prediction stats: mean={test_preds.mean():.2f}, min={test_preds.min():.2f}, max={test_preds.max():.2f}"
)

print(f"Final Validation Score: {final_rmse}")
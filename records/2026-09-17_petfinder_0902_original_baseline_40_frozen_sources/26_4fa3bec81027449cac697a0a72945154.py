import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
IMG_SIZE = 256
BATCH_SIZE = 12
EPOCHS = 50
LR_BACKBONE = 5e-6
LR_HEAD = 5e-4
WEIGHT_DECAY = 1e-4
MIXUP_ALPHA = 0.3
NUM_WORKERS = 2
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
class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion where metadata queries visual features.

    Projects meta features into K/V space and uses projected image features
    as Q, enabling dynamic, input-dependent fusion where metadata can
    selectively attend to different visual aspects based on context.
    """

    def __init__(self, img_dim, meta_dim, hidden_dim=256, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Project image features to query space
        self.q_proj = nn.Linear(img_dim, hidden_dim)
        # Project meta features to key/value space
        self.k_proj = nn.Linear(meta_dim, hidden_dim)
        self.v_proj = nn.Linear(meta_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        # Scale factor for attention
        self.scale = self.head_dim ** -0.5

    def forward(self, img_feat, meta_feat):
        # img_feat: (B, D_img), meta_feat: (B, D_meta)
        B = img_feat.size(0)

        # Project to Q, K, V spaces
        q = self.q_proj(img_feat).view(B, self.num_heads, self.head_dim)  # (B, heads, head_dim)
        k = self.k_proj(meta_feat).view(B, self.num_heads, self.head_dim)  # (B, heads, head_dim)
        v = self.v_proj(meta_feat).view(B, self.num_heads, self.head_dim)  # (B, heads, head_dim)

        # Compute attention scores: Q @ K^T / sqrt(head_dim)
        attn_scores = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B, heads, 1)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, heads, 1)

        # Weighted sum of values
        attn_output = torch.bmm(attn_weights, v)  # (B, heads, head_dim)
        attn_output = attn_output.view(B, self.hidden_dim)  # (B, hidden_dim)

        # Output projection and normalization
        out = self.out_proj(attn_output)
        out = self.norm(out)

        return out


class PawpularityModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Primary backbone: ConvNeXt Large
        self.backbone_primary = timm.create_model(
            "convnext_large.fb_in22k_ft_in1k", pretrained=True, num_classes=0
        )
        # Secondary backbone: EfficientNet B0
        self.backbone_secondary = timm.create_model(
            "efficientnet_b0.ra2_in1k", pretrained=True, num_classes=0
        )

        primary_dim = self.backbone_primary.num_features  # 1024 for convnext_large
        secondary_dim = self.backbone_secondary.num_features  # 1280 for efficientnet_b0
        combined_dim = primary_dim + secondary_dim  # 2304

        meta_dim = len(META_COLS)
        hidden_dim = 256

        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(combined_dim, meta_dim, hidden_dim=hidden_dim, num_heads=4)

        # Head as specified in the plan
        self.head = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, images, meta):
        # Extract features from both backbones
        primary_feat = self.backbone_primary(images)  # (B, primary_dim)
        secondary_feat = self.backbone_secondary(images)  # (B, secondary_dim)

        # Concatenate features from both branches
        img_feat = torch.cat([primary_feat, secondary_feat], dim=1)  # (B, combined_dim)

        # Cross-attention fusion
        z = self.fusion(img_feat, meta)  # (B, hidden_dim)

        # Expand to 2*hidden_dim for the head
        z_expanded = torch.cat([z, z], dim=1)  # (B, 2*hidden_dim)

        out = self.head(z_expanded).squeeze(-1)  # (B,)
        return out


model = PawpularityModel().to(DEVICE)

# Differential learning rates
backbone_params = list(model.backbone_primary.parameters()) + list(model.backbone_secondary.parameters())
head_params = (
    list(model.fusion.parameters())
    + list(model.head.parameters())
)
optimizer = optim.AdamW(
    [
        {"params": backbone_params, "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY},
        {"params": head_params, "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY},
    ]
)

total_steps = EPOCHS * len(train_loader)


def lr_lambda(step):
    # Cosine warmup of 3% then cosine decay
    warmup_steps = int(0.03 * total_steps)
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1 + np.cos(np.pi * progress))


scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# AMP scaler for mixed precision training
scaler = torch.cuda.amp.GradScaler()


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
PATIENCE = 20

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

        # Mixed precision training
        with torch.autocast('cuda'):
            preds = model(imgs, metas)
            loss = nn.HuberLoss(delta=0.1)(preds, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item()
        n_batches += 1

    avg_train_loss = running_loss / max(n_batches, 1)

    # Validation (plain inference, no TTA - honest metric)
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, metas, targets in val_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            metas = metas.to(DEVICE, non_blocking=True)
            with torch.autocast('cuda'):
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

# ── Final validation inference WITHOUT TTA (honest metric) ─────────────
model.eval()
all_val_preds, all_val_targets = [], []
with torch.no_grad():
    for imgs, metas, targets in val_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        metas = metas.to(DEVICE, non_blocking=True)
        with torch.autocast('cuda'):
            preds = model(imgs, metas)
        all_val_preds.append(preds.float().cpu().numpy())
        all_val_targets.append(targets.numpy())

all_val_preds = np.concatenate(all_val_preds)
all_val_targets = np.concatenate(all_val_targets)
final_rmse = np.sqrt(mean_squared_error(all_val_targets * 100, all_val_preds * 100))
print(f"Best Validation RMSE (without TTA): {final_rmse:.4f}")

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
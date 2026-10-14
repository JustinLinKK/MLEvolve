import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# ── Reproducibility ──────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Load data ────────────────────────────────────────────────────
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

FEATURE_COLS = [
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
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
print(
    f"Target stats: mean={train_df['Pawpularity'].mean():.2f}, "
    f"std={train_df['Pawpularity'].std():.2f}"
)

# ── Stratified split by Pawpularity decile ───────────────────────
train_df["decile"] = pd.qcut(
    train_df["Pawpularity"], q=10, labels=False, duplicates="drop"
)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
train_idx, val_idx = next(sss.split(train_df, train_df["decile"]))
train_split = train_df.iloc[train_idx].reset_index(drop=True)
val_split = train_df.iloc[val_idx].reset_index(drop=True)
print(f"Train split: {len(train_split)}, Val split: {len(val_split)}")

# ── Image transforms ─────────────────────────────────────────────
IMG_SIZE = 384
train_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class PetDataset(Dataset):
    def __init__(self, df, img_dir, transform, is_train=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['Id']}.jpg")
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))

        if self.transform:
            img = self.transform(img)

        feats = row[FEATURE_COLS].values.astype(np.float32)

        if self.is_train:
            target = np.float32(row["Pawpularity"])
            return img, feats, target
        return img, feats


# ── Model ────────────────────────────────────────────────────────
class PawpularityModel(nn.Module):
    def __init__(self, num_features=12, dropout=0.3):
        super().__init__()
        # Dual vision backbones for complementary representations
        # ViT-Small/16: global semantic understanding
        self.backbone_vit = timm.create_model(
            "vit_small_patch16_384",
            pretrained=True,
            num_classes=0,  # remove classification head → 384-dim feature
        )
        vit_feat_dim = self.backbone_vit.num_features  # 384 for vit_small

        # ConvNeXt-Base: local/textural cues (blur, occlusion, fine details)
        self.backbone_conv = timm.create_model(
            "convnext_base.fb_in22k_ft_in1k",
            pretrained=True,
            num_classes=0,  # remove classification head → 768-dim feature
        )
        conv_feat_dim = self.backbone_conv.num_features  # 768 for convnext_base

        # Metadata MLP
        self.meta_mlp = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Fusion head combining both visual features + metadata
        total_feat_dim = vit_feat_dim + conv_feat_dim + 32
        self.head = nn.Sequential(
            nn.Linear(total_feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

    def forward(self, images, meta):
        # Extract features from both backbones
        vis_feat_vit = self.backbone_vit(images)  # (B, 384)
        vis_feat_conv = self.backbone_conv(images)  # (B, 768)
        meta_feat = self.meta_mlp(meta)  # (B, 32)

        # Concatenate all features
        x = torch.cat([vis_feat_vit, vis_feat_conv, meta_feat], dim=1)
        out = self.head(x)
        return out.squeeze(-1)


model = PawpularityModel().to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── DataLoaders ──────────────────────────────────────────────────
train_ds = PetDataset(train_split, "./input/train", train_transform, is_train=True)
val_ds = PetDataset(val_split, "./input/train", val_transform, is_train=False)
test_ds = PetDataset(test_df, "./input/test", val_transform, is_train=False)

train_loader = DataLoader(
    train_ds,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
)
val_loader = DataLoader(
    val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
)

# ── Training setup ───────────────────────────────────────────────
EPOCHS = 35
LR_BACKBONE = 1e-5
LR_HEAD = 5e-4
WEIGHT_DECAY = 1e-4

optimizer = optim.AdamW(
    [
        {"params": model.backbone_vit.parameters(), "lr": LR_BACKBONE},
        {"params": model.backbone_conv.parameters(), "lr": LR_BACKBONE},
        {
            "params": list(model.meta_mlp.parameters()) + list(model.head.parameters()),
            "lr": LR_HEAD,
            "weight_decay": WEIGHT_DECAY,
        },
    ]
)

total_steps = EPOCHS * len(train_loader)
warmup_steps = max(1, total_steps // 6)


def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + np.cos(np.pi * progress))


scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# Mixup helper
def mixup_data(x, y, alpha=0.1):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


# ── Training loop ────────────────────────────────────────────────
best_val_rmse = float("inf")
best_state = None
patience = 12
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    n_batches = 0

    for imgs, metas, targets in train_loader:
        imgs = imgs.to(DEVICE, non_blocking=True).float()
        metas = metas.to(DEVICE, non_blocking=True).float()
        targets = targets.to(DEVICE, non_blocking=True).float()

        # Apply mixup for regularization (slightly stronger for larger model)
        imgs, targets = mixup_data(imgs, targets, alpha=0.15)

        optimizer.zero_grad()
        preds = model(imgs, metas)
        loss = nn.functional.mse_loss(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        n_batches += 1

    avg_loss = running_loss / max(1, n_batches)

    # Validation
    model.eval()
    all_preds = []
    with torch.no_grad():
        for imgs, metas in val_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            metas = metas.to(DEVICE, non_blocking=True)
            preds = model(imgs, metas)
            all_preds.append(preds.cpu().numpy())

    val_preds = np.concatenate(all_preds)
    val_targets = val_split["Pawpularity"].values
    val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val RMSE: {val_rmse:.4f}")

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# ── Restore best model ───────────────────────────────────────────
model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
model.eval()
print(f"Best Val RMSE: {best_val_rmse:.4f}")

# ── Final validation inference ───────────────────────────────────
val_preds_final = []
with torch.no_grad():
    for imgs, metas in val_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        metas = metas.to(DEVICE, non_blocking=True)
        preds = model(imgs, metas)
        val_preds_final.append(preds.cpu().numpy())

val_preds_final = np.concatenate(val_preds_final)
final_rmse = np.sqrt(
    mean_squared_error(val_split["Pawpularity"].values, val_preds_final)
)
print(f"Final Validation Score: {final_rmse:.4f}")

# ── Test predictions & submission ────────────────────────────────
test_preds = []
with torch.no_grad():
    for imgs, metas in test_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        metas = metas.to(DEVICE, non_blocking=True)
        preds = model(imgs, metas)
        test_preds.append(preds.cpu().numpy())

test_preds = np.concatenate(test_preds)
# Clip to valid range [1, 100]
test_preds = np.clip(test_preds, 1.0, 100.0)

os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame({"Id": test_df["Id"], "Pawpularity": test_preds})
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {len(submission)} rows")
print(submission.head())

# Ensure the LAST line printed is the required format
print(f"Final Validation Score: {final_rmse:.4f}")

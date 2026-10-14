import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import warnings
import cv2
import timm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Load data ───────────────────────────────────────────────────
train_df = pd.read_csv("input/train.csv")
test_df = pd.read_csv("input/test.csv")
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

metadata_cols = [
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

# ── Reproducible validation split ───────────────────────────────
# Stratify by Pawpularity decile
train_df["decile"] = pd.qcut(
    train_df["Pawpularity"], q=10, labels=False, duplicates="drop"
)
val_mask = np.zeros(len(train_df), dtype=bool)
for d in train_df["decile"].unique():
    idx = train_df.index[train_df["decile"] == d].values
    np.random.seed(SEED + int(d))
    val_idx = np.random.choice(idx, size=max(1, len(idx) // 5), replace=False)
    val_mask[val_idx] = True

train_split = train_df[~val_mask].reset_index(drop=True)
val_split = train_df[val_mask].reset_index(drop=True)
print(f"Train split: {len(train_split)}, Val split: {len(val_split)}")


# ── Extract hand-crafted image quality features ────────────────
def extract_quality_features(img_path):
    """Extract hand-crafted image quality features from an image."""
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        # Return default values if image can't be loaded
        return np.array([0.0] * 10, dtype=np.float32)

    img_np = np.array(img, dtype=np.float32)
    h, w = img_np.shape[:2]

    # (a) Laplacian variance (sharpness/focus proxy)
    gray = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # (b) Mean brightness per channel
    mean_r = img_np[:, :, 0].mean()
    mean_g = img_np[:, :, 1].mean()
    mean_b = img_np[:, :, 2].mean()

    # (c) Saturation (max-min of RGB channels)
    max_rgb = img_np.max(axis=2)
    min_rgb = img_np.min(axis=2)
    saturation = ((max_rgb - min_rgb) / (max_rgb + 1e-8)).mean()

    # (d) Aspect ratio
    aspect_ratio = w / h

    # (e) Edge density via Sobel filter magnitude mean
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_density = sobel_mag.mean()

    # (f) Simple subject centering metric (centroid of non-background pixels)
    # Threshold to find foreground
    threshold = gray.mean() * 0.7
    mask = gray > threshold
    if mask.sum() > 0:
        ys, xs = np.where(mask)
        centroid_x = xs.mean() / w
        centroid_y = ys.mean() / h
        center_offset = np.sqrt((centroid_x - 0.5)**2 + (centroid_y - 0.5)**2)
    else:
        center_offset = 0.5

    return np.array([
        laplacian_var,      # Sharpness
        mean_r,             # Red brightness
        mean_g,             # Green brightness
        mean_b,             # Blue brightness
        saturation,         # Colorfulness
        aspect_ratio,       # Aspect ratio
        edge_density,       # Edge density
        center_offset,      # Subject centering
        w,                  # Width
        h                   # Height
    ], dtype=np.float32)


def extract_all_quality_features(df, img_dir):
    """Extract quality features for all images in dataframe."""
    n_samples = len(df)
    quality_feats = np.zeros((n_samples, 10), dtype=np.float32)

    for i in range(n_samples):
        img_path = os.path.join(img_dir, f"{df.iloc[i]['Id']}.jpg")
        quality_feats[i] = extract_quality_features(img_path)
        if (i + 1) % 500 == 0:
            print(f"  Quality features: {i+1}/{n_samples}")

    return quality_feats


# ── Extract frozen ResNet50 embeddings ──────────────────────────
class PetDataset(Dataset):
    def __init__(self, df, img_dir, transforms_list):
        """transforms_list: list of transforms for TTA (one per view)."""
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transforms_list = transforms_list

    def __len__(self):
        return len(self.df) * len(self.transforms_list)

    def __getitem__(self, idx):
        img_idx = idx // len(self.transforms_list)
        tta_idx = idx % len(self.transforms_list)
        row = self.df.iloc[img_idx]
        img_path = os.path.join(self.img_dir, f"{row['Id']}.jpg")
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        img = self.transforms_list[tta_idx](img)
        meta = np.array([row[c] for c in metadata_cols], dtype=np.float32)
        return img, meta


# TTA: 3 deterministic views - center crop, horizontal flip + center crop, full-image resize
def make_tta_transforms():
    """Return list of 3 deterministic transforms for TTA using cleaner strategy."""
    base_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    to_tensor = transforms.ToTensor()
    resize_224 = transforms.Resize((IMG_SIZE, IMG_SIZE))

    # (a) Center crop at 256px -> resize to 224
    t_center = transforms.Compose([
        transforms.CenterCrop(256),
        resize_224,
        to_tensor,
        base_norm,
    ])

    # (b) Horizontal flip + center crop at 256px -> resize to 224
    t_flip = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.CenterCrop(256),
        resize_224,
        to_tensor,
        base_norm,
    ])

    # (c) Full-image resize to 224 (no crop)
    t_full = transforms.Compose([
        resize_224,
        to_tensor,
        base_norm,
    ])

    return [t_center, t_flip, t_full]

tta_transforms = make_tta_transforms()


# ── Fine-tuned ConvNeXt-Large Regression Model ──────────────────
class PetRegressionDataset(Dataset):
    """Dataset for fine-tuning ConvNeXt-Large with Pawpularity regression."""

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

        img = self.transform(img)
        label = float(row["Pawpularity"])

        return img, label


class ConvNeXtRegressor(nn.Module):
    """Fine-tuned ConvNeXt-Large with MLP head for Pawpularity regression."""

    def __init__(self):
        super().__init__()
        # Create backbone without classifier
        self.backbone = timm.create_model(
            'convnext_large.fb_in22k_ft_in1k',
            pretrained=True,
            num_classes=0
        )

        # Get feature dimension
        feat_dim = self.backbone.num_features  # 1536 for ConvNeXt-Large

        # Custom MLP head
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features)
        return out.squeeze(-1)


def get_train_transform():
    """Training-time augmentations."""
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25),
    ])


def get_val_transform():
    """Validation/test deterministic transform."""
    return transforms.Compose([
        transforms.CenterCrop(256),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform_flip():
    """Validation/test transform with horizontal flip for TTA."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.CenterCrop(256),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Build ConvNeXt-Large backbone using timm (remove classifier head for embedding extraction)
feature_extractor = timm.create_model('convnext_large.fb_in22k_ft_in1k', pretrained=True, num_classes=0)
feature_extractor = feature_extractor.to(DEVICE)
feature_extractor.eval()


# ── Train Fine-Tuned ConvNeXt-Large ────────────────────────────
def train_finetuned_convnext(train_df, val_df, img_dir, epochs=15, patience=4):
    """Train fine-tuned ConvNeXt-Large with cosine annealing and early stopping."""

    # Create datasets
    train_dataset = PetRegressionDataset(train_df, img_dir, get_train_transform(), is_train=True)
    val_dataset = PetRegressionDataset(val_df, img_dir, get_val_transform(), is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # Initialize model
    model = ConvNeXtRegressor().to(DEVICE)

    # Differential learning rates
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head.parameters())

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': head_params, 'lr': 1e-4}
    ], weight_decay=1e-4)

    # Cosine annealing scheduler with warmup
    warmup_epochs = 2
    total_epochs = epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-7)

    # Mixed precision
    scaler = GradScaler()
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    patience_counter = 0
    best_model_state = None

    print("Starting fine-tuned ConvNeXt-Large training...")

    for epoch in range(total_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        n_batches = 0

        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with autocast():
                predictions = model(images)
                loss = criterion(predictions, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / n_batches

        # Validation phase
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE, non_blocking=True)

                with autocast():
                    preds = model(images)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.numpy())

        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        val_rmse = np.sqrt(mean_squared_error(val_labels, val_preds))

        # Update scheduler
        scheduler.step()

        print(f"Epoch {epoch+1}/{total_epochs} | Train Loss: {avg_train_loss:.4f} | Val RMSE: {val_rmse:.4f}")

        # Early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"Best Val RMSE: {best_val_rmse:.4f}")
    return model, best_val_rmse


def predict_with_tta(model, df, img_dir):
    """Generate predictions using TTA (center crop + horizontal flip)."""
    model.eval()

    # Create two datasets: normal and flipped
    dataset_normal = PetRegressionDataset(df, img_dir, get_val_transform(), is_train=False)
    dataset_flip = PetRegressionDataset(df, img_dir, get_val_transform_flip(), is_train=False)

    loader_normal = DataLoader(dataset_normal, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    loader_flip = DataLoader(dataset_flip, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    preds_normal = []
    preds_flip = []

    with torch.no_grad():
        for images, _ in loader_normal:
            images = images.to(DEVICE, non_blocking=True)
            with autocast():
                preds = model(images)
            preds_normal.extend(preds.cpu().numpy())

        for images, _ in loader_flip:
            images = images.to(DEVICE, non_blocking=True)
            with autocast():
                preds = model(images)
            preds_flip.extend(preds.cpu().numpy())

    # Average TTA predictions
    preds_normal = np.array(preds_normal)
    preds_flip = np.array(preds_flip)
    final_preds = (preds_normal + preds_flip) / 2.0

    return final_preds


def extract_embeddings(df, img_dir):
    """Extract frozen ViT features + metadata for a dataframe using TTA (3 views averaged)."""
    n_views = len(tta_transforms)
    ds = PetDataset(df, img_dir, tta_transforms)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    n_samples = len(df)
    all_feats = np.zeros((n_samples, n_views, 1536), dtype=np.float32)
    all_meta = np.zeros((n_samples, len(metadata_cols)), dtype=np.float32)
    n_batches = len(dl)
    running_idx = 0
    for i, (imgs, metas) in enumerate(dl):
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.no_grad():
            feats = feature_extractor(imgs)  # (B, 1536) for ConvNeXt-Large
        feats_np = feats.cpu().numpy()
        B = feats_np.shape[0]
        # Map back to original sample indices using running counter
        for j in range(B):
            global_idx = running_idx + j
            if global_idx < n_samples * n_views:
                orig_idx = global_idx // n_views
                view_idx = global_idx % n_views
                all_feats[orig_idx, view_idx, :] = feats_np[j]
                all_meta[orig_idx, :] = metas[j].numpy()
        running_idx += B
        if (i + 1) % 20 == 0 or i == n_batches - 1:
            print(f"  Batch {i+1}/{n_batches}")

    # Average over TTA views
    avg_feats = all_feats.mean(axis=1)  # (n_samples, 1536)
    return avg_feats, all_meta


print("Extracting train embeddings...")
train_img_feats, train_meta = extract_embeddings(train_split, "input/train")
print("Extracting val embeddings...")
val_img_feats, val_meta = extract_embeddings(val_split, "input/train")
print("Extracting test embeddings...")
test_img_feats, test_meta = extract_embeddings(test_df, "input/test")

# Extract quality features
print("Extracting quality features...")
train_quality_feats = extract_all_quality_features(train_split, "input/train")
val_quality_feats = extract_all_quality_features(val_split, "input/train")
test_quality_feats = extract_all_quality_features(test_df, "input/test")

# ── Train Fine-Tuned ConvNeXt-Large ────────────────────────────
print("\n" + "="*50)
print("Training Fine-Tuned ConvNeXt-Large")
print("="*50)
deep_model, deep_val_rmse = train_finetuned_convnext(train_split, val_split, "input/train", epochs=15, patience=4)

# Generate deep model predictions
print("Generating deep model predictions...")
deep_val_preds = predict_with_tta(deep_model, val_split, "input/train")
deep_test_preds = predict_with_tta(deep_model, test_df, "input/test")

# Define y arrays first (needed for deep model evaluation and later)
y_train = train_split["Pawpularity"].values.astype(np.float32)
y_val = val_split["Pawpularity"].values.astype(np.float32)

# Evaluate deep model on validation set
deep_val_rmse_actual = np.sqrt(mean_squared_error(y_val, deep_val_preds))
print(f"Deep Model Val RMSE (with TTA): {deep_val_rmse_actual:.4f}")

# Use full 1536-dim embeddings directly (no PCA - LightGBM handles high-dim features well)
# Combine features: [avg_img_feats(1536) | quality_feats(~10) | meta(12)]
X_train = np.hstack([train_img_feats, train_quality_feats, train_meta])
X_val = np.hstack([val_img_feats, val_quality_feats, val_meta])
X_test = np.hstack([test_img_feats, test_quality_feats, test_meta])

print(f"Feature shape: {X_train.shape}")

# ── Train LightGBM with 5-fold CV on train split ───────────────
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,
    "num_leaves": 64,
    "max_depth": -1,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))
cv_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=6000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)],
    )

    pred_va = model.predict(X_va)
    cv_preds[va_idx] = pred_va
    rmse = np.sqrt(mean_squared_error(y_va, pred_va))
    cv_scores.append(rmse)
    print(f"Fold {fold+1}: RMSE={rmse:.4f}, best_iter={model.best_iteration}")

    test_preds += model.predict(X_test) / 5

cv_rmse = np.sqrt(mean_squared_error(y_train, cv_preds))
print(f"CV RMSE (on train split): {cv_rmse:.4f}")
print(f"Per-fold scores: {[f'{s:.4f}' for s in cv_scores]}")

# ── Evaluate on held-out validation set ─────────────────────────
# Retrain on full train split (train+val) for final val prediction?
# No — we need to evaluate on the actual held-out val set.
# Use the CV model's predictions on val via retraining on train_split only.
# Actually, let's just train one model on train_split and predict val_split.

dtrain_full = lgb.Dataset(X_train, label=y_train)
dval_final = lgb.Dataset(X_val, label=y_val, reference=dtrain_full)
final_model = lgb.train(
    params,
    dtrain_full,
    num_boost_round=6000,
    valid_sets=[dval_final],
    callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)],
)
val_pred = final_model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Held-out Val RMSE: {val_rmse:.4f}")

# ── Ensemble: combine LightGBM and Deep Model predictions ───────
# final_pred = 0.5 * lgb_pred + 0.5 * deep_pred (both clipped to [0,100])
lgb_val_pred_clipped = np.clip(val_pred, 0, 100)
deep_val_pred_clipped = np.clip(deep_val_preds, 0, 100)
ensemble_val_pred = 0.5 * lgb_val_pred_clipped + 0.5 * deep_val_pred_clipped

lgb_test_pred_clipped = np.clip(test_preds, 0, 100)
deep_test_pred_clipped = np.clip(deep_test_preds, 0, 100)
ensemble_test_pred = 0.5 * lgb_test_pred_clipped + 0.5 * deep_test_pred_clipped

# Evaluate ensemble on held-out validation set
ensemble_val_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_pred))
print(f"Ensemble Held-out Val RMSE: {ensemble_val_rmse:.4f}")

# ── Generate submission ─────────────────────────────────────────
os.makedirs("submission", exist_ok=True)
submission = pd.DataFrame({"Id": test_df["Id"], "Pawpularity": ensemble_test_pred})
# Clip to valid range
submission["Pawpularity"] = submission["Pawpularity"].clip(0, 100)
submission.to_csv("submission/submission.csv", index=False)
print(f"Submission saved: {len(submission)} rows")
print(submission.head())

print(f"Final Validation Score: {ensemble_val_rmse:.4f}")
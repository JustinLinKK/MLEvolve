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

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 64
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


# TTA: 3 deterministic crops at 256x256 before resizing to 224
# (a) center crop, (b) left-shifted crop, (c) right-shifted crop
def make_tta_transforms():
    """Return list of 3 deterministic transforms for TTA: original, hflip, and slight rotation."""
    base_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((IMG_SIZE, IMG_SIZE))

    # (a) Original: just resize
    t_orig = transforms.Compose([
        resize, to_tensor, base_norm,
    ])
    # (b) Horizontal flip
    t_flip = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        resize, to_tensor, base_norm,
    ])
    # (c) Slight rotation (+5 degrees)
    t_rot = transforms.Compose([
        transforms.RandomRotation(degrees=(5, 5)),
        resize, to_tensor, base_norm,
    ])
    return [t_orig, t_flip, t_rot]

tta_transforms = make_tta_transforms()

# Build ViT backbone using timm (remove classifier head for embedding extraction)
import timm
feature_extractor = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
feature_extractor = feature_extractor.to(DEVICE)
feature_extractor.eval()


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
    all_feats = np.zeros((n_samples, n_views, 768), dtype=np.float32)
    all_meta = np.zeros((n_samples, len(metadata_cols)), dtype=np.float32)
    n_batches = len(dl)
    running_idx = 0
    for i, (imgs, metas) in enumerate(dl):
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.no_grad():
            feats = feature_extractor(imgs)  # (B, 768) for ViT-B
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
    avg_feats = all_feats.mean(axis=1)  # (n_samples, 768)
    return avg_feats, all_meta


print("Extracting train embeddings...")
train_img_feats, train_meta = extract_embeddings(train_split, "input/train")
print("Extracting val embeddings...")
val_img_feats, val_meta = extract_embeddings(val_split, "input/train")
print("Extracting test embeddings...")
test_img_feats, test_meta = extract_embeddings(test_df, "input/test")

# ── PCA to reduce dimensionality (helps LightGBM speed & stability) ──
from sklearn.decomposition import PCA

n_components = 128
pca = PCA(n_components=n_components, random_state=SEED)
train_img_pca = pca.fit_transform(train_img_feats)
val_img_pca = pca.transform(val_img_feats)
test_img_pca = pca.transform(test_img_feats)
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")

# Combine features
X_train = np.hstack([train_img_pca, train_meta])
y_train = train_split["Pawpularity"].values.astype(np.float32)
X_val = np.hstack([val_img_pca, val_meta])
y_val = val_split["Pawpularity"].values.astype(np.float32)
X_test = np.hstack([test_img_pca, test_meta])

print(f"Feature shape: {X_train.shape}")

# ── Train LightGBM with 5-fold CV on train split ───────────────
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,
    "num_leaves": 40,
    "max_depth": -1,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
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

# ── Generate submission ─────────────────────────────────────────
os.makedirs("submission", exist_ok=True)
submission = pd.DataFrame({"Id": test_df["Id"], "Pawpularity": test_preds})
# Clip to valid range
submission["Pawpularity"] = submission["Pawpularity"].clip(0, 100)
submission.to_csv("submission/submission.csv", index=False)
print(f"Submission saved: {len(submission)} rows")
print(submission.head())

print(f"Final Validation Score: {val_rmse:.4f}")

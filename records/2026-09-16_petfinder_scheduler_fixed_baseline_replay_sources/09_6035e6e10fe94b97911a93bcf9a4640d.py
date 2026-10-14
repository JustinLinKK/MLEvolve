import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import root_mean_squared_error, mean_squared_error
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import warnings
from PIL import Image

warnings.filterwarnings("ignore")

os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Load data ──────────────────────────────────────────────
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
print(
    f"Pawpularity stats: mean={train_df['Pawpularity'].mean():.2f}, std={train_df['Pawpularity'].std():.2f}"
)

METADATA_COLS = [
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

IMG_SIZE = 224

# ── Transforms ─────────────────────────────────────────────
val_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class PawDataset(Dataset):
    def __init__(self, file_paths, data_dir="./input/train/"):
        self.file_paths = file_paths
        self.data_dir = data_dir

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, f"{self.file_paths[idx]}.jpg")
        image = Image.open(img_path).convert("RGB")
        image = val_transform(image)
        return image


def extract_features(model, dataset, batch_size=64, num_workers=4):
    """Extract embeddings from all images in dataset."""
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    all_feats = []
    with torch.no_grad():
        for batch in loader:
            feats = model(batch.to(device))
            all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


# ── Build timm model for feature extraction ────────────────
print("\nLoading timm model for feature extraction...")
model_name = "vit_base_patch16_224.augreg_in21k_ft_in1k"
feature_model = timm.create_model(model_name, pretrained=True, num_classes=0)
feature_model = feature_model.to(device)
feature_model.eval()
feat_dim = feature_model.num_features
print(f"Feature dimension: {feat_dim}")

# ── Extract features for train and test ────────────────────
print("\nExtracting training image features...")
train_dataset = PawDataset(train_df["Id"].values.tolist(), data_dir="./input/train/")
train_img_feats = extract_features(
    feature_model, train_dataset, batch_size=64, num_workers=4
)
print(f"Train image features shape: {train_img_feats.shape}")

print("\nExtracting test image features...")
test_dataset = PawDataset(test_df["Id"].values.tolist(), data_dir="./input/test/")
test_img_feats = extract_features(
    feature_model, test_dataset, batch_size=64, num_workers=4
)
print(f"Test image features shape: {test_img_feats.shape}")

# Free GPU memory
del feature_model
torch.cuda.empty_cache()

# ── Combine features ───────────────────────────────────────
X_train_full = np.hstack(
    [train_img_feats, train_df[METADATA_COLS].values.astype(np.float32)]
)
X_test_full = np.hstack(
    [test_img_feats, test_df[METADATA_COLS].values.astype(np.float32)]
)
y_train = train_df["Pawpularity"].values.astype(np.float32)

print(f"\nCombined feature matrix shape: {X_train_full.shape}")

# ── Stratified split by Pawpularity decile ─────────────────
deciles = pd.qcut(y_train, q=10, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_val_preds = np.zeros(len(X_train_full))
fold_scores = []
best_iters_list = []

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": -1,
}

print("\nTraining 5-fold CV with image+metadata features...")
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_full, deciles)):
    X_tr, X_va = X_train_full[tr_idx], X_train_full[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dvalid],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_pred = model.predict(X_va)
    all_val_preds[va_idx] = val_pred

    rmse = np.sqrt(mean_squared_error(y_va, val_pred))
    fold_scores.append(rmse)
    best_iters_list.append(model.best_iteration)
    print(f"  Fold {fold+1}: RMSE = {rmse:.4f} (best_iter={model.best_iteration})")

cv_rmse = np.sqrt(mean_squared_error(y_train, all_val_preds))
print(f"\nCV RMSE (all folds): {cv_rmse:.4f}")
print(f"Per-fold scores: {[f'{s:.4f}' for s in fold_scores]}")
print(f"Best iterations: {best_iters_list}")

# ── Retrain on full training data ──────────────────────────
avg_best_iter = int(np.mean(best_iters_list)) * 1.2  # slight buffer
print(f"\nRetraining on full data with {int(avg_best_iter)} iterations...")
dtrain_full = lgb.Dataset(X_train_full, label=y_train)
final_model = lgb.train(
    params,
    dtrain_full,
    num_boost_round=int(avg_best_iter),
    valid_sets=[dtrain_full],
    callbacks=[lgb.log_evaluation(period=0)],
)

# ── Generate test predictions ──────────────────────────────
test_preds = final_model.predict(X_test_full)
test_preds = np.clip(test_preds, 0, 100)

# ── Validation score ───────────────────────────────────────
final_score = cv_rmse
print(f"\nFinal Validation Score: {final_score:.4f}")

# ── Write submission ───────────────────────────────────────
submission = pd.DataFrame({"Id": test_df["Id"], "Pawpularity": test_preds})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved. Shape: {submission.shape}")
print(submission.head())
print(
    f"\nPrediction stats: mean={test_preds.mean():.2f}, std={test_preds.std():.2f}, min={test_preds.min():.2f}, max={test_preds.max():.2f}"
)

# Final line as required
print(f"Final Validation Score: {final_score}")

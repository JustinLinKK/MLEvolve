import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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

# ── CNN Model Builder ───────────────────────────────────────
def build_cnn_model(model_name="convnext_small.fb_in22k_ft_in1k"):
    """Build a timm CNN model for regression with metadata MLP head."""
    base_model = timm.create_model(model_name, pretrained=True, num_classes=0)
    # Enable gradient checkpointing to save memory
    if hasattr(base_model, 'set_grad_checkpointing'):
        base_model.set_grad_checkpointing(True)
    elif hasattr(base_model, 'gradient_checkpointing'):
        base_model.gradient_checkpointing = True
    feat_dim = base_model.num_features

    class CNNRegressor(nn.Module):
        def __init__(self, base_model, feat_dim, meta_dim=12):
            super().__init__()
            self.base_model = base_model
            self.meta_mlp = nn.Sequential(
                nn.Linear(meta_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU()
            )
            # Concatenate image features (feat_dim) + metadata features (32)
            combined_dim = feat_dim + 32
            self.head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(combined_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1)
            )

        def forward(self, images, metadata):
            img_feats = self.base_model(images)  # [B, feat_dim]
            meta_feats = self.meta_mlp(metadata.float())  # [B, 32]
            combined = torch.cat([img_feats, meta_feats], dim=1)  # [B, feat_dim+32]
            out = self.head(combined)  # [B, 1]
            return out.squeeze(-1)

    return CNNRegressor(base_model, feat_dim, meta_dim=12)


class PawDatasetWithMeta(Dataset):
    """Dataset that returns both image and metadata for CNN training."""
    def __init__(self, file_paths, metadata, data_dir="./input/train/", transform=None):
        self.file_paths = file_paths
        self.metadata = metadata
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, f"{self.file_paths[idx]}.jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        meta = torch.tensor(self.metadata[idx], dtype=torch.float32)
        return image, meta


def train_cnn_branch(train_df, test_df, deciles, skf, METADATA_COLS, IMG_SIZE, device):
    """Train the CNN regression branch with 5-fold CV."""
    print("\n" + "="*60)
    print("Training CNN Regression Branch")
    print("="*60)

    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform_cnn = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    y_train = train_df["Pawpularity"].values.astype(np.float32)
    X_meta_train = train_df[METADATA_COLS].values.astype(np.float32)
    X_meta_test = test_df[METADATA_COLS].values.astype(np.float32)

    cnn_oof_preds = np.zeros(len(train_df))
    fold_rmse_list = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_meta_train, deciles)):
        print(f"\n--- CNN Fold {fold+1} ---")

        # Create datasets
        train_dataset = PawDatasetWithMeta(
            train_df["Id"].iloc[tr_idx].tolist(),
            X_meta_train[tr_idx],
            data_dir="./input/train/",
            transform=train_transform
        )
        val_dataset = PawDatasetWithMeta(
            train_df["Id"].iloc[va_idx].tolist(),
            X_meta_train[va_idx],
            data_dir="./input/train/",
            transform=val_transform_cnn
        )

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

        # Build model
        model = build_cnn_model().to(device)

        # Parameter groups: backbone lr=1e-4, head lr=1e-3
        backbone_params = list(model.base_model.parameters())
        head_params = list(model.meta_mlp.parameters()) + list(model.head.parameters())

        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": 1e-4},
            {"params": head_params, "lr": 1e-3},
        ], weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
        criterion = nn.MSELoss()

        best_val_rmse = float('inf')
        patience = 4
        counter = 0
        best_state = None

        for epoch in range(15):
            # Training
            model.train()
            train_loss = 0.0
            for images, metas in train_loader:
                images = images.to(device)
                metas = metas.to(device)

                optimizer.zero_grad()
                outputs = model(images, metas)
                targets = train_df["Pawpularity"].values[tr_idx].astype(np.float32)
                targets = torch.tensor(targets, dtype=torch.float32).to(device)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(images)

            train_loss /= len(train_dataset)
            scheduler.step()

            # Validation
            model.eval()
            val_preds = []
            with torch.no_grad():
                for images, metas in val_loader:
                    images = images.to(device)
                    metas = metas.to(device)
                    outputs = model(images, metas)
                    val_preds.append(outputs.cpu().numpy())

            val_preds = np.concatenate(val_preds)
            val_targets = y_train[va_idx]
            val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))

            print(f"  Epoch {epoch+1}/15: Train Loss={train_loss:.4f}, Val RMSE={val_rmse:.4f}")

            # Early stopping
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Load best model and get OOF predictions
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        oof_preds = []
        with torch.no_grad():
            for images, metas in val_loader:
                images = images.to(device)
                metas = metas.to(device)
                outputs = model(images, metas)
                oof_preds.append(outputs.cpu().numpy())

        oof_preds = np.concatenate(oof_preds)
        cnn_oof_preds[va_idx] = oof_preds
        fold_rmse_list.append(best_val_rmse)
        print(f"  Best Val RMSE: {best_val_rmse:.4f}")

        del model
        torch.cuda.empty_cache()

    cnn_cv_rmse = np.sqrt(mean_squared_error(y_train, cnn_oof_preds))
    print(f"\nCNN Branch CV RMSE: {cnn_cv_rmse:.4f}")
    print(f"Per-fold RMSEs: {[f'{r:.4f}' for r in fold_rmse_list]}")

    # Generate test predictions using all folds' models
    # For simplicity, retrain on full data for final test predictions
    print("\nRetraining CNN on full data for test predictions...")
    full_train_dataset = PawDatasetWithMeta(
        train_df["Id"].values.tolist(),
        X_meta_train,
        data_dir="./input/train/",
        transform=train_transform
    )
    full_train_loader = DataLoader(full_train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)

    test_dataset = PawDatasetWithMeta(
        test_df["Id"].values.tolist(),
        X_meta_test,
        data_dir="./input/test/",
        transform=val_transform_cnn
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    model = build_cnn_model().to(device)
    backbone_params = list(model.base_model.parameters())
    head_params = list(model.meta_mlp.parameters()) + list(model.head.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1e-4},
        {"params": head_params, "lr": 1e-3},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    criterion = nn.MSELoss()

    for epoch in range(15):
        model.train()
        train_loss = 0.0
        for images, metas in full_train_loader:
            images = images.to(device)
            metas = metas.to(device)

            optimizer.zero_grad()
            outputs = model(images, metas)
            targets = torch.tensor(y_train, dtype=torch.float32).to(device)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(images)

        train_loss /= len(full_train_dataset)
        scheduler.step()
        print(f"  Full Train Epoch {epoch+1}/15: Loss={train_loss:.4f}")

    # Test predictions
    model.eval()
    test_preds_cnn = []
    with torch.no_grad():
        for images, metas in test_loader:
            images = images.to(device)
            metas = metas.to(device)
            outputs = model(images, metas)
            test_preds_cnn.append(outputs.cpu().numpy())

    test_preds_cnn = np.concatenate(test_preds_cnn)
    test_preds_cnn = np.clip(test_preds_cnn, 0, 100)

    del model
    torch.cuda.empty_cache()

    return cnn_oof_preds, test_preds_cnn, cnn_cv_rmse

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
    feature_model, train_dataset, batch_size=32, num_workers=2
)
print(f"Train image features shape: {train_img_feats.shape}")

print("\nExtracting test image features...")
test_dataset = PawDataset(test_df["Id"].values.tolist(), data_dir="./input/test/")
test_img_feats = extract_features(
    feature_model, test_dataset, batch_size=32, num_workers=2
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

# ── Retrain on full training data (LightGBM branch) ──────────────────────────
avg_best_iter = int(np.mean(best_iters_list)) * 1.2  # slight buffer
print(f"\nRetraining LightGBM on full data with {int(avg_best_iter)} iterations...")
dtrain_full = lgb.Dataset(X_train_full, label=y_train)
final_lgb_model = lgb.train(
    params,
    dtrain_full,
    num_boost_round=int(avg_best_iter),
    valid_sets=[dtrain_full],
    callbacks=[lgb.log_evaluation(period=0)],
)

# ── Generate LightGBM test predictions ──────────────────────────────────────
lgb_test_preds = final_lgb_model.predict(X_test_full)
lgb_test_preds = np.clip(lgb_test_preds, 0, 100)

# ── Train CNN Regression Branch ─────────────────────────────────────────────
try:
    cnn_oof_preds, cnn_test_preds, cnn_cv_rmse = train_cnn_branch(
        train_df, test_df, deciles, skf, METADATA_COLS, IMG_SIZE, device
    )
    use_cnn = True
except Exception as e:
    print(f"\nCNN branch failed: {e}. Falling back to LightGBM only.")
    use_cnn = False
    cnn_cv_rmse = float('inf')

if use_cnn:
    # ── Ensemble: weighted average of LightGBM and CNN ─────────────────────────
    w_lgb = 0.5
    w_cnn = 0.5

    ensemble_oof = w_lgb * all_val_preds + w_cnn * cnn_oof_preds
    ensemble_oof = np.clip(ensemble_oof, 0, 100)

    ensemble_test = w_lgb * lgb_test_preds + w_cnn * cnn_test_preds
    ensemble_test = np.clip(ensemble_test, 0, 100)

    ensemble_cv_rmse = np.sqrt(mean_squared_error(y_train, ensemble_oof))
    print(f"\nEnsemble CV RMSE: {ensemble_cv_rmse:.4f}")
    print(f"  LightGBM CV RMSE: {cv_rmse:.4f}")
    print(f"  CNN CV RMSE:      {cnn_cv_rmse:.4f}")

    final_score = ensemble_cv_rmse
    final_test_preds = ensemble_test
else:
    # Fallback: use LightGBM only
    final_score = cv_rmse
    final_test_preds = lgb_test_preds
    print(f"\nUsing LightGBM-only predictions. CV RMSE: {cv_rmse:.4f}")

print(f"\nFinal Validation Score: {final_score:.4f}")

# ── Write submission ───────────────────────────────────────
submission = pd.DataFrame({"Id": test_df["Id"], "Pawpularity": final_test_preds})
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved. Shape: {submission.shape}")
print(submission.head())
print(
    f"\nPrediction stats: mean={final_test_preds.mean():.2f}, std={final_test_preds.std():.2f}, min={final_test_preds.min():.2f}, max={final_test_preds.max():.2f}"
)

# Final line as required
print(f"Final Validation Score: {final_score}")

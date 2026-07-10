import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score as _mlevolve_original_roc_auc_score


def roc_auc_score(y_true, y_score, *args, **kwargs):
    try:
        return _mlevolve_original_roc_auc_score(y_true, y_score, *args, **kwargs)
    except ValueError as exc:
        if 'Only one class present' in str(exc):
            return 0.5
        raise

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoModel, AutoConfig

class _MlevolveProbeVisionConfig:
    hidden_size = 1152


class _MlevolveProbeConfig:
    vision_config = _MlevolveProbeVisionConfig()


class _MlevolveProbeImageBackbone(nn.Module):
    def __init__(self, feature_dim=1152):
        super().__init__()
        self.config = _MlevolveProbeConfig()
        self.proj = nn.Linear(3, feature_dim)

    def get_image_features(self, pixel_values, *args, **kwargs):
        pooled = torch.nn.functional.adaptive_avg_pool2d(pixel_values.float(), (1, 1)).flatten(1)
        return self.proj(pooled)


def _mlevolve_probe_or_load_automodel(*args, **kwargs):
    if os.environ.get('MLEVOLVE_PROBE_MODE') == '1':
        return _MlevolveProbeImageBackbone()
    return AutoModel.from_pretrained(*args, **kwargs)

import torch.optim as optim
import time
import gc

# A unique identifier for this model configuration.
MODEL_FAMILY = "siglip2-so400m-patch16-256_feature_extractor"

# --- Main Configuration & Hyperparameters ---
DATA_DIR = "./input"
TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "train")
TEST_IMAGE_DIR = os.path.join(DATA_DIR, "test")
TRAIN_LABELS_PATH = os.path.join(DATA_DIR, "train_labels.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
IMAGE_SIZE = 256  # For SigLIP compatibility
VALIDATION_SPLIT = 0.1
RANDOM_STATE = 42

HP = {
    "PHYSICAL_BATCH_SIZE": 256,
    "NUM_WORKERS": 4,
    "EPOCHS": 10,
    "LEARNING_RATE": 1e-3,
    "WEIGHT_DECAY": 1e-2,
    "EARLY_STOPPING_PATIENCE": 2,
    "SUBMISSION_DIR": "./submission",
    "CHECKPOINT_PATH": "./working/best_model_8b1794d06c54422d950607470e28c4e2.pth",
}

# ==============================================================================
# STEP 1: DATA PROCESSING AND FEATURE ENGINEERING
# ==============================================================================


class ReinhardNormalizer:
    """
    A class to perform Reinhard stain normalization.
    It is fit on a target image and can then be used to transform other images
    to match the target's color profile.
    """

    def __init__(self):
        self.target_means = None
        self.target_stds = None

    def fit(self, target_image):
        """
        Fits the normalizer to a target image.
        Args:
            target_image (np.ndarray): The target image in BGR format (as read by cv2).
        """
        target_lab = cv2.cvtColor(target_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(target_lab)
        l_mask = np.ma.masked_equal(l, 0)
        l_mask = np.ma.masked_equal(l_mask, 255)
        a_mask = np.ma.masked_equal(a, 0)
        a_mask = np.ma.masked_equal(a_mask, 255)
        b_mask = np.ma.masked_equal(b, 0)
        b_mask = np.ma.masked_equal(b_mask, 255)

        self.target_means = [np.mean(l_mask), np.mean(a_mask), np.mean(b_mask)]
        self.target_stds = [np.std(l_mask), np.std(a_mask), np.std(b_mask)]

    def transform(self, image):
        """
        Transforms an image to match the fitted target.
        """
        if self.target_means is None or self.target_stds is None:
            raise RuntimeError("Normalizer has not been fit. Call .fit() first.")

        source_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(source_lab)

        l_mask = np.ma.masked_equal(l, 0)
        l_mask = np.ma.masked_equal(l_mask, 255)
        a_mask = np.ma.masked_equal(a, 0)
        a_mask = np.ma.masked_equal(a_mask, 255)
        b_mask = np.ma.masked_equal(b, 0)
        b_mask = np.ma.masked_equal(b_mask, 255)

        source_means = [np.mean(l_mask), np.mean(a_mask), np.mean(b_mask)]
        source_stds = [np.std(l_mask), np.std(a_mask), np.std(b_mask)]
        source_stds = [std if std > 1e-6 else 1.0 for std in source_stds]

        for i in range(3):
            channel = [l, a, b][i]
            channel = (
                (channel.astype(np.float32) - source_means[i])
                * (self.target_stds[i] / source_stds[i])
            ) + self.target_means[i]
            channel = np.clip(channel, 0, 255).astype(np.uint8)
            [l, a, b][i] = channel

        normalized_lab = cv2.merge([l, a, b])
        normalized_bgr = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)
        return normalized_bgr


class PcamDataset(Dataset):
    def __init__(self, df, image_dir, normalizer, transform=None, is_test=False):
        self.df = df
        self.image_dir = image_dir
        self.normalizer = normalizer
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["id"]
        image_path = os.path.join(self.image_dir, f"{image_id}.tif")

        image_cv2 = cv2.imread(image_path)
        if image_cv2 is None:
            print(
                f"Warning: Could not read image {image_path}. Returning a black image."
            )
            image_cv2 = np.zeros((96, 96, 3), dtype=np.uint8)

        normalized_image = self.normalizer.transform(image_cv2)
        image_pil = Image.fromarray(cv2.cvtColor(normalized_image, cv2.COLOR_BGR2RGB))

        if self.transform:
            image_tensor = self.transform(image_pil)

        if self.is_test:
            return image_tensor, image_id
        else:
            label = torch.tensor(row["label"], dtype=torch.float32)
            return image_tensor, label


def get_data_loaders(batch_size, num_workers=2):
    df_labels = pd.read_csv(TRAIN_LABELS_PATH)
    train_df, val_df = train_test_split(
        df_labels,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_STATE,
        stratify=df_labels["label"],
    )

    # Select reference image for stain normalization from the training set to prevent data leakage.
    reference_image_id = train_df.iloc[0]["id"]
    reference_image_path = os.path.join(TRAIN_IMAGE_DIR, f"{reference_image_id}.tif")
    reference_image_cv2 = cv2.imread(reference_image_path)
    normalizer = ReinhardNormalizer()
    normalizer.fit(reference_image_cv2)
    print("Stain normalizer fitted to reference image.")

    siglip_normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            siglip_normalize,
        ]
    )
    val_test_transform = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            siglip_normalize,
        ]
    )

    train_dataset = PcamDataset(train_df, TRAIN_IMAGE_DIR, normalizer, train_transform)
    val_dataset = PcamDataset(val_df, TRAIN_IMAGE_DIR, normalizer, val_test_transform)
    test_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    test_dataset = PcamDataset(
        test_df, TEST_IMAGE_DIR, normalizer, val_test_transform, is_test=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(
        f"DataLoaders created:\n  - Train: {len(train_dataset)} images, {len(train_loader)} batches\n  - Val:   {len(val_dataset)} images, {len(val_loader)} batches\n  - Test:  {len(test_dataset)} images, {len(test_loader)} batches"
    )
    return train_loader, val_loader, test_loader


# ==============================================================================
# STEP 2: MODEL DESIGN
# ==============================================================================


class PcamSiglipModel(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        config = AutoConfig.from_pretrained("google/siglip2-so400m-patch16-256")
        self.feature_dim = config.vision_config.hidden_size
        print("Loading pretrained SigLIP model...")
        self.siglip_model = _mlevolve_probe_or_load_automodel(
            "google/siglip2-so400m-patch16-256"
        )
        print("SigLIP model loaded.")
        for param in self.siglip_model.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(self.feature_dim, 1)
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.siglip_model.eval()
        features = self.siglip_model.get_image_features(pixel_values=pixel_values)
        logits = self.classifier(features)
        return logits


def get_model() -> nn.Module:
    print("Initializing model...")
    model = PcamSiglipModel()
    return model


def get_loss_function() -> nn.Module:
    return nn.BCEWithLogitsLoss()


# ==============================================================================
# STEP 3: DATATYPE & PRECISION CONFIGURATION
# ==============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_TF32 = True
if (
    USE_TF32
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] >= 8
):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    TF32_ENABLED = True
else:
    TF32_ENABLED = False

USE_AMP = True
AMP_DTYPE = torch.bfloat16


def get_grad_scaler() -> torch.cuda.amp.GradScaler:
    return torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE == "cuda"))


def log_precision_settings():
    print("--- Precision Settings ---")
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"TF32 Enabled on Matmul/cuDNN: {TF32_ENABLED}")
        print(
            f"Automatic Mixed Precision (AMP): {'Enabled' if USE_AMP else 'Disabled'}"
        )
        if USE_AMP:
            dtype_str = "bfloat16" if AMP_DTYPE == torch.bfloat16 else "float16"
            print(f"AMP Datatype: {dtype_str}")
    else:
        print("Running on CPU, no advanced precision features enabled.")
    print("--------------------------")


# ==============================================================================
# STEP 4: TRAINING & EVALUATION
# ==============================================================================


def train_one_epoch(
    model, loader, loss_fn, optimizer, scaler, scheduler, device, dtype
):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=dtype, enabled=USE_AMP):
            logits = model(images)
            loss = loss_fn(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device, dtype):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            with torch.autocast(device_type=device, dtype=dtype, enabled=USE_AMP):
                logits = model(images)
                preds = torch.sigmoid(logits)
            all_preds.append(preds.float().cpu().numpy())
            all_labels.append(labels.numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    auc_score = roc_auc_score(all_labels, all_preds)
    return auc_score


def predict_test(model, loader, device, dtype):
    model.eval()
    all_preds, all_ids = [], []
    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device)
            with torch.autocast(device_type=device, dtype=dtype, enabled=USE_AMP):
                logits = model(images)
                preds = torch.sigmoid(logits)
            all_preds.append(preds.float().cpu().numpy())
            all_ids.extend(ids)
    all_preds = np.concatenate(all_preds).flatten()
    return pd.DataFrame({"id": all_ids, "label": all_preds})


def main():
    log_precision_settings()
    print("\n--- Training Configuration ---")
    for key, val in HP.items():
        print(f"{key}: {val}")
    print("----------------------------\n")

    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=HP["PHYSICAL_BATCH_SIZE"], num_workers=HP["NUM_WORKERS"]
    )

    model = get_model().to(DEVICE)
    loss_fn = get_loss_function()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=HP["LEARNING_RATE"],
        weight_decay=HP["WEIGHT_DECAY"],
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=HP["LEARNING_RATE"],
        epochs=HP["EPOCHS"],
        steps_per_epoch=len(train_loader),
    )
    scaler = get_grad_scaler()

    best_auc = 0.0
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(HP["EPOCHS"]):
        epoch_start_time = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            scaler,
            scheduler,
            DEVICE,
            AMP_DTYPE,
        )
        val_auc = evaluate(model, val_loader, DEVICE, AMP_DTYPE)
        epoch_duration = time.time() - epoch_start_time
        print(
            f"Epoch {epoch+1}/{HP['EPOCHS']} - Train Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}, Time: {epoch_duration:.2f}s, LR: {scheduler.get_last_lr()[0]:.1e}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            os.makedirs(os.path.dirname(HP["CHECKPOINT_PATH"]), exist_ok=True)
            torch.save(model.state_dict(), HP["CHECKPOINT_PATH"])
            print(f"  -> New best model saved with AUC: {best_auc:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= HP["EARLY_STOPPING_PATIENCE"]:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break

        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print(f"\nTotal Training Time: {(time.time() - start_time):.2f} seconds")

    print("\nLoading best model for test set prediction...")
    model.load_state_dict(torch.load(HP["CHECKPOINT_PATH"]))

    print("Generating predictions on the test set...")
    submission_df = predict_test(model, test_loader, DEVICE, AMP_DTYPE)

    os.makedirs(HP["SUBMISSION_DIR"], exist_ok=True)
    submission_path = os.path.join(HP["SUBMISSION_DIR"], "submission_8b1794d06c54422d950607470e28c4e2.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved to: {submission_path}")

    print(f"Final Validation Score: {best_auc}")


if __name__ == "__main__":
    main()
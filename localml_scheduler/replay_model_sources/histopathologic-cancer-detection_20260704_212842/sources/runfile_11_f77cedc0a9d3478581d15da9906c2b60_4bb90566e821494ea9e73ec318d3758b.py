import os
import pandas as pd
import numpy as np
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
import timm
import torch.optim as optim
import time
import gc

# A unique identifier for this model configuration.
MODEL_FAMILY = "efficientnet_b0"

# --- Main Configuration & Hyperparameters ---
DATA_DIR = "./input"
TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, "train")
TEST_IMAGE_DIR = os.path.join(DATA_DIR, "test")
TRAIN_LABELS_PATH = os.path.join(DATA_DIR, "train_labels.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
IMAGE_SIZE = 96  # Native image size for the dataset
VALIDATION_SPLIT = 0.1
RANDOM_STATE = 42

HP = {
    "PHYSICAL_BATCH_SIZE": 128, # Reduced for larger model
    "NUM_WORKERS": 4,
    "EPOCHS": 4, # Reduced for faster run, can be increased
    "LEARNING_RATE": 3e-4,
    "WEIGHT_DECAY": 1e-2,
    "EARLY_STOPPING_PATIENCE": 2,
    "SUBMISSION_DIR": "./submission",
    "CHECKPOINT_PATH": "./working/best_model_f77cedc0a9d3478581d15da9906c2b60.pth",
}

# ==============================================================================
# STEP 1: DATA PROCESSING AND FEATURE ENGINEERING
# ==============================================================================

class PcamDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, is_test=False):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["id"]
        image_path = os.path.join(self.image_dir, f"{image_id}.tif")

        # Open with PIL, which is what torchvision transforms expect
        try:
            image_pil = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Could not read image {image_path}. Returning a black image.")
            image_pil = Image.new('RGB', (96, 96), color = 'black')


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

    # Use standard ImageNet normalization for EfficientNet
    imagenet_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            imagenet_normalize,
        ]
    )
    val_test_transform = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            imagenet_normalize,
        ]
    )

    train_dataset = PcamDataset(train_df, TRAIN_IMAGE_DIR, train_transform)
    val_dataset = PcamDataset(val_df, TRAIN_IMAGE_DIR, val_test_transform)
    test_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    test_dataset = PcamDataset(
        test_df, TEST_IMAGE_DIR, val_test_transform, is_test=True
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
        batch_size=int(batch_size * 1.5),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(batch_size * 1.5),
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

def get_model() -> nn.Module:
    print(f"Initializing model: {MODEL_FAMILY}")
    model = timm.create_model(
        MODEL_FAMILY,
        pretrained=True,
        num_classes=1, # single logit output for binary classification
        drop_rate=0.2,
        drop_path_rate=0.2
    )
    return model


def get_loss_function() -> nn.Module:
    return nn.BCEWithLogitsLoss()


# ==============================================================================
# STEP 3: DATATYPE & PRECISION CONFIGURATION
# ==============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_TF32 = False # TF32 is for Ampere+ GPUs, but let's be explicit
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
    num_steps = len(loader)
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=dtype, enabled=USE_AMP):
            logits = model(images)
            loss = loss_fn(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step() # OneCycleLR steps every batch
        total_loss += loss.item()
    return total_loss / num_steps


def evaluate(model, loader, device, dtype):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            with torch.autocast(device_type=device, dtype=dtype, enabled=USE_AMP):
                logits = model(images)
                preds = torch.sigmoid(logits)
            all_preds.append(preds.float().cpu().numpy())
            all_labels.append(labels.float().cpu().numpy())
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
        model.parameters(),
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
    submission_path = os.path.join(HP["SUBMISSION_DIR"], "submission_f77cedc0a9d3478581d15da9906c2b60.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved to: {submission_path}")

    print(f"Final Validation Score: {best_auc}")


if __name__ == "__main__":
    main()
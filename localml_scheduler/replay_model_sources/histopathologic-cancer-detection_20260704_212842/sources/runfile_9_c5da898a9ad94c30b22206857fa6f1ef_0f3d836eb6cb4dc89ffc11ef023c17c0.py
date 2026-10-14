import os
import gc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
import timm
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score as _mlevolve_original_roc_auc_score


def roc_auc_score(y_true, y_score, *args, **kwargs):
    try:
        return _mlevolve_original_roc_auc_score(y_true, y_score, *args, **kwargs)
    except ValueError as exc:
        if 'Only one class present' in str(exc):
            return 0.5
        raise


# ==============================================================================
# 0. GLOBAL CONFIGURATION & CONSTANTS
# ==============================================================================
# As per Scheduler Model Family Contract, define a top-level constant
MODEL_FAMILY = "efficientnet_96"
MODEL_NAME = "efficientnet_b0"  # Specific model from the family

# Paths
INPUT_DIR = "./input"
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")
TRAIN_LABELS_PATH = os.path.join(INPUT_DIR, "train_labels.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Data Processing Parameters
IMG_SIZE = 96  # Native image size to prevent OOM errors
VALIDATION_SPLIT = 0.1
RANDOM_STATE = 42

# ==============================================================================
# STAGE 1: DATA PROCESSING & FEATURE ENGINEERING
# ==============================================================================


class HistopathologyDataset(Dataset):
    """
    Custom PyTorch Dataset for loading histopathology images.
    """

    def __init__(self, dataframe, image_dir, transform=None, is_test=False):
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = self.df.iloc[idx]["id"]
        img_path = os.path.join(self.image_dir, f"{img_id}.tif")

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image: {img_path}, returning blank image. Error: {e}")
            image = Image.new("RGB", (96, 96), color="black")

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, img_id

        label = self.df.iloc[idx]["label"]
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        return image, label_tensor


def get_train_transforms(img_size):
    """Returns a composition of transforms for the training set with aggressive augmentation."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=45),
            transforms.ColorJitter(
                brightness=0.25, contrast=0.25, saturation=0.1, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_validation_transforms(img_size):
    """Returns a composition of transforms for validation/test set (no augmentation)."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def create_dataloaders(input_dir, batch_size, num_workers, val_split, random_state):
    """Creates and returns train, validation, and test dataloaders."""
    print("Creating dataloaders...")
    train_labels_df = pd.read_csv(os.path.join(input_dir, "train_labels.csv"))
    train_df, val_df = train_test_split(
        train_labels_df,
        test_size=val_split,
        random_state=random_state,
        stratify=train_labels_df["label"],
    )

    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")

    train_transforms = get_train_transforms(IMG_SIZE)
    val_transforms = get_validation_transforms(IMG_SIZE)

    train_dataset = HistopathologyDataset(
        dataframe=train_df, image_dir=TRAIN_IMG_DIR, transform=train_transforms
    )
    val_dataset = HistopathologyDataset(
        dataframe=val_df, image_dir=TRAIN_IMG_DIR, transform=val_transforms
    )

    test_df = pd.read_csv(os.path.join(input_dir, "sample_submission.csv"))
    test_dataset = HistopathologyDataset(
        dataframe=test_df,
        image_dir=TEST_IMG_DIR,
        transform=val_transforms,
        is_test=True,
    )
    print(f"Test set size: {len(test_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print("Dataloaders created successfully.")
    return train_dataloader, val_dataloader, test_dataloader


# ==============================================================================
# STAGE 2: MODEL DESIGN
# ==============================================================================


def create_model(pretrained=True):
    """Creates a timm model with a custom classifier for binary classification."""
    model = timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=1)
    return model


def get_loss_function():
    """Returns the loss function suitable for the task (binary classification)."""
    return nn.BCEWithLogitsLoss()


# ==============================================================================
# STAGE 3: DATATYPE & PRECISION
# ==============================================================================


@dataclass
class PrecisionConfig:
    """A dataclass to hold precision-related settings."""

    device: torch.device
    use_amp: bool
    amp_dtype: torch.dtype
    use_grad_scaler: bool


def get_precision_config():
    """Determines and configures the device and precision settings for the pipeline."""
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Per pipeline decisions, use bfloat16 AMP
    use_amp = True
    amp_dtype = torch.bfloat16
    use_grad_scaler = False  # Not needed for bfloat16

    if use_cuda and not torch.cuda.is_bf16_supported():
        print(
            "Warning: bfloat16 is not supported. Falling back to fp16 with GradScaler."
        )
        amp_dtype = torch.float16
        use_grad_scaler = True  # Required for fp16 to prevent underflow

    if not use_cuda:
        print("Warning: CUDA not available. Disabling AMP.")
        use_amp = False

    config = PrecisionConfig(
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        use_grad_scaler=use_grad_scaler,  # Note: Not used in this training loop, but configured for completeness
    )
    print(
        f"Precision config: device={config.device}, use_amp={config.use_amp}, amp_dtype={config.amp_dtype}"
    )
    return config


# ==============================================================================
# STAGE 4: TRAINING & EVALUATION
# ==============================================================================
class TrainingConfig:
    """Configuration class for training hyperparameters."""

    N_EPOCHS = 3
    LR = 1e-4
    WEIGHT_DECAY = 1e-6
    # As per hardware-aware guidance, use a large batch size and more workers
    BATCH_SIZE = 512
    NUM_WORKERS = 8
    MODEL_SAVE_PATH = "./working/best_model_c5da898a9ad94c30b22206857fa6f1ef.pth"
    SUBMISSION_PATH = "./submission/submission_c5da898a9ad94c30b22206857fa6f1ef.csv"


def train_one_epoch(
    model, loader, optimizer, scheduler, loss_fn, device, precision_config, scaler
):
    """Runs a single training epoch."""
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(
            device_type=device.type,
            dtype=precision_config.amp_dtype,
            enabled=precision_config.use_amp,
        ):
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, loss_fn, device, precision_config):
    """Runs validation and returns loss and AUC score."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=precision_config.amp_dtype,
            enabled=precision_config.use_amp,
        ):
            outputs = model(images)
            loss = loss_fn(outputs, labels)
        total_loss += loss.item()
        preds = torch.sigmoid(outputs)
        all_preds.append(preds.float().cpu().numpy())
        all_labels.append(labels.float().cpu().numpy())
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    auc_score = roc_auc_score(all_labels, all_preds)
    return total_loss / len(loader), auc_score


@torch.no_grad()
def generate_submission(model, loader, submission_path, device, precision_config):
    """Generates submission.csv for the test set."""
    model.eval()
    all_preds, all_ids = [], []
    for images, ids in loader:
        images = images.to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=precision_config.amp_dtype,
            enabled=precision_config.use_amp,
        ):
            outputs = model(images)
        preds = torch.sigmoid(outputs)
        all_preds.append(preds.float().cpu().numpy())
        all_ids.extend(ids)
    all_preds = np.vstack(all_preds).flatten()
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df = pd.DataFrame({"id": all_ids, "label": all_preds})
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file created at: {submission_path}")
    print(submission_df.head())


def run_pipeline():
    """Main function to run the complete ML pipeline."""
    print("--- Starting Kaggle ML Pipeline ---")
    cfg = TrainingConfig()

    # Create working and submission directories
    os.makedirs(os.path.dirname(cfg.MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.SUBMISSION_PATH), exist_ok=True)

    # Get precision and device settings
    precision_config = get_precision_config()
    device = precision_config.device

    # 1. Data Loading (with hardware-optimized batch size)
    print(
        f"Recreating dataloaders with Batch Size: {cfg.BATCH_SIZE} and Workers: {cfg.NUM_WORKERS}"
    )
    train_loader, val_loader, test_loader = create_dataloaders(
        input_dir=INPUT_DIR,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        val_split=VALIDATION_SPLIT,
        random_state=RANDOM_STATE,
    )

    # 2. Model, Loss, Optimizer, Scheduler Setup
    model = create_model(pretrained=True).to(device)
    loss_fn = get_loss_function()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
    )
    total_steps = len(train_loader) * cfg.N_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=precision_config.use_grad_scaler)

    # 3. Training Loop
    best_val_auc = 0.0
    final_validation_score = 0.0

    for epoch in range(cfg.N_EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device, precision_config, scaler
        )
        val_loss, val_auc = validate(
            model, val_loader, loss_fn, device, precision_config
        )

        # Minimal logging: 1 line per epoch
        print(
            f"Epoch {epoch + 1}/{cfg.N_EPOCHS} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_validation_score = val_auc
            torch.save(model.state_dict(), cfg.MODEL_SAVE_PATH)
            print(f"  New best model saved with AUC: {val_auc:.4f}")

    # 4. Submission Generation
    print("\nTraining complete. Loading best model for submission.")
    model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH))
    generate_submission(
        model, test_loader, cfg.SUBMISSION_PATH, device, precision_config
    )

    # 5. Cleanup
    del model, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Final output line for the score parser
    print(f"Final Validation Score: {final_validation_score}")


# ==============================================================================
# SCRIPT ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    run_pipeline()
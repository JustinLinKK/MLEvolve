import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import AutoProcessor, AutoModel

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

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import roc_auc_score as _mlevolve_original_roc_auc_score


def roc_auc_score(y_true, y_score, *args, **kwargs):
    try:
        return _mlevolve_original_roc_auc_score(y_true, y_score, *args, **kwargs)
    except ValueError as exc:
        if 'Only one class present' in str(exc):
            return 0.5
        raise


# --- Single Configuration Block ---

# 1. Model & Path Configuration
MODEL_FAMILY = "siglip2_so400m_p16_256_96px"
MODEL_CHECKPOINT = "google/siglip2-so400m-patch16-256"
BASE_DIR = "./input"
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test")
TRAIN_LABELS_PATH = os.path.join(BASE_DIR, "train_labels.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_DIR, "sample_submission.csv")
MODEL_SAVE_PATH = "./working/best_model_14422040b904464faf1908650332d01b.pth"
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission_14422040b904464faf1908650332d01b.csv")

# 2. Data & Training Hyperparameters
VALID_SIZE = 0.2
RANDOM_STATE = 42
LEARNING_RATE = 0.000315
WEIGHT_DECAY = 0.01
EPOCHS = 2
BATCH_SIZE = 256  # Reduced to prevent OOM with a large model on a 32GB GPU
NUM_WORKERS = 8  # As suggested by hardware context

# --- Step 1: Data Processing & Feature Engineering ---


class HistopathologyDataset(Dataset):
    """
    Custom PyTorch Dataset for the PCam competition.
    It loads images, applies specified transformations, and returns the image tensor and label.
    """

    def __init__(self, df, image_dir, processor, transforms=None, is_test=False):
        self.df = df
        self.image_dir = image_dir
        self.processor = processor
        self.transforms = transforms
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]["id"]
        img_path = os.path.join(self.image_dir, f"{img_id}.tif")

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image file not found at {img_path}. Skipping.")
            dummy_processed = self.processor(
                images=Image.new("RGB", (96, 96)), return_tensors="pt"
            )
            if self.is_test:
                return dummy_processed["pixel_values"].squeeze(0), img_id
            return dummy_processed["pixel_values"].squeeze(0), torch.tensor(
                0, dtype=torch.float32
            )

        if self.transforms:
            image = self.transforms(image)

        processed_inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = processed_inputs["pixel_values"].squeeze(0)

        if self.is_test:
            return pixel_values, img_id
        else:
            label = self.df.iloc[idx]["label"]
            return pixel_values, torch.tensor(label, dtype=torch.float32)


def create_datasets():
    """
    Loads labels, splits data, and creates PyTorch Dataset objects.
    """
    print("Starting data processing and feature engineering...")
    labels_df = pd.read_csv(TRAIN_LABELS_PATH)
    test_df = pd.read_csv(SAMPLE_SUB_PATH)

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=VALID_SIZE, random_state=RANDOM_STATE
    )
    train_indices, val_indices = next(
        splitter.split(labels_df["id"], labels_df["label"])
    )

    train_df = labels_df.iloc[train_indices].reset_index(drop=True)
    val_df = labels_df.iloc[val_indices].reset_index(drop=True)

    print(
        f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}"
    )

    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            ),
        ]
    )

    print(f"Loading processor for model: {MODEL_CHECKPOINT}")
    processor = AutoProcessor.from_pretrained(MODEL_CHECKPOINT)

    train_dataset = HistopathologyDataset(
        df=train_df,
        image_dir=TRAIN_IMG_DIR,
        processor=processor,
        transforms=train_transforms,
    )
    val_dataset = HistopathologyDataset(
        df=val_df, image_dir=TRAIN_IMG_DIR, processor=processor, transforms=None
    )

    print("Data processing and feature engineering complete.")
    return train_dataset, val_dataset, test_df, processor


# --- Step 2: Model Design ---


class HistopathologyModel(nn.Module):
    """
    A model for histopathology image classification using a pretrained Siglip2 backbone.
    """

    def __init__(self, pretrained_model_name: str, freeze_backbone: bool = True):
        super().__init__()
        print(f"Initializing model with backbone: {pretrained_model_name}")
        self.backbone = _mlevolve_probe_or_load_automodel(pretrained_model_name)
        feature_dim = self.backbone.config.vision_config.hidden_size
        self.classifier = nn.Linear(feature_dim, 1)

        if freeze_backbone:
            print("Freezing backbone parameters for feature extraction.")
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            print("Backbone parameters will be fine-tuned.")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.backbone.get_image_features(pixel_values=pixel_values)
        logits = self.classifier(features)
        return logits


def create_model_and_loss():
    """Instantiates the model and the loss function."""
    print("Creating model and loss function...")
    model = HistopathologyModel(
        pretrained_model_name=MODEL_CHECKPOINT, freeze_backbone=True
    )
    criterion = nn.BCEWithLogitsLoss()
    print("Model and loss function created successfully.")
    return model, criterion


# --- Step 3: Datatype/Precision ---


def setup_precision_environment():
    """Configures the device and precision settings for training."""
    print("Setting up datatype and precision environment...")

    if torch.cuda.is_available():
        DEVICE = "cuda"
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled for CUDA matmul and cuDNN operations.")
    else:
        DEVICE = "cpu"
        print("CUDA not available. Using CPU.")

    AMP_ENABLED = False
    AMP_DTYPE = torch.float32
    if DEVICE == "cuda" and torch.cuda.is_bf16_supported():
        AMP_DTYPE = torch.bfloat16
        AMP_ENABLED = True
        print("Device supports bfloat16. Using AMP with bfloat16.")
    elif DEVICE == "cuda":
        print(
            "Warning: bfloat16 not supported. AMP is disabled. Training will use fp32 (with TF32 acceleration)."
        )
    else:
        print("Running on CPU. AMP is disabled.")

    SCALER = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)
    print(f"Gradient Scaler initialized (Enabled: {AMP_ENABLED}).")
    print("Datatype and precision setup complete.")
    return DEVICE, AMP_ENABLED, AMP_DTYPE, SCALER


# --- Step 4: Training & Evaluation ---


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    scaler,
    amp_enabled,
    amp_dtype,
):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        with torch.autocast(
            device_type=device.split(":")[0], dtype=amp_dtype, enabled=amp_enabled
        ):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


@torch.no_grad()
def validate(model, val_loader, criterion, device, amp_enabled, amp_dtype):
    """Validates the model and computes loss and AUC."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device).unsqueeze(1)
        all_labels.append(labels.cpu())
        with torch.autocast(
            device_type=device.split(":")[0], dtype=amp_dtype, enabled=amp_enabled
        ):
            logits = model(images)
            loss = criterion(logits, labels)
        total_loss += loss.item()
        all_preds.append(torch.sigmoid(logits).cpu())

    avg_loss = total_loss / len(val_loader)
    y_true = torch.cat(all_labels).numpy().flatten()
    y_pred = torch.cat(all_preds).numpy().flatten()
    try:
        auc_score = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc_score = 0.5
    return avg_loss, auc_score


@torch.no_grad()
def predict_test(model, test_loader, device, amp_enabled, amp_dtype):
    """Generates predictions for the test set."""
    model.eval()
    all_preds, all_ids = [], []
    for images, ids in test_loader:
        images = images.to(device)
        with torch.autocast(
            device_type=device.split(":")[0], dtype=amp_dtype, enabled=amp_enabled
        ):
            logits = model(images)
        preds = torch.sigmoid(logits)
        all_preds.append(preds.float().cpu().numpy().flatten())
        all_ids.extend(ids)
    return np.concatenate(all_preds), all_ids


def main():
    """Main function to run the full training and evaluation pipeline."""

    # Step 1: Data
    train_dataset, val_dataset, test_df, processor = create_datasets()

    # Step 2: Model
    model, criterion = create_model_and_loss()

    # Step 3: Precision
    DEVICE, AMP_ENABLED, AMP_DTYPE, SCALER = setup_precision_environment()
    model.to(DEVICE)

    # Step 4: Training & Evaluation Orchestration
    print("--- Starting Training & Evaluation Stage ---")
    print(f"Batch Size: {BATCH_SIZE}, Num Workers: {NUM_WORKERS}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, total_steps=EPOCHS * len(train_loader)
    )

    best_auc = 0.0
    print(f"Starting training for {EPOCHS} epochs on device: {DEVICE}")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            DEVICE,
            SCALER,
            AMP_ENABLED,
            AMP_DTYPE,
        )
        val_loss, val_auc = validate(
            model, val_loader, criterion, DEVICE, AMP_ENABLED, AMP_DTYPE
        )
        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
        )
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Best model saved with VAL AUC: {best_auc:.4f}")

    print("--- Training Finished ---")
    print("Loading best model for test set inference...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    test_dataset = HistopathologyDataset(
        df=test_df, image_dir=TEST_IMG_DIR, processor=processor, is_test=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print("Running inference on test data...")
    predictions, ids = predict_test(model, test_loader, DEVICE, AMP_ENABLED, AMP_DTYPE)

    print("Creating submission file...")
    submission_df = pd.DataFrame({"id": ids, "label": predictions})
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission file saved to {SUBMISSION_PATH}")

    # Final line of output must be the validation score
    print(f"Final Validation Score: {best_auc}")


if __name__ == "__main__":
    main()
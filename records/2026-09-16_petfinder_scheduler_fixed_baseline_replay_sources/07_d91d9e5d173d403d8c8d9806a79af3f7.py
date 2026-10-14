import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

FEATURES = [
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
TARGET = "Pawpularity"

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
print(
    f"Target stats: mean={train_df[TARGET].mean():.2f}, std={train_df[TARGET].std():.2f}"
)

# Create stratified split (by Pawpularity decile)
train_df["decile"] = pd.qcut(train_df[TARGET], q=10, labels=False, duplicates="drop")
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
train_idx, val_idx = next(sss.split(train_df, train_df["decile"]))

train_split = train_df.iloc[train_idx].reset_index(drop=True)
val_split = train_df.iloc[val_idx].reset_index(drop=True)

print(f"Train split: {len(train_split)}, Val split: {len(val_split)}")
print(
    f"Val target stats: mean={val_split[TARGET].mean():.2f}, std={val_split[TARGET].std():.2f}"
)


class PetDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['Id']}.jpg")

        from PIL import Image

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            img = self.transform(img)

        meta_vals = np.array(row[FEATURES].values, dtype=np.float64)
        meta = torch.tensor(meta_vals, dtype=torch.float32)

        if TARGET in self.df.columns:
            label = torch.tensor(row[TARGET], dtype=torch.float32)
        else:
            label = torch.tensor(0.0, dtype=torch.float32)

        return img, meta, label


# Transforms
IMG_SIZE = 300
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


class PawpularityModel(nn.Module):
    def __init__(self, num_meta_features=12, dropout=0.3):
        super().__init__()
        # EfficientNet-B3 backbone
        self.backbone = timm.create_model(
            "tf_efficientnet_b3", pretrained=True, num_classes=0
        )
        feat_dim = self.backbone.num_features  # 1536 for b3

        # Metadata branch
        self.meta_net = nn.Sequential(
            nn.Linear(num_meta_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Fusion head
        fused_dim = feat_dim + 32
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

    def forward(self, images, meta):
        img_feat = self.backbone(images)  # (B, 1536)
        meta_feat = self.meta_net(meta)  # (B, 32)
        fused = torch.cat([img_feat, meta_feat], dim=1)
        out = self.head(fused).squeeze(-1)  # (B,)
        return out


def mixup_data(x1, x2, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x1.size(0)
    index = torch.randperm(batch_size).to(x1.device)
    mixed_x1 = lam * x1 + (1 - lam) * x1[index]
    mixed_x2 = x2[index]  # mix metadata too (simple approach)
    return mixed_x1, mixed_x2, lam, y, y[index]


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


os.makedirs("./working", exist_ok=True)

# Build datasets
train_dataset = PetDataset(
    train_split, "./input/train/", transform=train_transform, is_train=True
)
val_dataset = PetDataset(
    val_split, "./input/train/", transform=val_transform, is_train=False
)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
)

# Model
model = PawpularityModel(num_meta_features=len(FEATURES), dropout=0.3).to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Differential learning rates: backbone lower, head higher
backbone_params = list(model.backbone.parameters())
head_params = list(model.meta_net.parameters()) + list(model.head.parameters())

optimizer = torch.optim.AdamW(
    [
        {"params": backbone_params, "lr": 2e-5, "weight_decay": 1e-4},
        {"params": head_params, "lr": 2e-3, "weight_decay": 1e-3},
    ]
)

num_epochs = 30
total_steps = len(train_loader) * num_epochs
warmup_steps = len(train_loader) * 2  # 2 epochs warmup

from torch.optim.lr_scheduler import LambdaLR


def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(
        max(1, total_steps - warmup_steps)
    )
    return max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))


scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
criterion = nn.MSELoss()

# Training loop
best_val_rmse = float("inf")
patience = 8
patience_counter = 0
use_mixup = True
mixup_alpha = 0.2

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    n_batches = 0

    for batch_imgs, batch_meta, batch_labels in train_loader:
        batch_imgs = batch_imgs.to(DEVICE, non_blocking=True)
        batch_meta = batch_meta.to(DEVICE, non_blocking=True)
        batch_labels = batch_labels.to(DEVICE, non_blocking=True)

        if use_mixup and np.random.rand() > 0.1:  # 90% of batches use mixup
            imgs, metas, lam, y_a, y_b = mixup_data(
                batch_imgs, batch_meta, batch_labels, mixup_alpha
            )
            preds = model(imgs, metas)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)
        else:
            preds = model(batch_imgs, batch_meta)
            loss = criterion(preds, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        n_batches += 1

    avg_loss = running_loss / max(n_batches, 1)

    # Validation
    model.eval()
    val_preds = []
    val_true = []
    with torch.no_grad():
        for batch_imgs, batch_meta, batch_labels in val_loader:
            batch_imgs = batch_imgs.to(DEVICE, non_blocking=True)
            batch_meta = batch_meta.to(DEVICE, non_blocking=True)
            preds = model(batch_imgs, batch_meta)
            val_preds.extend(preds.cpu().numpy())
            val_true.extend(batch_labels.numpy())

    val_rmse = np.sqrt(mean_squared_error(val_true, val_preds))
    print(
        f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Val RMSE: {val_rmse:.4f}"
    )

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        patience_counter = 0
        torch.save(model.state_dict(), "./working/best_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load best model
model.load_state_dict(torch.load("./working/best_model.pth", map_location=DEVICE))
model.eval()

# Final validation score
val_preds_final = []
val_true_final = []
with torch.no_grad():
    for batch_imgs, batch_meta, batch_labels in val_loader:
        batch_imgs = batch_imgs.to(DEVICE, non_blocking=True)
        batch_meta = batch_meta.to(DEVICE, non_blocking=True)
        preds = model(batch_imgs, batch_meta)
        val_preds_final.extend(preds.cpu().numpy())
        val_true_final.extend(batch_labels.numpy())

final_val_rmse = np.sqrt(mean_squared_error(val_true_final, val_preds_final))
print(f"\nBest Validation RMSE: {final_val_rmse:.4f}")

# Generate test predictions
test_dataset = PetDataset(
    test_df, "./input/test/", transform=val_transform, is_train=False
)
test_loader = DataLoader(
    test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
)

test_preds = []
with torch.no_grad():
    for batch_imgs, batch_meta, _ in test_loader:
        batch_imgs = batch_imgs.to(DEVICE, non_blocking=True)
        batch_meta = batch_meta.to(DEVICE, non_blocking=True)
        preds = model(batch_imgs, batch_meta)
        test_preds.extend(preds.cpu().numpy())

# Clip predictions to valid range [1, 100]
test_preds = np.clip(test_preds, 1.0, 100.0)

# Save submission
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame({"Id": test_df["Id"], "Pawpularity": test_preds})
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved with {len(submission)} rows")
print(
    f"Prediction stats: mean={test_preds.mean():.2f}, std={test_preds.std():.2f}, min={test_preds.min():.2f}, max={test_preds.max():.2f}"
)

print(f"Final Validation Score: {final_val_rmse:.4f}")

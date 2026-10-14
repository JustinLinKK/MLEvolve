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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
LR_BACKBONE = 1e-05
LR_HEAD = 0.001
WEIGHT_DECAY = 0.0001
MIXUP_ALPHA = 0.3
NUM_WORKERS = 4
META_COLS = ['Accessory', 'Action', 'Blur', 'Collage', 'Eyes', 'Face', 'Group', 'Human', 'Info', 'Near', 'Occlusion', 'Subject Focus']

class PetDataset(Dataset):

    def __init__(self, df, img_dir, augment=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.augment = augment
        base_t = [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        if augment:
            self.transform = transforms.Compose([transforms.RandomHorizontalFlip(0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05), transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)), transforms.RandomGrayscale(p=0.1)] + base_t)
        else:
            self.transform = transforms.Compose(base_t)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_path = os.path.join(self.img_dir, f"{row['Id']}.jpg")
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        meta = row[META_COLS].values.astype(np.float32)
        target = float(row['Pawpularity']) / 100.0
        return (img, meta, target)

class PawpularityModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features
        self.meta_mlp = nn.Sequential(nn.Linear(len(META_COLS), 64), nn.ReLU(), nn.Dropout(0.3))
        fused_dim = feat_dim + 64
        self.head = nn.Sequential(nn.LayerNorm(fused_dim), nn.Linear(fused_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, images, meta):
        x = self.backbone(images)
        m = self.meta_mlp(meta)
        z = torch.cat([x, m], dim=1)
        out = self.head(z).squeeze(-1)
        return out

def lr_lambda(step):
    warmup_steps = int(0.05 * total_steps)
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1 + np.cos(np.pi * progress))

def mixup_data(x, y, alpha=MIXUP_ALPHA):
    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(DEVICE)
    one_minus_lam = 1.0 - lam
    mixed_x = lam * x + one_minus_lam * x[index]
    mixed_y = lam * y + one_minus_lam * y[index]
    return (mixed_x.to(x.dtype), mixed_y.to(y.dtype))
best_state = None
patience_counter = 0
PATIENCE = 5
all_preds, all_targets = ([], [])
test_preds_list = []


def build_model(context=None):
    return PawpularityModel()

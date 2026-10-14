import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
import warnings
SEED = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
FEATURE_COLS = ['Accessory', 'Action', 'Blur', 'Collage', 'Eyes', 'Face', 'Group', 'Human', 'Info', 'Near', 'Occlusion', 'Subject Focus']
IMG_SIZE = 224

class PetDataset(Dataset):

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
            img = Image.open(img_path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        feats = row[FEATURE_COLS].values.astype(np.float32)
        if self.is_train:
            target = np.float32(row['Pawpularity'])
            return (img, feats, target)
        return (img, feats)

class PawpularityModel(nn.Module):

    def __init__(self, num_features=12, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features
        self.vis_norm = nn.LayerNorm(feat_dim)
        self.meta_mlp = nn.Sequential(nn.Linear(num_features, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 64), nn.GELU())
        meta_out = 64
        self.head = nn.Sequential(nn.Linear(feat_dim + meta_out, 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))

    def forward(self, images, meta):
        vis_feat = self.backbone(images)
        vis_feat = self.vis_norm(vis_feat)
        meta_feat = self.meta_mlp(meta)
        x = torch.cat([vis_feat, meta_feat], dim=1)
        out = self.head(x)
        return out.squeeze(-1)
EPOCHS = 20
LR_BACKBONE = 2e-05
LR_HEAD = 0.0005
WEIGHT_DECAY = 0.0001

def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + np.cos(np.pi * progress))

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return (mixed_x, mixed_y)
best_state = None
patience = 6
patience_counter = 0
val_preds_final = []
test_preds = []


def build_model(context=None):
    return PawpularityModel()

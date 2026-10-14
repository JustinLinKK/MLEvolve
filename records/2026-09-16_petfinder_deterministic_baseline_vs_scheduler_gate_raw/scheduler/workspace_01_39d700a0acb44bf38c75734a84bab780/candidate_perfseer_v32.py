import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
MODEL_FAMILY = 'resnet50_multimodal_pawpularity'
TABULAR_COLS = ['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory', 'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur']
SUBMISSION_DIR = './submission'

class PawpularityDataset(Dataset):

    def __init__(self, df, img_dir, transform=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['Id']
        img_path = os.path.join(self.img_dir, f'{img_id}.jpg')
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (256, 256), (128, 128, 128))
        if self.transform:
            image = self.transform(image)
        tabular = row[TABULAR_COLS].values.astype(np.float32)
        if self.is_train:
            target = float(row['Pawpularity'])
            return (image, tabular, target)
        else:
            return (image, tabular, img_id)

class PawpularityModel(nn.Module):

    def __init__(self, num_tabular_features=12):
        super(PawpularityModel, self).__init__()
        from torchvision.models import resnet50
        self.backbone = resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.tabular_mlp = nn.Sequential(nn.Linear(num_tabular_features, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU())
        fusion_dim = in_features + 32
        self.head = nn.Sequential(nn.Linear(fusion_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, images, tabular):
        img_feat = self.backbone(images)
        tab_feat = self.tabular_mlp(tabular)
        fused = torch.cat([img_feat, tab_feat], dim=1)
        out = self.head(fused)
        return out.squeeze(-1)

def build_transforms(is_train=True):
    if is_train:
        return transforms.Compose([transforms.Resize((256, 256)), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        return transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(256), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def select_validated_precision():
    """Return precision decision for this task."""
    return {'precision': 'fp32', 'reason': 'No matched FP32/AMP comparison provided; defaulting to FP32'}

class CandidateAdapter:

    def __init__(self):
        self.default_context = {'batch_size': 8, 'num_epochs': 1, 'learning_rate': 0.0001, 'device': 'cpu', 'criterion': None, 'optimizer': None, 'model': None, 'train_dataset': None, 'val_dataset': None, 'train_loader': None, 'val_loader': None}

    def _merge_context(self, context):
        ctx = dict(self.default_context)
        if context is not None:
            ctx.update(context)
        return ctx

    def build_model(self, context):
        ctx = self._merge_context(context)
        device = ctx.get('device', 'cpu')
        model = PawpularityModel(num_tabular_features=len(TABULAR_COLS))
        model.to(device)
        return model

    def build_optimizer(self, model, context):
        ctx = self._merge_context(context)
        lr = ctx.get('learning_rate', 0.0001)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
        return optimizer

    def build_train_batch(self, scenario, device):
        bs = scenario.get('batch_size', 8)
        fixture = scenario.get('fixture', {})
        if fixture:
            inputs = {}
            for name, shape in fixture.items():
                full_shape = (bs,) + tuple(shape)
                if name == 'image':
                    inputs[name] = torch.randn(full_shape).to(device)
                elif name == 'tabular':
                    inputs[name] = torch.randint(0, 2, full_shape).float().to(device)
                elif name == 'target':
                    inputs[name] = torch.rand(full_shape).to(device)
            return inputs
        images = torch.randn(bs, 3, 256, 256).to(device)
        tabular = torch.randint(0, 2, (bs, len(TABULAR_COLS))).float().to(device)
        target = torch.rand(bs).to(device)
        return {'image': images, 'tabular': tabular, 'target': target}

    def build_validation_batch(self, scenario, device):
        return self.build_train_batch(scenario, device)

    def training_step(self, model, batch, context):
        ctx = self._merge_context(context)
        criterion = ctx.get('criterion')
        if criterion is None:
            criterion = nn.MSELoss()
        images = batch['image'].to(ctx.get('device', 'cpu'))
        tabular = batch['tabular'].to(ctx.get('device', 'cpu'))
        target = batch['target'].to(ctx.get('device', 'cpu'))
        pred = model(images, tabular)
        loss = criterion(pred, target)
        return loss

    def validation_step(self, model, batch, context):
        ctx = self._merge_context(context)
        criterion = ctx.get('criterion')
        if criterion is None:
            criterion = nn.MSELoss()
        images = batch['image'].to(ctx.get('device', 'cpu'))
        tabular = batch['tabular'].to(ctx.get('device', 'cpu'))
        target = batch['target'].to(ctx.get('device', 'cpu'))
        model.eval()
        with torch.no_grad():
            pred = model(images, tabular)
            loss = criterion(pred, target)
        return loss

def main():
    settings = select_validated_precision()
    precision_quality = settings['precision_quality'] if 'precision_quality' in settings else 'fp32'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    train_df = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))
    test_df = pd.read_csv(os.path.join(INPUT_DIR, 'test.csv'))
    print(f'Train samples: {len(train_df)}, Test samples: {len(test_df)}')
    train_subset, val_subset = train_test_split(train_df, test_size=0.1, random_state=42)
    print(f'Training: {len(train_subset)}, Validation: {len(val_subset)}')
    train_transform = build_transforms(is_train=True)
    val_transform = build_transforms(is_train=False)
    test_transform = build_transforms(is_train=False)
    train_dataset = PawpularityDataset(train_subset, os.path.join(INPUT_DIR, 'train'), train_transform, is_train=True)
    val_dataset = PawpularityDataset(val_subset, os.path.join(INPUT_DIR, 'train'), val_transform, is_train=True)
    test_dataset = PawpularityDataset(test_df, os.path.join(INPUT_DIR, 'test'), test_transform, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    model = PawpularityModel(num_tabular_features=len(TABULAR_COLS)).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
    num_epochs = 15
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-06)
    best_val_rmse = float('inf')
    best_model_state = None
    patience = 3
    patience_counter = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for images, tabular, targets in train_loader:
            images = images.to(device)
            tabular = tabular.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            preds = model(images, tabular)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        avg_train_loss = train_loss / max(n_batches, 1)
        scheduler.step()
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for images, tabular, targets in val_loader:
                images = images.to(device)
                tabular = tabular.to(device)
                targets = targets.to(device)
                preds = model(images, tabular)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        print(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val RMSE: {val_rmse:.4f}')
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    val_preds_final = []
    val_targets_final = []
    with torch.no_grad():
        for images, tabular, targets in val_loader:
            images = images.to(device)
            tabular = tabular.to(device)
            targets = targets.to(device)
            preds = model(images, tabular)
            val_preds_final.extend(preds.cpu().numpy())
            val_targets_final.extend(targets.cpu().numpy())
    final_val_rmse = np.sqrt(mean_squared_error(val_targets_final, val_preds_final))
    print(f'Best Validation RMSE: {final_val_rmse:.4f}')
    model.eval()
    test_preds = []
    test_ids = []
    with torch.no_grad():
        for images, tabular, ids in test_loader:
            images = images.to(device)
            tabular = tabular.to(device)
            preds = model(images, tabular)
            test_preds.extend(preds.cpu().numpy())
            test_ids.extend(ids)
    test_preds = np.clip(test_preds, 0, 100)
    submission = pd.DataFrame({'Id': test_ids, 'Pawpularity': test_preds})
    submission_path = os.path.join(SUBMISSION_DIR, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f'Submission saved to {submission_path} with {len(submission)} rows')
    print(f'Final Validation Score: {final_val_rmse}')


def build_model(context=None):
    return CandidateAdapter().build_model(context)

"""
Training script for pose-based free throw prediction.

Uses K-fold cross-validation for reliable evaluation on small datasets.
"""
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path
from tqdm import tqdm

from pose_dataset import PoseSequenceDataset
from models.stgcn import STGCN, SimplePoseNet, TemporalPoseNet


class PoseAugmentation:
    """Data augmentation for pose sequences."""

    def __init__(self, noise_std=0.01, scale_range=(0.9, 1.1), flip_prob=0.5):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.flip_prob = flip_prob

    def __call__(self, x):
        """
        Args:
            x: (C, T, V) pose tensor
        """
        # Add Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        # Random scale
        if self.scale_range[0] != 1.0 or self.scale_range[1] != 1.0:
            scale = np.random.uniform(*self.scale_range)
            x = x * scale

        # Random horizontal flip (negate x-coordinate)
        if np.random.rand() < self.flip_prob:
            x[0] = -x[0]  # Flip x-axis

        return x


class AugmentedDataset(torch.utils.data.Dataset):
    """Wrapper to apply augmentation."""

    def __init__(self, dataset, augmentation=None):
        self.dataset = dataset
        self.augmentation = augmentation

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.augmentation:
            x = self.augmentation(x)
        return x, y


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return avg_loss, acc, precision, recall, f1, all_preds, all_labels


def train_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, patience):
    """Train a single fold with early stopping."""
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    return best_val_acc


def cross_validate(model_class, model_kwargs, data, n_folds=5, epochs=100, batch_size=8,
                   lr=1e-3, weight_decay=1e-4, patience=15, device='cuda', augment=True,
                   use_enhanced=False):
    """
    K-fold cross-validation.

    Returns:
        dict with mean and std of metrics
    """
    # Create full dataset
    full_dataset = PoseSequenceDataset(data, normalize=True, use_enhanced=use_enhanced)
    labels = np.array([d['label'] for d in data])

    # Setup augmentation
    aug = PoseAugmentation() if augment else None

    # K-fold
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(labels)), labels)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        # Create train/val subsets
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        # Apply augmentation to training
        if augment:
            train_subset = AugmentedDataset(train_subset, aug)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Create model
        model = model_class(**model_kwargs).to(device)

        # Compute class weights
        train_labels = labels[train_idx]
        n_made = train_labels.sum()
        n_miss = len(train_labels) - n_made
        class_weights = torch.FloatTensor([len(train_labels) / (2 * n_miss),
                                           len(train_labels) / (2 * n_made)]).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        # Train
        best_val_acc = train_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, patience)

        # Final evaluation
        val_loss, val_acc, val_prec, val_rec, val_f1, preds, true_labels = evaluate(model, val_loader, criterion, device)
        cm = confusion_matrix(true_labels, preds)

        fold_results.append({
            'accuracy': val_acc,
            'precision': val_prec,
            'recall': val_rec,
            'f1': val_f1,
            'confusion_matrix': cm
        })

        print(f"  Accuracy: {val_acc:.4f}")
        print(f"  Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

    # Aggregate results
    results = {
        'accuracy': np.mean([r['accuracy'] for r in fold_results]),
        'accuracy_std': np.std([r['accuracy'] for r in fold_results]),
        'precision': np.mean([r['precision'] for r in fold_results]),
        'recall': np.mean([r['recall'] for r in fold_results]),
        'f1': np.mean([r['f1'] for r in fold_results]),
        'f1_std': np.std([r['f1'] for r in fold_results]),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Train pose-based free throw predictor')
    parser.add_argument('--data', type=str, default='data/features/features_clean.json')
    parser.add_argument('--model', type=str, default='stgcn', choices=['stgcn', 'mlp', 'temporal'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--enhanced', action='store_true', help='Use enhanced features (pos + vel + accel)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"Device: {args.device}")

    # Load data
    with open(args.data) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} sequences")
    print(f"Labels: {sum(d['label'] for d in data)} made, {len(data) - sum(d['label'] for d in data)} miss")
    print(f"Enhanced features: {args.enhanced}")

    # Input channels: 3 (xyz) or 9 (xyz + vel + accel)
    in_channels = 9 if args.enhanced else 3

    # Model configs
    model_configs = {
        'stgcn': (STGCN, {'num_classes': 2, 'num_joints': 70, 'in_channels': in_channels,
                         'hidden_channels': 64, 'num_layers': 3, 'dropout': 0.3}),
        'mlp': (SimplePoseNet, {'num_classes': 2, 'num_joints': 70, 'in_channels': in_channels,
                                'hidden_dim': 256, 'dropout': 0.5}),
        'temporal': (TemporalPoseNet, {'num_classes': 2, 'num_joints': 70, 'in_channels': in_channels,
                                       'hidden_channels': 64, 'dropout': 0.3}),
    }

    model_class, model_kwargs = model_configs[args.model]

    print(f"\nTraining {args.model.upper()} with {args.n_folds}-fold cross-validation...")

    results = cross_validate(
        model_class=model_class,
        model_kwargs=model_kwargs,
        data=data,
        n_folds=args.n_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
        augment=not args.no_augment,
        use_enhanced=args.enhanced
    )

    print(f"\n{'='*50}")
    print(f"Final Results ({args.model.upper()}):")
    print(f"  Accuracy: {results['accuracy']:.4f} +/- {results['accuracy_std']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1: {results['f1']:.4f} +/- {results['f1_std']:.4f}")


if __name__ == '__main__':
    main()

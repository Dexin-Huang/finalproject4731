"""
Training script for release frame detector.

Trains a model to predict the exact frame where the ball is released
from a sequence of per-frame features.
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from models.release_detector import (
    get_model,
    create_gaussian_labels,
    gaussian_focal_loss,
    FEATURE_NAMES,
)


class ReleaseFeatureDataset(Dataset):
    """Dataset for release frame detection."""

    def __init__(
        self,
        features_data: List[Dict],
        feature_names: List[str],
        seq_len: int = 30,
        gaussian_sigma: float = 1.5,
        augment: bool = False,
    ):
        self.data = features_data
        self.feature_names = feature_names
        self.seq_len = seq_len
        self.gaussian_sigma = gaussian_sigma
        self.augment = augment

        # Preprocess data
        self.samples = []
        for item in features_data:
            features_seq = item['features']
            if len(features_seq) < seq_len:
                continue

            # Find release frame index within the sequence
            release_frame = item['release_frame']
            window_start = item.get('window_start', 0)

            # Release index relative to sequence start
            release_idx = release_frame - window_start

            if release_idx < 0 or release_idx >= len(features_seq):
                continue

            self.samples.append({
                'video_id': item['video_id'],
                'features': features_seq,
                'release_idx': release_idx,
                'confidence': item.get('confidence', 'high'),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def _extract_features(self, frame_data: Optional[Dict]) -> np.ndarray:
        """Extract feature vector from frame data."""
        if frame_data is None:
            return np.zeros(len(self.feature_names))

        features = []
        for name in self.feature_names:
            value = frame_data.get(name)
            if value is None:
                value = 0.0
            elif isinstance(value, bool):
                value = float(value)
            features.append(float(value))

        return np.array(features, dtype=np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        sample = self.samples[idx]
        features_seq = sample['features']
        release_idx = sample['release_idx']

        # Ensure consistent sequence length
        if len(features_seq) > self.seq_len:
            # Center on release frame
            start = max(0, release_idx - self.seq_len // 2)
            end = start + self.seq_len
            if end > len(features_seq):
                end = len(features_seq)
                start = end - self.seq_len
            features_seq = features_seq[start:end]
            release_idx = release_idx - start
        elif len(features_seq) < self.seq_len:
            # Pad with zeros
            padding = [None] * (self.seq_len - len(features_seq))
            features_seq = features_seq + padding

        # Extract features
        X = np.stack([self._extract_features(f) for f in features_seq])

        # Data augmentation
        if self.augment:
            # Add noise
            noise = np.random.randn(*X.shape) * 0.01
            X = X + noise

            # Random temporal shift (±2 frames)
            shift = np.random.randint(-2, 3)
            if shift != 0:
                X = np.roll(X, shift, axis=0)
                release_idx = np.clip(release_idx + shift, 0, self.seq_len - 1)

        # Normalize features
        X = self._normalize(X)

        # Create soft labels
        y = create_gaussian_labels(release_idx, self.seq_len, self.gaussian_sigma)

        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            release_idx,
        )

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize features to zero mean and unit variance."""
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)

        # Per-feature normalization
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-8
        X = (X - mean) / std

        return X


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_focal_loss: bool = True,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_mae = 0.0
    total_within_1 = 0
    total_within_3 = 0
    num_samples = 0

    for X, y, release_idx in dataloader:
        X = X.to(device)
        y = y.to(device)
        release_idx = release_idx.to(device)

        optimizer.zero_grad()

        logits = model(X)

        # Loss
        if use_focal_loss:
            loss = gaussian_focal_loss(logits, y, gamma=2.0)
        else:
            loss = F.cross_entropy(logits, y)

        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            pred_idx = logits.argmax(dim=-1)
            mae = (pred_idx - release_idx).abs().float().mean()
            within_1 = ((pred_idx - release_idx).abs() <= 1).sum()
            within_3 = ((pred_idx - release_idx).abs() <= 3).sum()

        total_loss += loss.item() * X.size(0)
        total_mae += mae.item() * X.size(0)
        total_within_1 += within_1.item()
        total_within_3 += within_3.item()
        num_samples += X.size(0)

    return {
        'loss': total_loss / num_samples,
        'mae': total_mae / num_samples,
        'within_1': total_within_1 / num_samples,
        'within_3': total_within_3 / num_samples,
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()

    total_mae = 0.0
    total_within_1 = 0
    total_within_3 = 0
    num_samples = 0
    all_errors = []

    with torch.no_grad():
        for X, y, release_idx in dataloader:
            X = X.to(device)
            release_idx = release_idx.to(device)

            logits = model(X)
            pred_idx = logits.argmax(dim=-1)

            errors = (pred_idx - release_idx).abs()
            all_errors.extend(errors.cpu().numpy().tolist())

            mae = errors.float().mean()
            within_1 = (errors <= 1).sum()
            within_3 = (errors <= 3).sum()

            total_mae += mae.item() * X.size(0)
            total_within_1 += within_1.item()
            total_within_3 += within_3.item()
            num_samples += X.size(0)

    return {
        'mae': total_mae / num_samples,
        'within_1': total_within_1 / num_samples,
        'within_3': total_within_3 / num_samples,
        'median_error': np.median(all_errors),
        'max_error': max(all_errors),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
) -> Tuple[nn.Module, Dict[str, float]]:
    """Train model with early stopping."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_mae = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.2f}")
        print(f"  Val   - MAE: {val_metrics['mae']:.2f}, Within-1: {val_metrics['within_1']*100:.1f}%, "
              f"Within-3: {val_metrics['within_3']*100:.1f}%")

        # Early stopping
        if val_metrics['mae'] < best_mae:
            best_mae = val_metrics['mae']
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = evaluate(model, val_loader, device)
    return model, final_metrics


def run_cross_validation(
    features_data: List[Dict],
    model_type: str = 'cnn',
    n_folds: int = 5,
    num_epochs: int = 50,
    batch_size: int = 8,
    seq_len: int = 30,
    lr: float = 1e-3,
    device: str = 'cuda',
):
    """Run k-fold cross-validation."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Filter valid samples
    valid_data = [
        d for d in features_data
        if len(d['features']) >= seq_len
    ]
    print(f"Valid samples: {len(valid_data)}/{len(features_data)}")

    if len(valid_data) < n_folds:
        print("Not enough samples for cross-validation!")
        return

    # K-fold CV
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    indices = np.arange(len(valid_data))

    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        print(f"\n{'='*50}")
        print(f"Fold {fold+1}/{n_folds}")
        print(f"{'='*50}")

        train_data = [valid_data[i] for i in train_idx]
        val_data = [valid_data[i] for i in val_idx]

        # Create datasets
        train_dataset = ReleaseFeatureDataset(
            train_data, FEATURE_NAMES, seq_len=seq_len, augment=True
        )
        val_dataset = ReleaseFeatureDataset(
            val_data, FEATURE_NAMES, seq_len=seq_len, augment=False
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        # Create model
        num_features = len(FEATURE_NAMES)
        model = get_model(model_type, num_features=num_features)
        model = model.to(device)

        # Train
        model, metrics = train_model(
            model, train_loader, val_loader, device,
            num_epochs=num_epochs, lr=lr,
        )

        fold_results.append(metrics)
        print(f"\nFold {fold+1} Results:")
        print(f"  MAE: {metrics['mae']:.2f} frames")
        print(f"  Within-1: {metrics['within_1']*100:.1f}%")
        print(f"  Within-3: {metrics['within_3']*100:.1f}%")

    # Summary
    print(f"\n{'='*50}")
    print("Cross-Validation Summary")
    print(f"{'='*50}")

    mae_values = [r['mae'] for r in fold_results]
    within_1_values = [r['within_1'] for r in fold_results]
    within_3_values = [r['within_3'] for r in fold_results]

    print(f"MAE: {np.mean(mae_values):.2f} ± {np.std(mae_values):.2f} frames")
    print(f"Within-1: {np.mean(within_1_values)*100:.1f}% ± {np.std(within_1_values)*100:.1f}%")
    print(f"Within-3: {np.mean(within_3_values)*100:.1f}% ± {np.std(within_3_values)*100:.1f}%")

    return fold_results


def train_final_model(
    features_data: List[Dict],
    model_type: str = 'cnn',
    output_path: str = 'models/release_detector.pt',
    num_epochs: int = 100,
    batch_size: int = 8,
    seq_len: int = 30,
    lr: float = 1e-3,
    device: str = 'cuda',
):
    """Train final model on all data and save."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Filter valid samples
    valid_data = [
        d for d in features_data
        if len(d['features']) >= seq_len
    ]
    print(f"Training on {len(valid_data)} samples")

    # Split into train/val (90/10)
    np.random.seed(42)
    indices = np.random.permutation(len(valid_data))
    val_size = max(1, len(valid_data) // 10)
    train_idx = indices[val_size:]
    val_idx = indices[:val_size]

    train_data = [valid_data[i] for i in train_idx]
    val_data = [valid_data[i] for i in val_idx]

    # Create datasets
    train_dataset = ReleaseFeatureDataset(
        train_data, FEATURE_NAMES, seq_len=seq_len, augment=True
    )
    val_dataset = ReleaseFeatureDataset(
        val_data, FEATURE_NAMES, seq_len=seq_len, augment=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Create model
    num_features = len(FEATURE_NAMES)
    model = get_model(model_type, num_features=num_features)
    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    model, metrics = train_model(
        model, train_loader, val_loader, device,
        num_epochs=num_epochs, lr=lr, patience=20,
    )

    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'num_features': num_features,
        'seq_len': seq_len,
        'feature_names': FEATURE_NAMES,
        'metrics': metrics,
    }, output_path)

    print(f"\nModel saved to {output_path}")
    print(f"Final metrics: MAE={metrics['mae']:.2f}, Within-3={metrics['within_3']*100:.1f}%")

    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Train release frame detector")
    parser.add_argument('--features', type=str, default='data/release_features/features.json',
                        help='Path to extracted features')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'lstm', 'transformer'],
                        help='Model architecture')
    parser.add_argument('--output', type=str, default='models/release_detector.pt',
                        help='Output path for trained model')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--seq_len', type=int, default=30,
                        help='Sequence length')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--cv', action='store_true',
                        help='Run cross-validation instead of final training')
    parser.add_argument('--folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    args = parser.parse_args()

    # Load features
    print(f"Loading features from {args.features}")
    with open(args.features) as f:
        features_data = json.load(f)
    print(f"Loaded {len(features_data)} samples")

    if args.cv:
        # Cross-validation
        run_cross_validation(
            features_data,
            model_type=args.model,
            n_folds=args.folds,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lr=args.lr,
            device=args.device,
        )
    else:
        # Train final model
        train_final_model(
            features_data,
            model_type=args.model,
            output_path=args.output,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lr=args.lr,
            device=args.device,
        )


if __name__ == '__main__':
    main()

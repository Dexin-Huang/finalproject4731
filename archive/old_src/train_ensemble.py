"""
Ensemble training with multiple seeds and model averaging.

Strategy: Train multiple models with different random seeds,
then average their predictions for more robust results.
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from pose_dataset import PoseSequenceDataset
from models.stgcn import STGCN, TemporalPoseNet


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()


class PoseAugmentation:
    def __init__(self, noise_std=0.015, scale_range=(0.9, 1.1), flip_prob=0.5):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.flip_prob = flip_prob

    def __call__(self, x):
        x = x + torch.randn_like(x) * self.noise_std
        x = x * np.random.uniform(*self.scale_range)
        if np.random.rand() < self.flip_prob:
            x[0] = -x[0]
        return x


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, aug=None):
        self.dataset = dataset
        self.aug = aug

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.aug:
            x = self.aug(x)
        return x, y


def train_single_model(model, train_loader, val_loader, criterion, device, epochs=100, lr=1e-3):
    """Train a single model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    best_state = None
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validate
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                preds.extend(model(x).argmax(dim=1).cpu().numpy())
                labels.extend(y.numpy())

        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_acc


def ensemble_predict(models, x, device):
    """Average predictions from multiple models."""
    all_probs = []
    for model in models:
        model.eval()
        with torch.no_grad():
            logits = model(x.to(device))
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
    return torch.stack(all_probs).mean(dim=0)


def main():
    with open('data/features/enhanced_all.json') as f:
        data = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_ensemble = 5  # Models per fold
    n_folds = 5

    print("="*60)
    print("ENSEMBLE TRAINING")
    print("="*60)
    print(f"Samples: {len(data)}")
    print(f"Ensemble size: {n_ensemble} models per fold")
    print(f"Device: {device}")

    full_dataset = PoseSequenceDataset(data, normalize=True, use_enhanced=True)
    labels = np.array([d['label'] for d in data])
    aug = PoseAugmentation()
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_probs = []
    all_labels = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(labels)), labels)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'='*50}")

        train_subset = AugmentedDataset(Subset(full_dataset, train_idx), aug)
        val_subset = Subset(full_dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

        # Class weights
        train_labels = labels[train_idx]
        n_made = train_labels.sum()
        n_miss = len(train_labels) - n_made
        class_weights = torch.FloatTensor([len(train_labels)/(2*n_miss),
                                           len(train_labels)/(2*n_made)]).to(device)

        # Train ensemble
        models = []
        for i in range(n_ensemble):
            torch.manual_seed(42 + i * 100 + fold)
            np.random.seed(42 + i * 100 + fold)

            # Alternate between model types
            if i % 2 == 0:
                model = STGCN(num_classes=2, num_joints=70, in_channels=9,
                             hidden_channels=64, dropout=0.3).to(device)
            else:
                model = TemporalPoseNet(num_classes=2, num_joints=70, in_channels=9,
                                       hidden_channels=64, dropout=0.3).to(device)

            criterion = FocalLoss(alpha=class_weights, gamma=2.0)
            model, acc = train_single_model(model, train_loader, val_loader, criterion, device, epochs=80)
            models.append(model)
            print(f"  Model {i+1}: {type(model).__name__} - val_acc={acc:.4f}")

        # Ensemble evaluation
        fold_probs = []
        fold_labels = []

        for x, y in val_loader:
            probs = ensemble_predict(models, x, device)
            fold_probs.append(probs)
            fold_labels.extend(y.numpy())

        fold_probs = torch.cat(fold_probs)
        fold_labels = np.array(fold_labels)
        fold_preds = fold_probs.argmax(dim=1).numpy()

        acc = accuracy_score(fold_labels, fold_preds)
        f1 = f1_score(fold_labels, fold_preds)
        auc = roc_auc_score(fold_labels, fold_probs[:, 1].numpy())
        cm = confusion_matrix(fold_labels, fold_preds)

        print(f"\n  Ensemble Result:")
        print(f"    Accuracy: {acc:.4f}")
        print(f"    F1: {f1:.4f}")
        print(f"    AUC: {auc:.4f}")
        print(f"    CM:\n{cm}")

        all_probs.append(fold_probs[:, 1].numpy())
        all_labels.append(fold_labels)

    # Final aggregation
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_preds = (all_probs > 0.5).astype(int)

    print("\n" + "="*60)
    print("FINAL ENSEMBLE RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Overall F1: {f1_score(all_labels, all_preds):.4f}")
    print(f"Overall AUC: {roc_auc_score(all_labels, all_probs):.4f}")

    # Confidence analysis
    high_conf_miss = all_probs < 0.3
    high_conf_make = all_probs > 0.7

    if high_conf_miss.sum() > 0:
        miss_acc = (all_labels[high_conf_miss] == 0).mean()
        print(f"\nHigh-conf MISS (prob<0.3): {miss_acc:.1%} ({high_conf_miss.sum()} samples)")

    if high_conf_make.sum() > 0:
        make_acc = (all_labels[high_conf_make] == 1).mean()
        print(f"High-conf MAKE (prob>0.7): {make_acc:.1%} ({high_conf_make.sum()} samples)")

    # Optimal threshold analysis
    print("\n" + "="*60)
    print("THRESHOLD ANALYSIS")
    print("="*60)
    for thresh in [0.25, 0.30, 0.35, 0.40]:
        miss_pred = all_probs < thresh
        if miss_pred.sum() > 0:
            miss_acc = (all_labels[miss_pred] == 0).mean()
            print(f"  prob < {thresh}: {miss_acc:.1%} miss accuracy ({miss_pred.sum()} samples)")


if __name__ == '__main__':
    main()

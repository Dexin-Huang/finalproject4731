"""
Train using only key joints for the shooting motion.

Hypothesis: Reducing noise by focusing on shooting arm joints
(shoulder, elbow, wrist, hand) may improve signal.

Usage:
    python src/train_key_joints.py --data data/features/enhanced_all.json
    python src/train_key_joints.py --data data/features/enhanced_all.json --output models/my_model.pth
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import argparse
import os
from datetime import datetime

KEY_JOINTS = list(range(0, 25))


class KeyJointDataset(Dataset):
    """Dataset using only key joints."""

    def __init__(self, data, key_joints=None, normalize=True):
        # Filter unlabeled data
        self.data = [d for d in data if d.get('label', -1) != -1]
        if len(self.data) == 0:
            raise ValueError("No labeled data found!")

        # Auto-detect number of joints
        sample = self.data[0]
        if 'keypoints_3d' in sample:
            total_joints = len(sample['keypoints_3d'][0])
        else:
            total_joints = 70

        # Use all joints if fewer than expected
        if key_joints is None:
            if total_joints <= 25:
                key_joints = list(range(total_joints))
            else:
                key_joints = KEY_JOINTS

        self.key_joints = key_joints
        self.n_joints = len(key_joints)

        self.poses = []
        self.velocities = []
        self.accelerations = []
        self.labels = []

        for seq in self.data:
            try:
                kp = np.array(seq['keypoints_3d'])[:, key_joints, :]
                vel = np.array(seq['velocity'])[:, key_joints, :]
                accel = np.array(seq['acceleration'])[:, key_joints, :]

                self.poses.append(kp)
                self.velocities.append(vel)
                self.accelerations.append(accel)
                self.labels.append(seq['label'])
            except (KeyError, IndexError):
                continue

        if len(self.labels) == 0:
            raise ValueError("No valid sequences found!")

        self.poses = np.array(self.poses)
        self.velocities = np.array(self.velocities)
        self.accelerations = np.array(self.accelerations)
        self.labels = np.array(self.labels)

        if normalize:
            center = self.poses[:, :, 0:1, :]
            self.poses = self.poses - center
            dists = np.linalg.norm(self.poses, axis=-1)
            max_dist = dists.max(axis=(1, 2), keepdims=True)
            max_dist = np.maximum(max_dist, 1e-6)
            self.poses = self.poses / max_dist[:, :, :, None]
            self.velocities = self.velocities / max_dist[:, :, :, None]
            self.accelerations = self.accelerations / max_dist[:, :, :, None]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pose = self.poses[idx].transpose(2, 0, 1)
        vel = self.velocities[idx].transpose(2, 0, 1)
        accel = self.accelerations[idx].transpose(2, 0, 1)
        features = np.concatenate([pose, vel, accel], axis=0)
        return torch.FloatTensor(features), torch.LongTensor([self.labels[idx]])[0]


class KeyJointNet(nn.Module):
    """Network for key joints."""

    def __init__(self, num_joints=25, in_channels=9, hidden_dim=64, dropout=0.4):
        super().__init__()
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        self.joint_attn = nn.Sequential(
            nn.Linear(num_joints, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_joints),
            nn.Softmax(dim=-1)
        )

        self.temporal = nn.Sequential(
            nn.Conv1d(in_channels * num_joints, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 2)
        )

    def forward(self, x):
        N, C, T, V = x.shape
        joint_var = x.var(dim=(1, 2))
        attn = self.joint_attn(joint_var)
        x = x * attn.unsqueeze(1).unsqueeze(2)
        x = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        x = self.temporal(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def main():
    parser = argparse.ArgumentParser(description='Train key joints model')
    parser.add_argument('--data', type=str, default='data/features/enhanced_all.json')
    parser.add_argument('--output', type=str, default='models/best_key_joints_model.pth')
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_folds = 5

    print("=" * 60)
    print("KEY JOINT TRAINING")
    print("=" * 60)
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Total samples in file: {len(data)}")
    print(f"Device: {device}")

    full_dataset = KeyJointDataset(data, key_joints=None)
    num_joints = full_dataset.n_joints
    key_joints = full_dataset.key_joints
    labels = full_dataset.labels  # FIXED: use dataset labels

    print(f"Labeled samples: {len(labels)}")
    print(f"Key joints: {num_joints}")

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_probs, all_labels, fold_accs = [], [], []
    best_model_state, best_model_acc = None, 0

    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(labels)), labels)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=8, shuffle=True)
        val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=8, shuffle=False)

        train_labels = labels[train_idx]
        n_made = train_labels.sum()
        n_miss = len(train_labels) - n_made
        weights = torch.FloatTensor([len(train_labels) / (2 * n_miss),
                                     len(train_labels) / (2 * n_made)]).to(device)

        model = KeyJointNet(num_joints=num_joints, in_channels=9).to(device)
        criterion = FocalLoss(alpha=weights, gamma=2.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

        best_acc, best_state = 0, None
        patience, patience_counter = 15, 0

        for epoch in range(80):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            preds, lbls = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    preds.extend(model(x.to(device)).argmax(dim=1).cpu().numpy())
                    lbls.extend(y.numpy())

            acc = accuracy_score(lbls, preds)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        model.load_state_dict(best_state)
        model.eval()

        if best_acc > best_model_acc:
            best_model_acc = best_acc
            best_model_state = best_state

        probs, lbls = [], []
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x.to(device))
                probs.append(F.softmax(logits, dim=1)[:, 1].cpu())
                lbls.extend(y.numpy())

        probs = torch.cat(probs).numpy()
        lbls = np.array(lbls)
        preds = (probs > 0.5).astype(int)

        acc = accuracy_score(lbls, preds)
        f1 = f1_score(lbls, preds)
        auc = roc_auc_score(lbls, probs)
        cm = confusion_matrix(lbls, preds)

        fold_accs.append(acc)
        all_probs.append(probs)
        all_labels.append(lbls)

        print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        print(f"  CM: TN={cm[0, 0]}, FP={cm[0, 1]}, FN={cm[1, 0]}, TP={cm[1, 1]}")

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_preds = (all_probs > 0.5).astype(int)

    print("\n" + "=" * 60)
    print("FINAL RESULTS (Key Joints)")
    print("=" * 60)
    print(f"Accuracy: {np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}")
    print(f"Overall Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Overall AUC: {roc_auc_score(all_labels, all_probs):.4f}")

    print("\nConfidence Analysis:")
    for thresh in [0.25, 0.30, 0.35, 0.40]:
        miss_pred = all_probs < thresh
        if miss_pred.sum() > 0:
            miss_acc = (all_labels[miss_pred] == 0).mean()
            print(f"  prob < {thresh}: {miss_acc:.1%} miss acc ({miss_pred.sum()} samples)")

    for thresh in [0.60, 0.65, 0.70, 0.75]:
        make_pred = all_probs > thresh
        if make_pred.sum() > 0:
            make_acc = (all_labels[make_pred] == 1).mean()
            print(f"  prob > {thresh}: {make_acc:.1%} make acc ({make_pred.sum()} samples)")

    if best_model_state is not None:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        checkpoint = {
            'model_state_dict': best_model_state,
            'model_class': 'KeyJointNet',
            'model_config': {'num_joints': num_joints, 'in_channels': 9, 'hidden_dim': 64},
            'key_joints': key_joints,
            'mean_accuracy': float(np.mean(fold_accs)),
            'std_accuracy': float(np.std(fold_accs)),
            'overall_auc': float(roc_auc_score(all_labels, all_probs)),
            'num_samples': len(all_labels),
            'timestamp': datetime.now().isoformat(),
        }
        torch.save(checkpoint, args.output)
        print(f"\nModel saved to: {args.output}")


if __name__ == '__main__':
    main()
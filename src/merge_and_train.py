"""
Merge curated and MHR70 datasets into a balanced training set.

Combines:
- Original curated: 42 makes, 60 misses
- New MHR70: 231 makes, 72 misses

Target: ~130 samples per class (all misses + undersampled makes)
"""
import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset

from training_utils import FocalLoss, KeyJointNet


# Key upper body joints for shooting analysis
KEY_JOINTS = [
    0,   # nose
    5, 6,   # shoulders
    7, 8,   # elbows
    41, 62,  # wrists
    9, 10,  # hips
    69,  # neck
    # Right hand key joints
    21, 25, 29,  # fingertips (thumb, index, middle)
    # Elbow detail
    64, 66,  # right olecranon, cubital fossa
]


def merge_datasets(orig_path, mhr70_path, target_per_class=130, seed=42):
    """Merge two datasets with balanced sampling."""
    random.seed(seed)
    np.random.seed(seed)

    with open(orig_path) as f:
        orig = json.load(f)
    with open(mhr70_path) as f:
        mhr70 = json.load(f)

    # Separate by label
    orig_makes = [d for d in orig if d.get('label', -1) == 1]
    orig_misses = [d for d in orig if d.get('label', -1) == 0]
    mhr70_makes = [d for d in mhr70 if d.get('label', -1) == 1]
    mhr70_misses = [d for d in mhr70 if d.get('label', -1) == 0]

    print(f"Original: {len(orig_makes)} makes, {len(orig_misses)} misses")
    print(f"MHR70: {len(mhr70_makes)} makes, {len(mhr70_misses)} misses")

    # Combine all misses
    all_misses = orig_misses + mhr70_misses
    print(f"Combined misses: {len(all_misses)}")

    # Combine and undersample makes
    all_makes = orig_makes + mhr70_makes
    n_target_makes = min(len(all_misses), target_per_class)

    if len(all_makes) > n_target_makes:
        # Stratified sampling: try to keep proportions from both sources
        orig_ratio = len(orig_makes) / len(all_makes)
        n_orig = int(n_target_makes * orig_ratio)
        n_mhr70 = n_target_makes - n_orig

        # Sample from each source
        sampled_orig_makes = random.sample(orig_makes, min(n_orig, len(orig_makes)))
        remaining = n_target_makes - len(sampled_orig_makes)
        sampled_mhr70_makes = random.sample(mhr70_makes, min(remaining, len(mhr70_makes)))

        sampled_makes = sampled_orig_makes + sampled_mhr70_makes
    else:
        sampled_makes = all_makes

    print(f"Sampled makes: {len(sampled_makes)}")

    # Tag samples with source for analysis
    for d in all_misses:
        d['source'] = 'orig' if d in orig_misses else 'mhr70'
    for d in sampled_makes:
        d['source'] = 'orig' if d in orig_makes else 'mhr70'

    merged = sampled_makes + all_misses
    random.shuffle(merged)

    return merged


class MergedDataset(Dataset):
    def __init__(self, data, key_joints=KEY_JOINTS):
        self.data = [d for d in data if d.get('label', -1) != -1]
        self.joint_indices = key_joints
        self.n_joints = len(key_joints)

        self.features = []
        self.labels = []
        self.sources = []

        for seq in self.data:
            try:
                kp = np.array(seq['keypoints_3d'])[:, self.joint_indices, :]
                vel = np.array(seq['velocity'])[:, self.joint_indices, :]
                accel = np.array(seq['acceleration'])[:, self.joint_indices, :]

                # Normalize: center on hips
                hip_idx_9 = self.joint_indices.index(9) if 9 in self.joint_indices else 0
                hip_idx_10 = self.joint_indices.index(10) if 10 in self.joint_indices else 0
                center = (kp[:, hip_idx_9:hip_idx_9+1, :] + kp[:, hip_idx_10:hip_idx_10+1, :]) / 2
                kp = kp - center

                scale = np.abs(kp).max()
                if scale > 1e-6:
                    kp = kp / scale
                    vel = vel / scale
                    accel = accel / scale

                # Combine: (9, 4, n_joints)
                feat = np.concatenate([
                    kp.transpose(2, 0, 1),
                    vel.transpose(2, 0, 1),
                    accel.transpose(2, 0, 1)
                ], axis=0)

                self.features.append(feat)
                self.labels.append(seq['label'])
                self.sources.append(seq.get('source', 'unknown'))
            except Exception as e:
                continue

        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels)
        self.sources = np.array(self.sources)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]), torch.tensor(self.labels[idx], dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig', type=str, default='data/features/enhanced_all.json')
    parser.add_argument('--mhr70', type=str, default='data/features/enhanced_mhr70.json')
    parser.add_argument('--output', type=str, default='models/best_merged_model.pth')
    parser.add_argument('--save-merged', type=str, default='data/features/merged_balanced.json')
    args = parser.parse_args()

    print("=" * 60)
    print("MERGED BALANCED TRAINING")
    print("=" * 60)

    # Merge datasets
    merged_data = merge_datasets(args.orig, args.mhr70, target_per_class=132)

    # Save merged dataset
    if args.save_merged:
        os.makedirs(os.path.dirname(args.save_merged) or '.', exist_ok=True)
        with open(args.save_merged, 'w') as f:
            json.dump(merged_data, f)
        print(f"Saved merged dataset to {args.save_merged}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = MergedDataset(merged_data)
    labels = dataset.labels
    sources = dataset.sources
    num_joints = dataset.n_joints

    print(f"\nMerged dataset: {len(labels)} samples")
    print(f"Makes: {(labels == 1).sum()}, Misses: {(labels == 0).sum()}")
    print(f"From orig: {(sources == 'orig').sum()}, From mhr70: {(sources == 'mhr70').sum()}")
    print(f"Key joints: {num_joints}")
    print(f"Device: {device}")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_probs, all_labels, fold_accs = [], [], []
    best_model_state, best_model_acc = None, 0

    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(labels)), labels)):
        print(f"\n--- Fold {fold + 1}/5 ---")

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=16, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=16)

        # Balanced weights
        train_labels = labels[train_idx]
        n_made = train_labels.sum()
        n_miss = len(train_labels) - n_made
        weights = torch.FloatTensor([len(train_labels) / (2 * n_miss),
                                     len(train_labels) / (2 * n_made)]).to(device)

        model = KeyJointNet(num_joints=num_joints, hidden_dim=64, dropout=0.4).to(device)
        criterion = FocalLoss(alpha=weights, gamma=2.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        best_acc, best_state = 0, None
        patience, patience_counter = 20, 0

        for epoch in range(100):
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

        print(f"  Acc: {acc:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
        print(f"  TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_preds = (all_probs > 0.5).astype(int)

    print("\n" + "=" * 60)
    print("FINAL RESULTS (Merged Balanced)")
    print("=" * 60)
    print(f"Accuracy: {np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}")
    print(f"Overall Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Overall F1: {f1_score(all_labels, all_preds):.4f}")
    print(f"Overall AUC: {roc_auc_score(all_labels, all_probs):.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

    # Confidence analysis
    print("\nConfidence Analysis:")
    for thresh in [0.3, 0.35, 0.4]:
        mask = all_probs < thresh
        if mask.sum() > 0:
            acc = (all_labels[mask] == 0).mean()
            print(f"  prob < {thresh}: {acc:.1%} miss acc ({mask.sum()} samples)")

    for thresh in [0.6, 0.65, 0.7]:
        mask = all_probs > thresh
        if mask.sum() > 0:
            acc = (all_labels[mask] == 1).mean()
            print(f"  prob > {thresh}: {acc:.1%} make acc ({mask.sum()} samples)")

    # Save
    if best_model_state:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        torch.save({
            'model_state_dict': best_model_state,
            'model_config': {'num_joints': num_joints, 'hidden_dim': 64},
            'key_joints': KEY_JOINTS,
            'accuracy': float(np.mean(fold_accs)),
            'auc': float(roc_auc_score(all_labels, all_probs)),
            'num_samples': len(all_labels),
            'timestamp': datetime.now().isoformat(),
        }, args.output)
        print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()

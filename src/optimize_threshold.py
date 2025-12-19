"""
Optimize decision threshold and calibrate probabilities.

Quick wins to improve model performance without retraining:
1. Find optimal decision threshold
2. Apply Platt scaling for probability calibration
3. Analyze confidence-based predictions
"""
import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset

from training_utils import KeyJointNet


# Key joints (must match training)
KEY_JOINTS = [0, 5, 6, 7, 8, 41, 62, 9, 10, 69, 21, 25, 29, 64, 66]


class MergedDataset(Dataset):
    def __init__(self, data, key_joints=KEY_JOINTS):
        self.data = [d for d in data if d.get('label', -1) != -1]
        self.joint_indices = key_joints
        self.n_joints = len(key_joints)

        self.features = []
        self.labels = []

        for seq in self.data:
            try:
                kp = np.array(seq['keypoints_3d'])[:, self.joint_indices, :]
                vel = np.array(seq['velocity'])[:, self.joint_indices, :]
                accel = np.array(seq['acceleration'])[:, self.joint_indices, :]

                hip_idx_9 = self.joint_indices.index(9) if 9 in self.joint_indices else 0
                hip_idx_10 = self.joint_indices.index(10) if 10 in self.joint_indices else hip_idx_9
                center = (kp[:, hip_idx_9:hip_idx_9+1, :] + kp[:, hip_idx_10:hip_idx_10+1, :]) / 2
                kp = kp - center
                scale = np.abs(kp).max()
                if scale > 1e-6:
                    kp, vel, accel = kp/scale, vel/scale, accel/scale

                feat = np.concatenate([
                    kp.transpose(2, 0, 1),
                    vel.transpose(2, 0, 1),
                    accel.transpose(2, 0, 1)
                ], axis=0)

                self.features.append(feat)
                self.labels.append(seq['label'])
            except Exception as e:
                continue

        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]), torch.tensor(self.labels[idx], dtype=torch.long)


def get_predictions(model, dataloader, device='cpu'):
    """Get all predictions and logits from model."""
    model.eval()
    all_logits = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)[:, 1]

            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.numpy())

    return (np.concatenate(all_logits),
            np.concatenate(all_probs),
            np.concatenate(all_labels))


def optimize_threshold(probs, labels):
    """Find optimal threshold for accuracy."""
    thresholds = np.arange(0.25, 0.75, 0.01)
    best_thresh = 0.5
    best_acc = 0

    results = []
    for thresh in thresholds:
        preds = (probs > thresh).astype(int)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, zero_division=0)
        results.append((thresh, acc, f1))
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

    return best_thresh, best_acc, results


def platt_scaling(logits, labels):
    """Fit Platt scaling (logistic regression on logits)."""
    # Use logit difference as feature
    logit_diff = logits[:, 1] - logits[:, 0]

    # Fit logistic regression
    calibrator = LogisticRegression(solver='lbfgs')
    calibrator.fit(logit_diff.reshape(-1, 1), labels)

    return calibrator


def confidence_analysis(probs, labels, thresholds=[0.3, 0.35, 0.4, 0.6, 0.65, 0.7]):
    """Analyze accuracy at different confidence levels."""
    print("\n" + "=" * 50)
    print("CONFIDENCE-BASED ANALYSIS")
    print("=" * 50)

    # Low confidence = predict miss
    print("\nLow confidence (predict MISS):")
    for thresh in [0.3, 0.35, 0.4]:
        mask = probs < thresh
        if mask.sum() > 0:
            acc = (labels[mask] == 0).mean()
            print(f"  prob < {thresh}: {acc:.1%} accuracy ({mask.sum()} samples)")

    # High confidence = predict make
    print("\nHigh confidence (predict MAKE):")
    for thresh in [0.6, 0.65, 0.7]:
        mask = probs > thresh
        if mask.sum() > 0:
            acc = (labels[mask] == 1).mean()
            print(f"  prob > {thresh}: {acc:.1%} accuracy ({mask.sum()} samples)")

    # Uncertain zone
    uncertain = (probs >= 0.4) & (probs <= 0.6)
    if uncertain.sum() > 0:
        print(f"\nUncertain zone (0.4-0.6): {uncertain.sum()} samples")
        print(f"  Accuracy in uncertain zone: {accuracy_score(labels[uncertain], (probs[uncertain] > 0.5).astype(int)):.1%}")

    # Confident predictions only
    confident = (probs < 0.4) | (probs > 0.6)
    if confident.sum() > 0:
        conf_preds = (probs[confident] > 0.5).astype(int)
        conf_acc = accuracy_score(labels[confident], conf_preds)
        print(f"\nConfident predictions only ({confident.sum()}/{len(labels)} samples):")
        print(f"  Accuracy: {conf_acc:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/best_merged_model.pth')
    parser.add_argument('--data', type=str, default='data/features/optimal_merged.json')
    parser.add_argument('--output', type=str, default='models/best_merged_calibrated.pth')
    args = parser.parse_args()

    print("=" * 60)
    print("THRESHOLD OPTIMIZATION & CALIBRATION")
    print("=" * 60)

    # Load data
    with open(args.data) as f:
        data = json.load(f)

    dataset = MergedDataset(data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    print(f"Samples: {len(dataset)}")
    print(f"Makes: {(dataset.labels == 1).sum()}, Misses: {(dataset.labels == 0).sum()}")

    # Load model
    checkpoint = torch.load(args.model, weights_only=False)
    config = checkpoint['model_config']

    model = KeyJointNet(
        num_joints=config['num_joints'],
        hidden_dim=config['hidden_dim'],
        dropout=config.get('dropout', 0.4)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"\nLoaded model from {args.model}")
    print(f"Original accuracy: {checkpoint.get('accuracy', 'N/A')}")
    print(f"Original AUC: {checkpoint.get('auc', 'N/A')}")

    # Get predictions
    logits, probs, labels = get_predictions(model, dataloader)

    # Current performance with 0.5 threshold
    preds_05 = (probs > 0.5).astype(int)
    acc_05 = accuracy_score(labels, preds_05)
    print(f"\nCurrent (threshold=0.5):")
    print(f"  Accuracy: {acc_05:.4f}")
    print(f"  AUC: {roc_auc_score(labels, probs):.4f}")

    # Optimize threshold
    print("\n" + "=" * 50)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 50)

    best_thresh, best_acc, results = optimize_threshold(probs, labels)
    print(f"\nOptimal threshold: {best_thresh:.2f}")
    print(f"Accuracy at optimal: {best_acc:.4f}")
    print(f"Improvement: +{(best_acc - acc_05) * 100:.1f}%")

    # Show threshold sweep
    print("\nThreshold sweep (selected):")
    for thresh, acc, f1 in results[::5]:  # Every 5th result
        marker = " <-- optimal" if abs(thresh - best_thresh) < 0.01 else ""
        print(f"  {thresh:.2f}: acc={acc:.3f}, f1={f1:.3f}{marker}")

    # Platt scaling
    print("\n" + "=" * 50)
    print("PLATT SCALING (CALIBRATION)")
    print("=" * 50)

    calibrator = platt_scaling(logits, labels)
    logit_diff = logits[:, 1] - logits[:, 0]
    calibrated_probs = calibrator.predict_proba(logit_diff.reshape(-1, 1))[:, 1]

    # Compare calibration
    print("\nCalibration comparison:")
    print("  Bin  | Original | Calibrated | Actual")
    print("  " + "-" * 40)

    for bin_edge in [0.3, 0.4, 0.5, 0.6, 0.7]:
        mask_orig = (probs >= bin_edge - 0.05) & (probs < bin_edge + 0.05)
        mask_cal = (calibrated_probs >= bin_edge - 0.05) & (calibrated_probs < bin_edge + 0.05)

        if mask_orig.sum() > 0:
            actual_orig = labels[mask_orig].mean()
            print(f"  {bin_edge:.1f}  | {probs[mask_orig].mean():.2f}     | {calibrated_probs[mask_orig].mean():.2f}       | {actual_orig:.2f}")

    # Confidence analysis
    confidence_analysis(probs, labels)

    # Optimal predictions
    preds_opt = (probs > best_thresh).astype(int)
    cm = confusion_matrix(labels, preds_opt)

    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"Threshold: {best_thresh:.2f}")
    print(f"Accuracy: {best_acc:.4f}")
    print(f"F1: {f1_score(labels, preds_opt):.4f}")
    print(f"AUC: {roc_auc_score(labels, probs):.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

    # Save calibrated model
    calibrated_checkpoint = copy.deepcopy(checkpoint)
    calibrated_checkpoint['optimal_threshold'] = float(best_thresh)
    calibrated_checkpoint['calibrator_coef'] = calibrator.coef_.tolist()
    calibrated_checkpoint['calibrator_intercept'] = calibrator.intercept_.tolist()
    calibrated_checkpoint['accuracy_at_optimal'] = float(best_acc)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save(calibrated_checkpoint, args.output)
    print(f"\nSaved calibrated model to {args.output}")


if __name__ == '__main__':
    main()

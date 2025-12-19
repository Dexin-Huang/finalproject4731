"""
Dataset loader for basketball free throw pose sequences.
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class PoseSequenceDataset(Dataset):
    """
    Dataset for free throw pose sequences.

    Each sample contains 4 frames of 3D pose keypoints around the release point.
    Format for ST-GCN: (C, T, V) = (3, 4, 70) or (9, 4, 70) with velocity/accel
    """

    def __init__(self, sequences, normalize=True, use_enhanced=False):
        """
        Args:
            sequences: List of sequence dicts with 'frames' and 'label'
            normalize: Whether to normalize keypoints
            use_enhanced: Use enhanced features (keypoints + velocity + acceleration)
        """
        self.sequences = sequences
        self.normalize = normalize
        self.use_enhanced = use_enhanced

        # Extract pose tensors
        self.poses = []
        self.velocities = []
        self.accelerations = []
        self.labels = []
        self.video_ids = []

        for seq in sequences:
            # Check if enhanced format
            if use_enhanced and 'keypoints_3d' in seq:
                # Enhanced format: pre-computed arrays
                pose = np.array(seq['keypoints_3d'])  # (4, 70, 3)
                velocity = np.array(seq['velocity'])  # (4, 70, 3)
                acceleration = np.array(seq['acceleration'])  # (4, 70, 3)
                self.velocities.append(velocity)
                self.accelerations.append(acceleration)
            else:
                # Original format: extract from frames
                frames_kp = []
                for f in seq['frames']:
                    kp = np.array(f['keypoints_3d'])
                    frames_kp.append(kp)
                pose = np.stack(frames_kp, axis=0)  # (T, V, C) = (4, 70, 3)

            self.poses.append(pose)
            self.labels.append(seq['label'])
            self.video_ids.append(seq['video_id'])

        self.poses = np.array(self.poses)  # (N, T, V, C)
        self.labels = np.array(self.labels)

        if use_enhanced:
            self.velocities = np.array(self.velocities)
            self.accelerations = np.array(self.accelerations)

        if self.normalize:
            self._normalize()

    def _normalize(self):
        """Normalize poses - center on hip and scale."""
        # Center on pelvis (joint 0 typically)
        pelvis = self.poses[:, :, 0:1, :]  # (N, T, 1, 3)
        self.poses = self.poses - pelvis

        # Scale by max distance from pelvis
        dists = np.linalg.norm(self.poses, axis=-1)  # (N, T, V)
        max_dist = dists.max(axis=(1, 2), keepdims=True)  # (N, 1, 1)
        max_dist = np.maximum(max_dist, 1e-6)  # Avoid division by zero
        self.poses = self.poses / max_dist[:, :, :, None]

        # Scale velocity and acceleration by same factor
        if self.use_enhanced:
            self.velocities = self.velocities / max_dist[:, :, :, None]
            self.accelerations = self.accelerations / max_dist[:, :, :, None]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return (C, T, V) format for ST-GCN
        pose = self.poses[idx]  # (T, V, C)
        pose = pose.transpose(2, 0, 1)  # (C, T, V)

        if self.use_enhanced:
            vel = self.velocities[idx].transpose(2, 0, 1)  # (C, T, V)
            accel = self.accelerations[idx].transpose(2, 0, 1)  # (C, T, V)
            # Concatenate: (9, T, V)
            features = np.concatenate([pose, vel, accel], axis=0)
            return (
                torch.FloatTensor(features),
                torch.LongTensor([self.labels[idx]])[0]
            )

        return (
            torch.FloatTensor(pose),
            torch.LongTensor([self.labels[idx]])[0]
        )


def load_pose_data(data_path, test_size=0.2, random_state=42, use_enhanced=False):
    """
    Load and split data into train/test sets.

    Args:
        data_path: Path to JSON data file
        test_size: Fraction for test set
        random_state: Random seed
        use_enhanced: Use enhanced features (velocity + acceleration)

    Returns:
        train_dataset, test_dataset, class_weights
    """
    with open(data_path) as f:
        data = json.load(f)

    # Split with stratification
    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=[d['label'] for d in data]
    )

    train_dataset = PoseSequenceDataset(train_data, use_enhanced=use_enhanced)
    test_dataset = PoseSequenceDataset(test_data, use_enhanced=use_enhanced)

    # Compute class weights for imbalanced data
    train_labels = [d['label'] for d in train_data]
    n_made = sum(train_labels)
    n_miss = len(train_labels) - n_made
    weight_made = len(train_labels) / (2 * n_made)
    weight_miss = len(train_labels) / (2 * n_miss)
    class_weights = torch.FloatTensor([weight_miss, weight_made])

    print(f"Train: {len(train_dataset)} samples ({sum(train_labels)} made, {len(train_labels)-sum(train_labels)} miss)")
    print(f"Test: {len(test_dataset)} samples")
    print(f"Class weights: miss={weight_miss:.2f}, made={weight_made:.2f}")

    return train_dataset, test_dataset, class_weights


if __name__ == "__main__":
    # Test loading - basic
    print("Testing basic features:")
    train_ds, test_ds, weights = load_pose_data("data/features/features_clean.json")
    x, y = train_ds[0]
    print(f"  Sample shape: {x.shape}")  # Should be (3, 4, 70)

    # Test loading - enhanced
    print("\nTesting enhanced features:")
    train_ds, test_ds, weights = load_pose_data(
        "data/features/enhanced_all.json",
        use_enhanced=True
    )
    x, y = train_ds[0]
    print(f"  Sample shape: {x.shape}")  # Should be (9, 4, 70)

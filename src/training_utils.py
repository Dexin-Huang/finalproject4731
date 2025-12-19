"""
Shared utilities for training scripts.

Contains common classes and functions used across multiple training scripts:
- KeyJointNet: Neural network model for shot prediction
- FocalLoss: Focal loss for handling class imbalance
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class KeyJointNet(nn.Module):
    """
    Neural network for shot prediction based on key joint movements.

    Uses joint attention and temporal convolutions to predict shot outcomes.

    Args:
        num_joints: Number of joints in the input
        in_channels: Number of input channels (default: 9 for 3D coords + velocity + acceleration)
        hidden_dim: Hidden dimension size (default: 64)
        dropout: Dropout rate (default: 0.4)
    """
    def __init__(self, num_joints, in_channels=9, hidden_dim=64, dropout=0.4):
        super().__init__()

        # Joint attention mechanism
        self.joint_attn = nn.Sequential(
            nn.Linear(num_joints, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_joints),
            nn.Softmax(dim=-1)
        )

        # Temporal convolution layers
        self.temporal = nn.Sequential(
            nn.Conv1d(in_channels * num_joints, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
        )

        # Pooling and classification
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 2)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (N, C, T, V) where:
               N = batch size
               C = number of channels (9)
               T = temporal frames (4)
               V = number of joints

        Returns:
            Logits of shape (N, 2) for binary classification
        """
        N, C, T, V = x.shape

        # Compute joint attention based on variance
        joint_var = x.var(dim=(1, 2))
        attn = self.joint_attn(joint_var)

        # Apply attention
        x = x * attn.unsqueeze(1).unsqueeze(2)

        # Reshape for temporal convolution
        x = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)

        # Temporal processing
        x = self.temporal(x)

        # Pool and classify
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Focal Loss applies a modulating term to the cross entropy loss to focus
    learning on hard examples and down-weight easy examples.

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)

    Args:
        alpha: Weighting factor for each class (optional)
        gamma: Focusing parameter (default: 2.0)
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Compute focal loss.

        Args:
            inputs: Predicted logits of shape (N, num_classes)
            targets: Ground truth labels of shape (N,)

        Returns:
            Scalar focal loss value
        """
        ce = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        focal_loss = ((1 - pt) ** self.gamma) * ce
        return focal_loss.mean()

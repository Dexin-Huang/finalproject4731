"""
Release Frame Detector Model

Predicts the release frame from a sequence of per-frame features.
Uses 1D CNN to model temporal patterns in ball-hand distance, velocities, etc.

Input: (batch, seq_len, num_features) - features for a window of frames
Output: (batch, seq_len) - per-frame probability of being the release frame
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class TemporalConvBlock(nn.Module):
    """1D Convolution block with batch norm and residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        residual = self.residual(x)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x + residual
        return x


class ReleaseFrameDetector(nn.Module):
    """
    1D CNN for release frame detection.

    Takes a sequence of frame features and outputs per-frame release probability.
    """

    def __init__(
        self,
        num_features: int = 18,
        hidden_channels: int = 64,
        num_layers: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.num_features = num_features

        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_channels)

        # Temporal conv layers
        layers = []
        for i in range(num_layers):
            in_ch = hidden_channels if i == 0 else hidden_channels * 2
            out_ch = hidden_channels * 2
            layers.append(TemporalConvBlock(
                in_ch, out_ch, kernel_size=kernel_size, dropout=dropout
            ))
        self.conv_layers = nn.Sequential(*layers)

        # Output projection (per-frame classification)
        self.output_proj = nn.Conv1d(hidden_channels * 2, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F) - batch of feature sequences

        Returns:
            logits: (B, T) - per-frame release probability logits
        """
        # Project input features
        x = self.input_proj(x)  # (B, T, H)

        # Transpose for conv: (B, T, H) -> (B, H, T)
        x = x.permute(0, 2, 1)

        # Apply temporal convolutions
        x = self.conv_layers(x)  # (B, H*2, T)

        # Per-frame output
        x = self.output_proj(x)  # (B, 1, T)
        x = x.squeeze(1)  # (B, T)

        return x

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict release frame index and confidence.

        Returns:
            frame_idx: (B,) - predicted frame indices
            confidence: (B,) - confidence scores (max softmax probability)
        """
        logits = self.forward(x)  # (B, T)
        probs = F.softmax(logits, dim=-1)

        confidence, frame_idx = probs.max(dim=-1)
        return frame_idx, confidence


class ReleaseFrameDetectorLSTM(nn.Module):
    """
    Bi-LSTM variant for release frame detection.

    May capture longer-range temporal dependencies better than CNN.
    """

    def __init__(
        self,
        num_features: int = 18,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.num_features = num_features

        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_size)

        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F) - batch of feature sequences

        Returns:
            logits: (B, T) - per-frame release probability logits
        """
        # Project input
        x = self.input_proj(x)  # (B, T, H)

        # Apply Bi-LSTM
        x, _ = self.lstm(x)  # (B, T, H*2)

        # Per-frame output
        logits = self.output_proj(x).squeeze(-1)  # (B, T)

        return logits

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict release frame index and confidence."""
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        confidence, frame_idx = probs.max(dim=-1)
        return frame_idx, confidence


class ReleaseFrameDetectorTransformer(nn.Module):
    """
    Transformer-based release frame detector.

    Uses self-attention to model temporal relationships.
    """

    def __init__(
        self,
        num_features: int = 18,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_seq_len: int = 60,
    ):
        super().__init__()

        self.num_features = num_features

        # Input projection
        self.input_proj = nn.Linear(num_features, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F) - batch of feature sequences

        Returns:
            logits: (B, T) - per-frame release probability logits
        """
        B, T, F = x.shape

        # Project and add positional encoding
        x = self.input_proj(x)  # (B, T, D)
        x = x + self.pos_encoding[:, :T, :]

        # Apply transformer
        x = self.transformer(x)  # (B, T, D)

        # Per-frame output
        logits = self.output_proj(x).squeeze(-1)  # (B, T)

        return logits

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict release frame index and confidence."""
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        confidence, frame_idx = probs.max(dim=-1)
        return frame_idx, confidence


def create_gaussian_labels(
    release_idx: int,
    seq_len: int,
    sigma: float = 1.5,
) -> np.ndarray:
    """
    Create soft labels with Gaussian distribution around release frame.

    Instead of one-hot labels, use Gaussian to allow for small timing errors.
    """
    x = np.arange(seq_len)
    gaussian = np.exp(-0.5 * ((x - release_idx) / sigma) ** 2)
    gaussian = gaussian / gaussian.sum()  # Normalize to sum to 1
    return gaussian


def gaussian_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Focal loss with soft Gaussian targets.

    Reduces loss for well-classified frames, focuses on hard cases.
    """
    probs = F.softmax(logits, dim=-1)

    # Cross-entropy with soft targets
    log_probs = F.log_softmax(logits, dim=-1)
    ce_loss = -(targets * log_probs).sum(dim=-1)

    # Focal weight
    pt = (probs * targets).sum(dim=-1)  # Probability of correct class
    focal_weight = (1 - pt) ** gamma

    loss = (focal_weight * ce_loss).mean()
    return loss


# Feature names for reference
FEATURE_NAMES = [
    'ball_x', 'ball_y', 'ball_area',
    'ball_velocity_x', 'ball_velocity_y', 'ball_speed', 'ball_acceleration',
    'dist_left_wrist', 'dist_right_wrist', 'min_hand_dist',
    'left_arm_angle', 'right_arm_angle',
    'left_arm_extension', 'right_arm_extension',
    'ball_shooter_overlap',
    'left_wrist_velocity', 'right_wrist_velocity',
    'ball_detected',  # Binary flag
]


def get_model(
    model_type: str = 'cnn',
    num_features: int = 18,
    **kwargs,
) -> nn.Module:
    """Factory function to create release detector model."""
    if model_type == 'cnn':
        return ReleaseFrameDetector(num_features=num_features, **kwargs)
    elif model_type == 'lstm':
        return ReleaseFrameDetectorLSTM(num_features=num_features, **kwargs)
    elif model_type == 'transformer':
        return ReleaseFrameDetectorTransformer(num_features=num_features, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    # Test models
    batch_size = 4
    seq_len = 30
    num_features = 18

    x = torch.randn(batch_size, seq_len, num_features)

    print("Testing ReleaseFrameDetector (CNN):")
    model_cnn = ReleaseFrameDetector(num_features=num_features)
    logits = model_cnn(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits.shape}")
    frame_idx, conf = model_cnn.predict(x)
    print(f"  Predicted frames: {frame_idx}")
    print(f"  Confidence: {conf}")
    print(f"  Parameters: {sum(p.numel() for p in model_cnn.parameters()):,}")

    print("\nTesting ReleaseFrameDetectorLSTM:")
    model_lstm = ReleaseFrameDetectorLSTM(num_features=num_features)
    logits = model_lstm(x)
    print(f"  Output shape: {logits.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_lstm.parameters()):,}")

    print("\nTesting ReleaseFrameDetectorTransformer:")
    model_tf = ReleaseFrameDetectorTransformer(num_features=num_features)
    logits = model_tf(x)
    print(f"  Output shape: {logits.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_tf.parameters()):,}")

    # Test Gaussian labels
    print("\nTesting Gaussian labels:")
    labels = create_gaussian_labels(release_idx=15, seq_len=30, sigma=1.5)
    print(f"  Labels shape: {labels.shape}")
    print(f"  Peak at: {labels.argmax()}")
    print(f"  Peak value: {labels.max():.3f}")

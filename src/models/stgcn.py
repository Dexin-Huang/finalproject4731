"""
Spatial Temporal Graph Convolutional Network (ST-GCN) for pose-based classification.

Adapted for basketball free throw prediction with 70 joints from SAM3D Body.
Lightweight version suitable for small datasets.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Graph:
    """
    Graph structure for skeleton-based pose data.

    For SAM3D Body's 70 joints, we use a learnable adjacency matrix
    since the exact skeleton hierarchy isn't standard.
    """

    def __init__(self, num_joints=70, strategy='uniform'):
        self.num_joints = num_joints
        self.strategy = strategy
        self.A = self._get_adjacency_matrix()

    def _get_adjacency_matrix(self):
        """
        Get adjacency matrix based on strategy.

        For unknown skeletons, we use:
        - 'uniform': All joints connected equally (learnable attention)
        - 'distance': Connections based on joint indices (nearby = connected)
        """
        if self.strategy == 'uniform':
            # Self-connections only, let model learn the rest
            A = np.eye(self.num_joints, dtype=np.float32)
        elif self.strategy == 'distance':
            # Connect joints that are close in index (heuristic for body parts)
            A = np.eye(self.num_joints, dtype=np.float32)
            for i in range(self.num_joints - 1):
                A[i, i+1] = 1
                A[i+1, i] = 1
        else:
            A = np.eye(self.num_joints, dtype=np.float32)

        return A


class SpatialGraphConv(nn.Module):
    """
    Spatial graph convolution layer.
    """

    def __init__(self, in_channels, out_channels, A, adaptive=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Base adjacency matrix
        self.register_buffer('A', torch.FloatTensor(A))

        # Adaptive adjacency (learnable)
        self.adaptive = adaptive
        if adaptive:
            self.PA = nn.Parameter(torch.FloatTensor(A.shape[0], A.shape[1]))
            nn.init.uniform_(self.PA, -1e-6, 1e-6)

        # Graph convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        Args:
            x: (N, C, T, V) - batch, channels, time, vertices/joints
        Returns:
            (N, C_out, T, V)
        """
        # Get effective adjacency
        A = self.A
        if self.adaptive:
            A = A + self.PA

        # Normalize adjacency
        D = torch.sum(A, dim=1, keepdim=True)
        D = torch.clamp(D, min=1e-6)
        A_norm = A / D

        # Graph convolution: aggregate neighbor features
        # x: (N, C, T, V), A: (V, V)
        # Output: (N, C, T, V)
        x = torch.einsum('nctv,vw->nctw', x, A_norm)

        # Point-wise convolution
        x = self.conv(x)
        x = self.bn(x)

        return x


class TemporalConv(nn.Module):
    """
    Temporal convolution layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.0):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: (N, C, T, V)
        Returns:
            (N, C_out, T, V)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x


class STGCNBlock(nn.Module):
    """
    Spatial-Temporal Graph Convolution Block.

    Combines spatial graph convolution with temporal convolution.
    """

    def __init__(self, in_channels, out_channels, A, stride=1, dropout=0.1):
        super().__init__()

        self.spatial = SpatialGraphConv(in_channels, out_channels, A, adaptive=True)
        self.temporal = TemporalConv(out_channels, out_channels, kernel_size=3, stride=stride, dropout=dropout)
        self.relu = nn.ReLU(inplace=True)

        # Residual connection
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x = self.spatial(x)
        x = self.relu(x)
        x = self.temporal(x)
        x = x + res
        x = self.relu(x)
        return x


class STGCN(nn.Module):
    """
    Spatial Temporal Graph Convolutional Network.

    Lightweight version for small datasets.
    """

    def __init__(
        self,
        num_classes=2,
        num_joints=70,
        in_channels=3,
        hidden_channels=64,
        num_layers=3,
        dropout=0.3,
        graph_strategy='uniform'
    ):
        super().__init__()

        self.num_joints = num_joints
        self.in_channels = in_channels

        # Build graph
        graph = Graph(num_joints, strategy=graph_strategy)
        A = graph.A

        # Input batch norm
        self.bn_in = nn.BatchNorm1d(in_channels * num_joints)

        # ST-GCN layers
        self.layers = nn.ModuleList()

        # First layer: in_channels -> hidden_channels
        self.layers.append(STGCNBlock(in_channels, hidden_channels, A, dropout=dropout))

        # Middle layers
        for _ in range(num_layers - 2):
            self.layers.append(STGCNBlock(hidden_channels, hidden_channels, A, dropout=dropout))

        # Last layer (optional temporal downsampling)
        if num_layers > 1:
            self.layers.append(STGCNBlock(hidden_channels, hidden_channels * 2, A, stride=1, dropout=dropout))
            final_channels = hidden_channels * 2
        else:
            final_channels = hidden_channels

        # Global pooling + classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(final_channels, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (N, C, T, V) = (batch, 3, 4, 70)
        Returns:
            (N, num_classes) logits
        """
        N, C, T, V = x.shape

        # Input normalization
        x = x.permute(0, 2, 3, 1).contiguous()  # (N, T, V, C)
        x = x.view(N * T, V * C)
        x = self.bn_in(x)
        x = x.view(N, T, V, C).permute(0, 3, 1, 2).contiguous()  # (N, C, T, V)

        # ST-GCN layers
        for layer in self.layers:
            x = layer(x)

        # Global pooling
        x = self.pool(x)  # (N, C, 1, 1)
        x = x.view(N, -1)

        # Classification
        x = self.dropout(x)
        x = self.fc(x)

        return x


class SimplePoseNet(nn.Module):
    """
    Simple baseline: flatten poses and use MLP.

    Good for comparison with ST-GCN on small datasets.
    """

    def __init__(self, num_classes=2, num_joints=70, num_frames=4, in_channels=3, hidden_dim=256, dropout=0.5):
        super().__init__()

        input_dim = num_joints * num_frames * in_channels

        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (N, C, T, V) = (batch, 3, 4, 70)
        """
        x = self.flatten(x)
        return self.net(x)


class TemporalPoseNet(nn.Module):
    """
    Temporal model using 1D convolutions over time.

    Treats each joint independently, then pools.
    """

    def __init__(self, num_classes=2, num_joints=70, in_channels=3, hidden_channels=64, dropout=0.3):
        super().__init__()

        # Per-joint temporal convolutions
        self.conv1 = nn.Conv1d(in_channels * num_joints, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_channels * 2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_channels * 2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: (N, C, T, V) = (batch, 3, 4, 70)
        """
        N, C, T, V = x.shape

        # Reshape: (N, C*V, T)
        x = x.permute(0, 1, 3, 2).contiguous()  # (N, C, V, T)
        x = x.view(N, C * V, T)

        # Temporal convolutions
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        # Pool and classify
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    # Test models
    batch_size = 4
    x = torch.randn(batch_size, 3, 4, 70)  # (N, C, T, V)

    print("Testing STGCN...")
    model = STGCN(num_classes=2, num_joints=70)
    out = model(x)
    print(f"  Input: {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTesting SimplePoseNet...")
    model = SimplePoseNet(num_classes=2)
    out = model(x)
    print(f"  Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTesting TemporalPoseNet...")
    model = TemporalPoseNet(num_classes=2)
    out = model(x)
    print(f"  Output: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

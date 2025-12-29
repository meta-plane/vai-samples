#!/usr/bin/env python3
"""
Generate PyTorch reference data for TNetBlock test
Uses yanx27 PointNet STN3d structure with PyTorch state_dict keys.

TNetBlock architecture:
- Input: [K, N] point cloud (PyTorch [C, N] format)
- TNet Output: [K, K] transformation matrix
- MatMul Output: [K, N] transformed points

PyTorch keys preserved (STN3d style):
  conv1, bn1, conv2, bn2, conv3, bn3 (MLP layers)
  fc1, bn4, fc2, bn5, fc3 (FC layers)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from safetensors.torch import save_file

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class STN3d(nn.Module):
    """
    STN3d from yanx27 PointNet (Spatial Transformer Network for 3D)
    Generates K×K transformation matrix
    """
    def __init__(self, K=3):
        super().__init__()
        self.K = K

        # MLP: K -> 64 -> 128 -> 1024 (Conv1d with BatchNorm)
        self.conv1 = nn.Conv1d(K, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        # FC: 1024 -> 512 -> 256 -> K*K
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, K*K)  # Last FC: no BN, no ReLU

    def forward(self, x):
        """
        x: [K, N] input points (PyTorch [C, N] format)
        Returns: [K, K] transformation matrix
        """
        K, N = x.shape

        # Add batch dimension for Conv1d: [K, N] -> [1, K, N]
        feat = x.unsqueeze(0)

        # MLP layers with BatchNorm and ReLU
        feat = torch.relu(self.bn1(self.conv1(feat)))  # [1, 64, N]
        feat = torch.relu(self.bn2(self.conv2(feat)))  # [1, 128, N]
        feat = torch.relu(self.bn3(self.conv3(feat)))  # [1, 1024, N]

        # MaxPool: [1, 1024, N] -> [1, 1024]
        feat = torch.max(feat, dim=2)[0]  # [1, 1024]

        # FC layers with BatchNorm+ReLU
        feat = torch.relu(self.bn4(self.fc1(feat)))  # [1, 512]
        feat = torch.relu(self.bn5(self.fc2(feat)))  # [1, 256]
        feat = self.fc3(feat)  # [1, K*K], no BN, no ReLU

        # Remove batch and reshape to transformation matrix
        transform = feat.view(K, K)  # [K, K]

        # Add identity matrix
        identity = torch.eye(K, device=x.device)
        transform = transform + identity

        return transform


def main():
    # Test configuration
    N = 8   # Number of points
    K = 3   # Dimension (for 3D points)

    print(f"Generating TNetBlock reference data (PyTorch keys)...")
    print(f"  Input: [{K}, {N}] (PyTorch [C, N] format)")
    print(f"  Transform: [{K}, {K}]")
    print(f"  Output: [{K}, {N}]")

    # Create model
    model = STN3d(K)
    model.eval()

    # Generate random input [K, N] - PyTorch [C, N] format
    x = torch.randn(K, N)

    # Forward pass
    with torch.no_grad():
        transform = model(x)  # [K, K]

        # Apply transformation: x.T @ transform -> output.T
        # PyTorch: [N, K] @ [K, K] = [N, K] -> transpose to [K, N]
        output = (x.T @ transform).T  # [K, N]

    # Prepare tensors with PyTorch state_dict keys
    tensors = {
        'input': x.contiguous(),
        'output': output.contiguous(),
        'transform': transform.contiguous(),
        'shape': torch.tensor([K, N], dtype=torch.float32),
    }

    # Add model weights with PyTorch state_dict keys
    print("\nExtracting weights (PyTorch keys)...")
    state_dict = model.state_dict()

    for key, value in state_dict.items():
        # Skip num_batches_tracked
        if 'num_batches_tracked' in key:
            continue

        # Conv1d weights: squeeze [C_out, C_in, 1] -> [C_out, C_in]
        if 'conv' in key and 'weight' in key:
            value = value.squeeze(-1)
            print(f"  {key}: {list(state_dict[key].shape)} -> {list(value.shape)}")
        else:
            print(f"  {key}: {list(value.shape)}")

        tensors[key] = value.contiguous()

    # Save SafeTensors
    output_dir = Path(__file__).parent
    safetensors_path = output_dir / 'reference.safetensors'
    save_file(tensors, str(safetensors_path))

    print(f"\n✅ Saved to: {safetensors_path}")
    print(f"  Total tensors: {len(tensors)}")

    # Show first few output values for debugging
    print(f"\nFirst 5 output values:")
    for i in range(min(5, K)):
        print(f"  output[{i}, 0] = {output[i, 0]:.6f}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Generate reference outputs from PyTorch PointNet implementation.

This script runs PyTorch models with fixed inputs and saves outputs
for comparison with the Vulkan implementation.

Based on: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path


class PointWiseMLP(nn.Module):
    """Single point-wise MLP layer for testing."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        # x: [B, C, N]
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class TNet(nn.Module):
    """Spatial Transformer Network."""
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        
        # MLP: k -> 64 -> 128 -> 1024
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        # FC: 1024 -> 512 -> 256 -> k*k
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # MLP layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Max pooling
        x = torch.max(x, 2, keepdim=False)[0]
        
        # FC layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        # Reshape to transformation matrix
        iden = torch.eye(self.k, device=x.device).flatten().unsqueeze(0)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        
        return x


class PointNetEncoder(nn.Module):
    """PointNet Encoder."""
    def __init__(self):
        super().__init__()
        
        # Input transform
        self.tnet1 = TNet(k=3)
        
        # MLP1: 3 -> 64 -> 64
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        
        # Feature transform
        self.tnet2 = TNet(k=64)
        
        # MLP2: 64 -> 128 -> 1024
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)
    
    def forward(self, x):
        # x: [B, 3, N]
        n_pts = x.size(2)
        
        # Input transform
        trans1 = self.tnet1(x)
        x = torch.bmm(trans1, x)
        
        # MLP1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Feature transform
        trans2 = self.tnet2(x)
        x = torch.bmm(trans2, x)
        
        # MLP2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        return x


class PointNetSegmentation(nn.Module):
    """PointNet Segmentation Network."""
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.encoder = PointNetEncoder()
        
        # Segmentation head: 2048 -> 512 -> 256 -> num_classes
        self.conv1 = nn.Conv1d(2048, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, num_classes, 1)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
    
    def forward(self, x):
        # x: [B, 3, N]
        batch_size, _, n_pts = x.size()
        
        # Encoder: [B, 3, N] -> [B, 1024, N]
        point_features = self.encoder(x)
        
        # Global feature: [B, 1024, N] -> [B, 1024, 1] -> [B, 1024, N]
        global_feature = torch.max(point_features, 2, keepdim=True)[0]
        global_feature = global_feature.expand(-1, -1, n_pts)
        
        # Concatenate: [B, 2048, N]
        features = torch.cat([point_features, global_feature], dim=1)
        
        # Segmentation head
        x = F.relu(self.bn1(self.conv1(features)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        
        return x


def generate_test_inputs():
    """Generate fixed test inputs for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Small point cloud for testing
    num_points = 16  # Small for debugging
    points = np.random.randn(1, 3, num_points).astype(np.float32)
    
    return torch.from_numpy(points)


def test_mlp_layer():
    """Test 1: Single MLP layer."""
    print("\nGenerating Test 1: Single MLP Layer")
    
    # Create model
    model = PointWiseMLP(3, 64)
    model.eval()
    
    # Fixed input
    x = generate_test_inputs()  # [1, 3, 16]
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    # Save weights and output
    output_dir = Path("test/references/mlp_layer")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'input': x.numpy(),
        'output': output.numpy(),
        'weight': model.conv.weight.data.numpy(),
        'bias': model.conv.bias.data.numpy(),
    }, output_dir / "reference.pth")
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Saved to: {output_dir}")


def test_tnet():
    """Test 2 & 3: TNet blocks."""
    print("\nGenerating Test 2-3: TNet blocks")
    
    for k in [3, 64]:
        print(f"  TNet k={k}")
        model = TNet(k=k)
        model.eval()
        
        # Input
        if k == 3:
            x = generate_test_inputs()  # [1, 3, 16]
        else:
            x = torch.randn(1, 64, 16)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Save
        output_dir = Path(f"test/references/tnet_k{k}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'input': x.numpy(),
            'output': output.numpy(),
            'state_dict': model.state_dict(),
        }, output_dir / "reference.pth")
        
        print(f"    Input: {x.shape}, Output: {output.shape}")


def test_encoder():
    """Test 4: PointNet Encoder."""
    print("\nGenerating Test 4: PointNet Encoder")
    
    model = PointNetEncoder()
    model.eval()
    
    x = generate_test_inputs()
    
    with torch.no_grad():
        output = model(x)
    
    output_dir = Path("test/references/encoder")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'input': x.numpy(),
        'output': output.numpy(),
        'state_dict': model.state_dict(),
    }, output_dir / "reference.pth")
    
    print(f"  Input: {x.shape}, Output: {output.shape}")


def test_full_network():
    """Test 5: Full PointNet Segmentation."""
    print("\nGenerating Test 5: Full PointNet Segmentation")
    
    model = PointNetSegmentation(num_classes=10)
    model.eval()
    
    x = generate_test_inputs()
    
    with torch.no_grad():
        output = model(x)
    
    output_dir = Path("test/references/full_network")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'input': x.numpy(),
        'output': output.numpy(),
        'state_dict': model.state_dict(),
    }, output_dir / "reference.pth")
    
    print(f"  Input: {x.shape}, Output: {output.shape}")
    
    # Also save in JSON format for easy loading in C++
    np.savetxt(output_dir / "input.txt", x.numpy().flatten(), fmt='%.6f')
    np.savetxt(output_dir / "output.txt", output.numpy().flatten(), fmt='%.6f')


def main():
    print("=" * 60)
    print("Generating PyTorch Reference Outputs")
    print("=" * 60)
    
    # Generate all test references
    test_mlp_layer()
    test_tnet()
    test_encoder()
    test_full_network()
    
    print("\n" + "=" * 60)
    print("✓ All reference outputs generated!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Build C++ tests: cd build && make pointnet-tests")
    print("2. Run tests: ./build/bin/debug/pointnet-tests")


if __name__ == '__main__':
    main()


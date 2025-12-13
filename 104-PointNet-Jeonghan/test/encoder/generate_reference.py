#!/usr/bin/env python3
"""
Generate PyTorch reference data for PointNetEncoder test
PointNetEncoder: Full encoder pipeline
- TNet1 (3x3) → MLP1 (3→64→64) → TNet2 (64x64) → MLP2 (64→128→1024) → MaxPool
- Input: [N, 3] point cloud
- Output: [1024] global feature vector
"""

import torch
import torch.nn as nn
import json
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class TNetBlock(nn.Module):
    """
    TNet: Spatial Transformer Network
    """
    def __init__(self, K):
        super().__init__()
        self.K = K
        
        # MLP: K -> 64 -> 128 -> 1024
        self.mlp1 = nn.Conv1d(K, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.mlp2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.mlp3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        
        # FC: 1024 -> 512 -> 256 -> K*K
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, K*K)
        
    def forward(self, x):
        """x: [N, K]"""
        N, K = x.shape
        
        # Generate transformation matrix
        feat = x.unsqueeze(2)  # [N, K, 1]
        feat = torch.relu(self.bn1(self.mlp1(feat)))
        feat = torch.relu(self.bn2(self.mlp2(feat)))
        feat = torch.relu(self.bn3(self.mlp3(feat)))
        feat = feat.squeeze(2)  # [N, 1024]
        feat = torch.max(feat, dim=0, keepdim=True)[0]  # [1, 1024]
        pooled = feat.squeeze(0)  # [1024]
        
        # FC layers
        feat = self.fc1(pooled)
        feat = torch.relu(self.bn4(feat.unsqueeze(0))).squeeze(0)
        feat = self.fc2(feat)
        feat = torch.relu(self.bn5(feat.unsqueeze(0))).squeeze(0)
        feat = self.fc3(feat)
        
        transform_no_id = feat.view(K, K)
        identity = torch.eye(K)
        transform = transform_no_id + identity
        
        # Return both transform matrix (no matmul) and original input
        return transform  # ONLY return transformation matrix


class MLPBlock(nn.Module):
    """MLP block with Conv1d"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        """x: [N, C, 1]"""
        return torch.relu(self.bn(self.conv(x)))


class PointNetEncoder(nn.Module):
    """
    PointNet Encoder
    Input: [N, 3] → Output: [1024] global feature
    """
    def __init__(self):
        super().__init__()
        
        # TNet1: 3x3 transformation
        self.tnet1 = TNetBlock(3)
        
        # MLP1: 3 → 64 → 64
        self.mlp1_0 = MLPBlock(3, 64)
        self.mlp1_1 = MLPBlock(64, 64)
        
        # TNet2: 64x64 transformation
        self.tnet2 = TNetBlock(64)
        
        # MLP2: 64 → 128 → 1024
        self.mlp2_0 = MLPBlock(64, 128)
        self.mlp2_1 = MLPBlock(128, 1024)
    
    def forward(self, x):
        """
        x: [N, 3] point cloud
        Returns: [1024] global feature + intermediate values
        """
        N = x.shape[0]
        
        # TNet1: [N, 3] → transformation matrix [3, 3]
        transform1 = self.tnet1(x)
        x_transformed = torch.matmul(x, transform1)  # Apply transformation
        
        # MLP1: [N, 3] → [N, 64] → [N, 64]
        feat = x_transformed.unsqueeze(2)  # [N, 3, 1]
        feat = self.mlp1_0(feat)  # [N, 64, 1]
        feat = self.mlp1_1(feat)  # [N, 64, 1]
        feat_mlp1 = feat.squeeze(2)  # [N, 64]
        
        # TNet2: [N, 64] → transformation matrix [64, 64]
        transform2 = self.tnet2(feat_mlp1)
        feat_transformed = torch.matmul(feat_mlp1, transform2)  # Apply transformation
        
        # MLP2: [N, 64] → [N, 128] → [N, 1024]
        feat = feat_transformed.unsqueeze(2)  # [N, 64, 1]
        feat = self.mlp2_0(feat)  # [N, 128, 1]
        feat = self.mlp2_1(feat)  # [N, 1024, 1]
        feat = feat.squeeze(2)  # [N, 1024]
        
        # Global max pooling: [N, 1024] → [1024]
        global_feat = torch.max(feat, dim=0)[0]  # [1024]
        
        return {
            'output': global_feat,
            'tnet1_matrix': transform1,
            'after_matmul1': x_transformed,
            'after_mlp1': feat_mlp1,
            'tnet2_matrix': transform2,
            'after_matmul2': feat_transformed,
            'after_mlp2': feat
        }


def main():
    # Test configuration
    N = 16  # Number of points (small for fast testing)
    
    # Create model and set to eval mode
    model = PointNetEncoder()
    model.eval()
    
    # Generate random input
    input_points = torch.randn(N, 3)
    
    # Forward pass
    with torch.no_grad():
        results = model(input_points)
    
    print(f"Input shape: [{N}, 3]")
    print(f"Output shape: [{results['output'].shape[0]}]")
    print(f"Output (first 5): {results['output'][:5].numpy()}")
    print(f"TNet1 matrix shape: {results['tnet1_matrix'].shape}")
    print(f"After MatMul1 shape: {results['after_matmul1'].shape}")
    print(f"After MLP1 shape: {results['after_mlp1'].shape}")
    
    # Build reference dictionary
    reference = {
        'input_shape': [float(N), 3.0],
        'input': input_points.numpy().flatten().tolist(),
        'expected_output': results['output'].numpy().flatten().tolist(),
        
        # Intermediate values for debugging
        'tnet1_matrix': results['tnet1_matrix'].numpy().flatten().tolist(),
        'after_matmul1': results['after_matmul1'].numpy().flatten().tolist(),
        'after_mlp1': results['after_mlp1'].numpy().flatten().tolist(),
        'tnet2_matrix': results['tnet2_matrix'].numpy().flatten().tolist(),
        'after_matmul2': results['after_matmul2'].numpy().flatten().tolist(),
        'after_mlp2': results['after_mlp2'].numpy().flatten().tolist(),
        
        # TNet1 weights (3x3 transformation)
        'tnet1.mlp0.weight': model.tnet1.mlp1.weight.squeeze().T.contiguous().detach().numpy().flatten().tolist(),
        'tnet1.mlp0.bias': model.tnet1.mlp1.bias.detach().numpy().flatten().tolist(),
        'tnet1.mlp0.bn_mean': model.tnet1.bn1.running_mean.detach().numpy().flatten().tolist(),
        'tnet1.mlp0.bn_var': model.tnet1.bn1.running_var.detach().numpy().flatten().tolist(),
        'tnet1.mlp0.bn_gamma': model.tnet1.bn1.weight.detach().numpy().flatten().tolist(),
        'tnet1.mlp0.bn_beta': model.tnet1.bn1.bias.detach().numpy().flatten().tolist(),
        
        'tnet1.mlp1.weight': model.tnet1.mlp2.weight.squeeze().T.contiguous().detach().numpy().flatten().tolist(),
        'tnet1.mlp1.bias': model.tnet1.mlp2.bias.detach().numpy().flatten().tolist(),
        'tnet1.mlp1.bn_mean': model.tnet1.bn2.running_mean.detach().numpy().flatten().tolist(),
        'tnet1.mlp1.bn_var': model.tnet1.bn2.running_var.detach().numpy().flatten().tolist(),
        'tnet1.mlp1.bn_gamma': model.tnet1.bn2.weight.detach().numpy().flatten().tolist(),
        'tnet1.mlp1.bn_beta': model.tnet1.bn2.bias.detach().numpy().flatten().tolist(),
        
        'tnet1.mlp2.weight': model.tnet1.mlp3.weight.squeeze().T.contiguous().detach().numpy().flatten().tolist(),
        'tnet1.mlp2.bias': model.tnet1.mlp3.bias.detach().numpy().flatten().tolist(),
        'tnet1.mlp2.bn_mean': model.tnet1.bn3.running_mean.detach().numpy().flatten().tolist(),
        'tnet1.mlp2.bn_var': model.tnet1.bn3.running_var.detach().numpy().flatten().tolist(),
        'tnet1.mlp2.bn_gamma': model.tnet1.bn3.weight.detach().numpy().flatten().tolist(),
        'tnet1.mlp2.bn_beta': model.tnet1.bn3.bias.detach().numpy().flatten().tolist(),
        
        'tnet1.fc0.weight': model.tnet1.fc1.weight.T.contiguous().detach().numpy().flatten().tolist(),
        'tnet1.fc0.bias': model.tnet1.fc1.bias.detach().numpy().flatten().tolist(),
        'tnet1.fc0.mean': model.tnet1.bn4.running_mean.detach().numpy().flatten().tolist(),
        'tnet1.fc0.var': model.tnet1.bn4.running_var.detach().numpy().flatten().tolist(),
        'tnet1.fc0.gamma': model.tnet1.bn4.weight.detach().numpy().flatten().tolist(),
        'tnet1.fc0.beta': model.tnet1.bn4.bias.detach().numpy().flatten().tolist(),
        
        'tnet1.fc1.weight': model.tnet1.fc2.weight.T.contiguous().detach().numpy().flatten().tolist(),
        'tnet1.fc1.bias': model.tnet1.fc2.bias.detach().numpy().flatten().tolist(),
        'tnet1.fc1.mean': model.tnet1.bn5.running_mean.detach().numpy().flatten().tolist(),
        'tnet1.fc1.var': model.tnet1.bn5.running_var.detach().numpy().flatten().tolist(),
        'tnet1.fc1.gamma': model.tnet1.bn5.weight.detach().numpy().flatten().tolist(),
        'tnet1.fc1.beta': model.tnet1.bn5.bias.detach().numpy().flatten().tolist(),
        
        'tnet1.fc2.weight': model.tnet1.fc3.weight.T.contiguous().detach().numpy().flatten().tolist(),
        'tnet1.fc2.bias': model.tnet1.fc3.bias.detach().numpy().flatten().tolist(),
        
        # MLP1 weights
        'mlp1.mlp0.weight': model.mlp1_0.conv.weight.squeeze().T.contiguous().detach().numpy().flatten().tolist(),
        'mlp1.mlp0.bias': model.mlp1_0.conv.bias.detach().numpy().flatten().tolist(),
        'mlp1.mlp0.bn_mean': model.mlp1_0.bn.running_mean.detach().numpy().flatten().tolist(),
        'mlp1.mlp0.bn_var': model.mlp1_0.bn.running_var.detach().numpy().flatten().tolist(),
        'mlp1.mlp0.bn_gamma': model.mlp1_0.bn.weight.detach().numpy().flatten().tolist(),
        'mlp1.mlp0.bn_beta': model.mlp1_0.bn.bias.detach().numpy().flatten().tolist(),
        
        'mlp1.mlp1.weight': model.mlp1_1.conv.weight.squeeze().T.contiguous().detach().numpy().flatten().tolist(),
        'mlp1.mlp1.bias': model.mlp1_1.conv.bias.detach().numpy().flatten().tolist(),
        'mlp1.mlp1.bn_mean': model.mlp1_1.bn.running_mean.detach().numpy().flatten().tolist(),
        'mlp1.mlp1.bn_var': model.mlp1_1.bn.running_var.detach().numpy().flatten().tolist(),
        'mlp1.mlp1.bn_gamma': model.mlp1_1.bn.weight.detach().numpy().flatten().tolist(),
        'mlp1.mlp1.bn_beta': model.mlp1_1.bn.bias.detach().numpy().flatten().tolist(),
        
        # TNet2 weights (64x64 transformation)
        'tnet2.mlp0.weight': model.tnet2.mlp1.weight.squeeze().T.contiguous().detach().numpy().flatten().tolist(),
        'tnet2.mlp0.bias': model.tnet2.mlp1.bias.detach().numpy().flatten().tolist(),
        'tnet2.mlp0.bn_mean': model.tnet2.bn1.running_mean.detach().numpy().flatten().tolist(),
        'tnet2.mlp0.bn_var': model.tnet2.bn1.running_var.detach().numpy().flatten().tolist(),
        'tnet2.mlp0.bn_gamma': model.tnet2.bn1.weight.detach().numpy().flatten().tolist(),
        'tnet2.mlp0.bn_beta': model.tnet2.bn1.bias.detach().numpy().flatten().tolist(),
        
        'tnet2.mlp1.weight': model.tnet2.mlp2.weight.squeeze().T.contiguous().detach().numpy().flatten().tolist(),
        'tnet2.mlp1.bias': model.tnet2.mlp2.bias.detach().numpy().flatten().tolist(),
        'tnet2.mlp1.bn_mean': model.tnet2.bn2.running_mean.detach().numpy().flatten().tolist(),
        'tnet2.mlp1.bn_var': model.tnet2.bn2.running_var.detach().numpy().flatten().tolist(),
        'tnet2.mlp1.bn_gamma': model.tnet2.bn2.weight.detach().numpy().flatten().tolist(),
        'tnet2.mlp1.bn_beta': model.tnet2.bn2.bias.detach().numpy().flatten().tolist(),
        
        'tnet2.mlp2.weight': model.tnet2.mlp3.weight.squeeze().T.contiguous().detach().numpy().flatten().tolist(),
        'tnet2.mlp2.bias': model.tnet2.mlp3.bias.detach().numpy().flatten().tolist(),
        'tnet2.mlp2.bn_mean': model.tnet2.bn3.running_mean.detach().numpy().flatten().tolist(),
        'tnet2.mlp2.bn_var': model.tnet2.bn3.running_var.detach().numpy().flatten().tolist(),
        'tnet2.mlp2.bn_gamma': model.tnet2.bn3.weight.detach().numpy().flatten().tolist(),
        'tnet2.mlp2.bn_beta': model.tnet2.bn3.bias.detach().numpy().flatten().tolist(),
        
        'tnet2.fc0.weight': model.tnet2.fc1.weight.T.contiguous().detach().numpy().flatten().tolist(),
        'tnet2.fc0.bias': model.tnet2.fc1.bias.detach().numpy().flatten().tolist(),
        'tnet2.fc0.mean': model.tnet2.bn4.running_mean.detach().numpy().flatten().tolist(),
        'tnet2.fc0.var': model.tnet2.bn4.running_var.detach().numpy().flatten().tolist(),
        'tnet2.fc0.gamma': model.tnet2.bn4.weight.detach().numpy().flatten().tolist(),
        'tnet2.fc0.beta': model.tnet2.bn4.bias.detach().numpy().flatten().tolist(),
        
        'tnet2.fc1.weight': model.tnet2.fc2.weight.T.contiguous().detach().numpy().flatten().tolist(),
        'tnet2.fc1.bias': model.tnet2.fc2.bias.detach().numpy().flatten().tolist(),
        'tnet2.fc1.mean': model.tnet2.bn5.running_mean.detach().numpy().flatten().tolist(),
        'tnet2.fc1.var': model.tnet2.bn5.running_var.detach().numpy().flatten().tolist(),
        'tnet2.fc1.gamma': model.tnet2.bn5.weight.detach().numpy().flatten().tolist(),
        'tnet2.fc1.beta': model.tnet2.bn5.bias.detach().numpy().flatten().tolist(),
        
        'tnet2.fc2.weight': model.tnet2.fc3.weight.T.contiguous().detach().numpy().flatten().tolist(),
        'tnet2.fc2.bias': model.tnet2.fc3.bias.detach().numpy().flatten().tolist(),
        
        # MLP2 weights
        'mlp2.mlp0.weight': model.mlp2_0.conv.weight.squeeze().T.contiguous().detach().numpy().flatten().tolist(),
        'mlp2.mlp0.bias': model.mlp2_0.conv.bias.detach().numpy().flatten().tolist(),
        'mlp2.mlp0.bn_mean': model.mlp2_0.bn.running_mean.detach().numpy().flatten().tolist(),
        'mlp2.mlp0.bn_var': model.mlp2_0.bn.running_var.detach().numpy().flatten().tolist(),
        'mlp2.mlp0.bn_gamma': model.mlp2_0.bn.weight.detach().numpy().flatten().tolist(),
        'mlp2.mlp0.bn_beta': model.mlp2_0.bn.bias.detach().numpy().flatten().tolist(),
        
        'mlp2.mlp1.weight': model.mlp2_1.conv.weight.squeeze().T.contiguous().detach().numpy().flatten().tolist(),
        'mlp2.mlp1.bias': model.mlp2_1.conv.bias.detach().numpy().flatten().tolist(),
        'mlp2.mlp1.bn_mean': model.mlp2_1.bn.running_mean.detach().numpy().flatten().tolist(),
        'mlp2.mlp1.bn_var': model.mlp2_1.bn.running_var.detach().numpy().flatten().tolist(),
        'mlp2.mlp1.bn_gamma': model.mlp2_1.bn.weight.detach().numpy().flatten().tolist(),
        'mlp2.mlp1.bn_beta': model.mlp2_1.bn.bias.detach().numpy().flatten().tolist(),
    }
    
    # Save to JSON
    output_path = 'test/encoder/reference.json'
    with open(output_path, 'w') as f:
        json.dump(reference, f, indent=2)
    
    print(f"\n✓ Reference data saved to {output_path}")
    print(f"  Total parameters: {len(reference)} keys")


if __name__ == '__main__':
    main()

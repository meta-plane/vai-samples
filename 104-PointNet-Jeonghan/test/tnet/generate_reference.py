#!/usr/bin/env python3
"""
Generate PyTorch reference data for TNetBlock test
TNetBlock: Spatial Transformer Network
- Input: [N, K] point cloud
- Output: [N, K] transformed point cloud via learned K×K transformation matrix
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
    Learns a K×K transformation matrix and applies it to input
    """
    def __init__(self, K):
        super().__init__()
        self.K = K
        
        # MLP: K -> 64 -> 128 -> 1024
        self.mlp1 = nn.Conv1d(K, 64, 1)
        self.mlp2 = nn.Conv1d(64, 128, 1)
        self.mlp3 = nn.Conv1d(128, 1024, 1)
        
        # FC: 1024 -> 512 -> 256 -> K*K
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, K*K)
        
    def forward(self, x):
        """
        x: [N, K] input points
        Returns: [N, K] transformed points
        """
        N, K = x.shape
        
        # Path A: Generate transformation matrix
        # x: [N, K] -> [N, K, 1] for conv1d
        feat = x.unsqueeze(2)  # [N, K, 1]
        
        # MLP layers
        feat = self.mlp1(feat)  # [N, 64, 1]
        feat = self.mlp2(feat)  # [N, 128, 1]
        feat = self.mlp3(feat)  # [N, 1024, 1]
        
        # MaxPool: [N, 1024, 1] -> [N, 1024]
        feat = feat.squeeze(2)  # [N, 1024]
        feat = torch.max(feat, dim=0, keepdim=True)[0]  # [1, 1024]
        
        # FC layers
        feat = self.fc1(feat)  # [1, 512]
        feat = self.fc2(feat)  # [1, 256]
        feat = self.fc3(feat)  # [1, K*K]
        
        # Reshape to transformation matrix
        transform = feat.view(K, K)  # [K, K]
        
        # Path B: Apply transformation
        # x: [N, K] @ transform: [K, K] = [N, K]
        output = torch.matmul(x, transform)
        
        return output, transform

def main():
    # Test configuration
    N = 8   # Number of points
    K = 3   # Dimension (for 3D points)
    
    print(f"Generating TNetBlock reference data...")
    print(f"  Input: [{N}, {K}]")
    print(f"  Output: [{N}, {K}]")
    print(f"  Transform: [{K}, {K}]")
    
    # Create model
    model = TNetBlock(K)
    model.eval()
    
    # Generate random input
    x = torch.randn(N, K)
    
    # Forward pass
    with torch.no_grad():
        output, transform = model(x)
    
    # Collect weights
    weights = {
        # MLP weights
        'mlp.mlp0.conv.weight': model.mlp1.weight.squeeze().detach().numpy().tolist(),
        'mlp.mlp0.conv.bias': model.mlp1.bias.detach().numpy().tolist(),
        'mlp.mlp1.conv.weight': model.mlp2.weight.squeeze().detach().numpy().tolist(),
        'mlp.mlp1.conv.bias': model.mlp2.bias.detach().numpy().tolist(),
        'mlp.mlp2.conv.weight': model.mlp3.weight.squeeze().detach().numpy().tolist(),
        'mlp.mlp2.conv.bias': model.mlp3.bias.detach().numpy().tolist(),
        
        # FC weights
        'fc.fc0.weight': model.fc1.weight.detach().numpy().tolist(),
        'fc.fc0.bias': model.fc1.bias.detach().numpy().tolist(),
        'fc.fc1.weight': model.fc2.weight.detach().numpy().tolist(),
        'fc.fc1.bias': model.fc2.bias.detach().numpy().tolist(),
        'fc.fc2.weight': model.fc3.weight.detach().numpy().tolist(),
        'fc.fc2.bias': model.fc3.bias.detach().numpy().tolist(),
    }
    
    # Prepare data for JSON
    data = {
        'shape': [float(N), float(K)],
        'input': x.numpy().flatten().tolist(),
        'output': output.numpy().flatten().tolist(),
        'transform': transform.numpy().flatten().tolist(),
        'weights': weights
    }
    
    # Save to file
    output_file = 'test/tnet/reference.json'
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Saved to {output_file}")
    print(f"  Input values: {N * K}")
    print(f"  Output values: {N * K}")
    print(f"  Transform values: {K * K}")
    print(f"  Weight tensors: {len(weights)}")

if __name__ == '__main__':
    main()

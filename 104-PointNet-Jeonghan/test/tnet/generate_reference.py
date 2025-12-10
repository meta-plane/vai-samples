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
    Includes BatchNorm for compatibility with Vulkan implementation
    """
    def __init__(self, K):
        super().__init__()
        self.K = K
        
        # MLP: K -> 64 -> 128 -> 1024 (with BatchNorm)
        self.mlp1 = nn.Conv1d(K, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.mlp2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.mlp3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        
        # FC: 1024 -> 512 -> 256 -> K*K (with BatchNorm+ReLU for first 2, matches paper)
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, K*K)  # Last FC: no BN, no ReLU
        
    def forward(self, x):
        """
        x: [N, K] input points
        Returns: [N, K] transformed points
        """
        N, K = x.shape
        
        # Path A: Generate transformation matrix
        # x: [N, K] -> [N, K, 1] for conv1d
        feat = x.unsqueeze(2)  # [N, K, 1]
        
        # MLP layers with BatchNorm and ReLU
        feat = torch.relu(self.bn1(self.mlp1(feat)))  # [N, 64, 1]
        feat = torch.relu(self.bn2(self.mlp2(feat)))  # [N, 128, 1]
        feat = torch.relu(self.bn3(self.mlp3(feat)))  # [N, 1024, 1]
        
        # MaxPool: [N, 1024, 1] -> [N, 1024]
        feat = feat.squeeze(2)  # [N, 1024]
        feat = torch.max(feat, dim=0, keepdim=True)[0]  # [1, 1024]
        pooled = feat.squeeze(0)  # [1024] - 1D for FC layers
        
        # FC layers with BatchNorm+ReLU (matches paper STN3d)
        feat = self.fc1(pooled)            # [1024] -> [512]
        feat = self.bn4(feat.unsqueeze(0)) # Add batch: [1, 512] -> BN
        feat = feat.squeeze(0)             # Remove batch: [512]
        fc_out0 = torch.relu(feat)         # [512]
        
        feat = self.fc2(fc_out0)           # [512] -> [256]
        feat = self.bn5(feat.unsqueeze(0)) # Add batch: [1, 256] -> BN
        feat = feat.squeeze(0)             # Remove batch: [256]
        fc_out1 = torch.relu(feat)         # [256]
        
        fc_out2 = self.fc3(fc_out1)        # [256] -> [K*K], no BN, no ReLU
        
        # Reshape to transformation matrix
        transform_no_id = fc_out2.view(K, K)  # [K, K] before identity
        
        # Add identity matrix (matches paper STN3d)
        identity = torch.eye(K)  # [K, K]
        transform = transform_no_id + identity  # [K, K]
        
        # Path B: Apply transformation
        # x: [N, K] @ transform: [K, K] = [N, K]
        output = torch.matmul(x, transform)
        
        # Store intermediates for debugging
        self._intermediates = {
            'pooled': pooled,
            'fc_out0': fc_out0,
            'fc_out1': fc_out1,
            'fc_out2': fc_out2,
            'transform_no_identity': transform_no_id,
        }
        
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
        # MLP weights (Conv1d weights are [Cout, Cin, 1] → squeeze → transpose → [Cin, Cout])
        # MLP 0: Conv + BN + ReLU
        'mlp.mlp0.weight': model.mlp1.weight.squeeze().t().detach().numpy().tolist(),
        'mlp.mlp0.bias': model.mlp1.bias.detach().numpy().tolist(),
        'mlp.mlp0.mean': model.bn1.running_mean.detach().numpy().tolist(),
        'mlp.mlp0.var': model.bn1.running_var.detach().numpy().tolist(),
        'mlp.mlp0.gamma': model.bn1.weight.detach().numpy().tolist(),
        'mlp.mlp0.beta': model.bn1.bias.detach().numpy().tolist(),
        
        # MLP 1: Conv + BN + ReLU
        'mlp.mlp1.weight': model.mlp2.weight.squeeze().t().detach().numpy().tolist(),
        'mlp.mlp1.bias': model.mlp2.bias.detach().numpy().tolist(),
        'mlp.mlp1.mean': model.bn2.running_mean.detach().numpy().tolist(),
        'mlp.mlp1.var': model.bn2.running_var.detach().numpy().tolist(),
        'mlp.mlp1.gamma': model.bn2.weight.detach().numpy().tolist(),
        'mlp.mlp1.beta': model.bn2.bias.detach().numpy().tolist(),
        
        # MLP 2: Conv + BN + ReLU
        'mlp.mlp2.weight': model.mlp3.weight.squeeze().t().detach().numpy().tolist(),
        'mlp.mlp2.bias': model.mlp3.bias.detach().numpy().tolist(),
        'mlp.mlp2.mean': model.bn3.running_mean.detach().numpy().tolist(),
        'mlp.mlp2.var': model.bn3.running_var.detach().numpy().tolist(),
        'mlp.mlp2.gamma': model.bn3.weight.detach().numpy().tolist(),
        'mlp.mlp2.beta': model.bn3.bias.detach().numpy().tolist(),
        
        # FC weights with BatchNorm (FCBNSequence format: block0, block1, lastBlock)
        # Block 0: FC1 + BN4 + ReLU
        'fc.block0.weight': model.fc1.weight.t().detach().numpy().tolist(),
        'fc.block0.bias': model.fc1.bias.detach().numpy().tolist(),
        'fc.block0.mean': model.bn4.running_mean.detach().numpy().tolist(),
        'fc.block0.var': model.bn4.running_var.detach().numpy().tolist(),
        'fc.block0.gamma': model.bn4.weight.detach().numpy().tolist(),
        'fc.block0.beta': model.bn4.bias.detach().numpy().tolist(),
        
        # Block 1: FC2 + BN5 + ReLU
        'fc.block1.weight': model.fc2.weight.t().detach().numpy().tolist(),
        'fc.block1.bias': model.fc2.bias.detach().numpy().tolist(),
        'fc.block1.mean': model.bn5.running_mean.detach().numpy().tolist(),
        'fc.block1.var': model.bn5.running_var.detach().numpy().tolist(),
        'fc.block1.gamma': model.bn5.weight.detach().numpy().tolist(),
        'fc.block1.beta': model.bn5.bias.detach().numpy().tolist(),
        
        # Last block: FC3 only (no BN, no ReLU)
        'fc.lastBlock.weight': model.fc3.weight.t().detach().numpy().tolist(),
        'fc.lastBlock.bias': model.fc3.bias.detach().numpy().tolist(),
    }
    
    # Prepare data for JSON - flatten weights to top level
    data = {
        'shape': [float(N), float(K)],
        'input': x.numpy().flatten().tolist(),
        'output': output.numpy().flatten().tolist(),
        'transform': transform.numpy().flatten().tolist(),
        # Debug: intermediate values
        'debug_pooled': model._intermediates['pooled'].numpy().tolist(),
        'debug_fc_out0': model._intermediates['fc_out0'].numpy().tolist(),
        'debug_fc_out1': model._intermediates['fc_out1'].numpy().tolist(),
        'debug_fc_out2': model._intermediates['fc_out2'].numpy().tolist(),
        'debug_transform_no_id': model._intermediates['transform_no_identity'].numpy().flatten().tolist(),
    }
    
    # Add all weights to top level (flatten the structure)
    data.update(weights)
    
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

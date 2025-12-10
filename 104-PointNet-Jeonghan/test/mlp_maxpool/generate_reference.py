#!/usr/bin/env python3
"""
Generate reference for MLP+MaxPool (TNet's first part)
This isolates the MLP→MaxPool pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json

class MLPMaxPool(nn.Module):
    def __init__(self, K=3):
        super().__init__()
        self.K = K
        
        # MLP: K -> 64 -> 128 -> 1024
        self.mlp1 = nn.Conv1d(K, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.mlp2 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.mlp3 = nn.Conv1d(128, 1024, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(1024)
        
    def forward(self, x):
        # x: [N, K]
        feat = x.unsqueeze(2)  # [N, K, 1]
        
        # MLP layers
        feat = F.relu(self.bn1(self.mlp1(feat)))  # [N, 64, 1]
        feat = F.relu(self.bn2(self.mlp2(feat)))  # [N, 128, 1]
        feat = F.relu(self.bn3(self.mlp3(feat)))  # [N, 1024, 1]
        
        # MaxPool
        feat = feat.squeeze(2)  # [N, 1024]
        pooled = torch.max(feat, dim=0)[0]  # [1024]
        
        return pooled

def main():
    torch.manual_seed(42)
    
    N = 8
    K = 3
    
    print(f"Generating MLP+MaxPool reference...")
    print(f"  Input: [{N}, {K}]")
    print(f"  Output: [1024]")
    
    model = MLPMaxPool(K)
    model.eval()
    
    x = torch.randn(N, K)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"  Output range: [{output.min():.6f}, {output.max():.6f}]")
    
    weights = {
        'mlp.mlp0.weight': model.mlp1.weight.squeeze().t().detach().numpy().tolist(),
        'mlp.mlp0.bias': model.mlp1.bias.detach().numpy().tolist(),
        'mlp.mlp0.mean': model.bn1.running_mean.detach().numpy().tolist(),
        'mlp.mlp0.var': model.bn1.running_var.detach().numpy().tolist(),
        'mlp.mlp0.gamma': model.bn1.weight.detach().numpy().tolist(),
        'mlp.mlp0.beta': model.bn1.bias.detach().numpy().tolist(),
        
        'mlp.mlp1.weight': model.mlp2.weight.squeeze().t().detach().numpy().tolist(),
        'mlp.mlp1.bias': model.mlp2.bias.detach().numpy().tolist(),
        'mlp.mlp1.mean': model.bn2.running_mean.detach().numpy().tolist(),
        'mlp.mlp1.var': model.bn2.running_var.detach().numpy().tolist(),
        'mlp.mlp1.gamma': model.bn2.weight.detach().numpy().tolist(),
        'mlp.mlp1.beta': model.bn2.bias.detach().numpy().tolist(),
        
        'mlp.mlp2.weight': model.mlp3.weight.squeeze().t().detach().numpy().tolist(),
        'mlp.mlp2.bias': model.mlp3.bias.detach().numpy().tolist(),
        'mlp.mlp2.mean': model.bn3.running_mean.detach().numpy().tolist(),
        'mlp.mlp2.var': model.bn3.running_var.detach().numpy().tolist(),
        'mlp.mlp2.gamma': model.bn3.weight.detach().numpy().tolist(),
        'mlp.mlp2.beta': model.bn3.bias.detach().numpy().tolist(),
    }
    
    data = {
        'shape': [float(N), float(K)],
        'input': x.numpy().flatten().tolist(),
        'output': output.numpy().tolist(),
    }
    
    data.update(weights)
    
    output_file = 'test/mlp_maxpool/reference.json'
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Saved to {output_file}")
    print(f"  Weight tensors: {len(weights)}")

if __name__ == '__main__':
    main()

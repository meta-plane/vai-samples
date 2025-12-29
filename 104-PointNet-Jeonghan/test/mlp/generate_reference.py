#!/usr/bin/env python3
"""
Generate reference data for PointWiseMLPNode test in JSON format.
Tests: Conv1d(1x1) + BatchNorm1d + ReLU chain
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from safetensors.torch import save_file

# Fixed seed
torch.manual_seed(42)
np.random.seed(42)

# Config
N, C_in, C_out = 8, 3, 64  # 8 points, 3 input channels, 64 output channels

# Create MLP block (Conv1d 1x1 + BatchNorm + ReLU)
class PointWiseMLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x: [B, C_in, N]
        x = self.conv(x)      # [B, C_out, N]
        x = self.bn(x)        # [B, C_out, N]
        x = self.relu(x)      # [B, C_out, N]
        return x

# Create model
mlp = PointWiseMLP(C_in, C_out)

# Manually initialize BatchNorm parameters for testing (not using default values)
with torch.no_grad():
    mlp.bn.running_mean.uniform_(-0.5, 0.5)  # Random mean
    mlp.bn.running_var.uniform_(0.5, 1.5)     # Random variance (must be positive)
    mlp.bn.weight.uniform_(0.5, 1.5)          # gamma (scale)
    mlp.bn.bias.uniform_(-0.5, 0.5)           # beta (shift)

mlp.eval()

# Input: [B=1, C_in=3, N=8]
input_data = torch.randn(1, C_in, N)

# Forward
with torch.no_grad():
    output_data = mlp(input_data)

# Keep [C, N] layout (no transpose needed - PyTorch convention)
input_cn = input_data.squeeze(0).numpy()   # [3, 8]
output_cn = output_data.squeeze(0).numpy()  # [64, 8]

# Extract weights and biases
# Conv1d weight: [C_out, C_in, 1] -> keep PyTorch format [C_out, C_in]
conv_weight = mlp.conv.weight.detach().squeeze(-1).numpy()  # [64, 3]
conv_bias = mlp.conv.bias.detach().numpy()                   # [64]

# BatchNorm parameters
bn_mean = mlp.bn.running_mean.numpy()     # [64]
bn_var = mlp.bn.running_var.numpy()       # [64]
bn_gamma = mlp.bn.weight.detach().numpy()     # [64]
bn_beta = mlp.bn.bias.detach().numpy()        # [64]

# Create JSON structure (all floats for parseNDArray compatibility)
data = {
    "input": input_cn.flatten().tolist(),           # [24 values] (3*8)
    "expected": output_cn.flatten().tolist(),       # [512 values] (64*8)
    "conv_weight": conv_weight.flatten().tolist(),  # [192 values] (64*3)
    "conv_bias": conv_bias.tolist(),                # [64 values]
    "bn_mean": bn_mean.tolist(),                    # [64 values]
    "bn_var": bn_var.tolist(),                      # [64 values]
    "bn_gamma": bn_gamma.tolist(),                  # [64 values]
    "bn_beta": bn_beta.tolist(),                    # [64 values]
    "bn_eps": float(mlp.bn.eps),
    "shape": [float(C_in), float(N), float(C_out)]  # [3, 8, 64] - C_in, N, C_out order
}

# Prepare output directory
output_dir = Path("test/mlp")
output_dir.mkdir(parents=True, exist_ok=True)

# Save JSON (backward compatibility)
json_path = output_dir / "reference.json"
with open(json_path, 'w') as f:
    json.dump(data, f, indent=2)

# Save SafeTensors (preferred format) - ensure all tensors are contiguous
tensors = {
    "input": torch.from_numpy(input_cn).contiguous(),            # [3, 8]
    "expected": torch.from_numpy(output_cn).contiguous(),        # [64, 8]
    "conv_weight": torch.from_numpy(conv_weight).contiguous(),   # [64, 3]
    "conv_bias": torch.from_numpy(conv_bias).contiguous(),       # [64]
    "bn_mean": torch.from_numpy(bn_mean).contiguous(),           # [64]
    "bn_var": torch.from_numpy(bn_var).contiguous(),             # [64]
    "bn_gamma": torch.from_numpy(bn_gamma).contiguous(),         # [64]
    "bn_beta": torch.from_numpy(bn_beta).contiguous(),           # [64]
    "bn_eps": torch.tensor([mlp.bn.eps], dtype=torch.float32),   # [1]
    "shape": torch.tensor([C_in, N, C_out], dtype=torch.float32)  # [3] - C_in, N, C_out
}

safetensors_path = output_dir / "reference.safetensors"
save_file(tensors, str(safetensors_path))

print("PointWiseMLP Reference Generated (PyTorch Convention: [C, N])")
print(f"  Input shape:  {input_cn.shape} -> {input_cn.size} values")
print(f"  Output shape: {output_cn.shape} -> {output_cn.size} values")
print(f"  Conv weight:  {conv_weight.shape} (PyTorch: [C_out, C_in])")
print(f"  Conv bias:    {conv_bias.shape}")
print(f"  BN mean:      {bn_mean.shape}")
print(f"  BN var:       {bn_var.shape}")
print(f"  BN gamma:     {bn_gamma.shape}")
print(f"  BN beta:      {bn_beta.shape}")
print(f"  BN eps:       {mlp.bn.eps}")
print(f"\n✅ Saved to:")
print(f"  - {json_path} (legacy)")
print(f"  - {safetensors_path} (preferred)")

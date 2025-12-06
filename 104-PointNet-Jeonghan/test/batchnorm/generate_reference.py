#!/usr/bin/env python3
"""
Generate reference data for BatchNorm1D test in JSON format.
Simple test: 5 points, 3 channels = 15 values.
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path

# Fixed seed
torch.manual_seed(42)
np.random.seed(42)

# Config
N, C = 5, 3  # 5 points, 3 channels

# Create BatchNorm
bn = nn.BatchNorm1d(C)
bn.eval()

# Input: [B=1, C=3, N=5]
input_data = torch.randn(1, C, N)

# Forward
with torch.no_grad():
    output_data = bn(input_data)

# Transpose to [N, C] for C++
input_nc = input_data.squeeze(0).transpose(0, 1).numpy()   # [5, 3]
output_nc = output_data.squeeze(0).transpose(0, 1).numpy()  # [5, 3]

# Create JSON structure (shape as float array for parseNDArray compatibility)
data = {
    "input": input_nc.flatten().tolist(),      # [15 values]
    "expected": output_nc.flatten().tolist(),  # [15 values]
    "mean": bn.running_mean.numpy().tolist(),  # [3 values]
    "var": bn.running_var.numpy().tolist(),    # [3 values]
    "gamma": bn.weight.data.numpy().tolist(),  # [3 values]
    "beta": bn.bias.data.numpy().tolist(),     # [3 values]
    "eps": float(bn.eps),
    "shape": [float(N), float(C)]  # Convert to floats for parseNDArray
}

# Save JSON
output_dir = Path("test/batchnorm")
output_dir.mkdir(parents=True, exist_ok=True)

json_path = output_dir / "reference.json"
with open(json_path, 'w') as f:
    json.dump(data, f, indent=2)

print("BatchNorm1D Reference Generated (JSON)")
print(f"  Input:  {input_nc.shape} -> {input_nc.size} values")
print(f"  Output: {output_nc.shape} -> {output_nc.size} values")
print(f"  Mean:   {bn.running_mean.shape}")
print(f"  Var:    {bn.running_var.shape}")
print(f"  Gamma:  {bn.weight.shape}")
print(f"  Beta:   {bn.bias.shape}")
print(f"  Eps:    {bn.eps}")
print(f"\n✓ Saved to {json_path}")

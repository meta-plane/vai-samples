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

# Keep PyTorch format [C, N] - no transpose needed!
input_cn = input_data.squeeze(0).numpy()   # [3, 5] - PyTorch native
output_cn = output_data.squeeze(0).numpy()  # [3, 5] - PyTorch native

# Create JSON structure (shape as float array for parseNDArray compatibility)
data = {
    "input": input_cn.flatten().tolist(),      # [15 values] - [C, N] layout
    "expected": output_cn.flatten().tolist(),  # [15 values] - [C, N] layout
    "mean": bn.running_mean.numpy().tolist(),  # [3 values]
    "var": bn.running_var.numpy().tolist(),    # [3 values]
    "gamma": bn.weight.data.numpy().tolist(),  # [3 values]
    "beta": bn.bias.data.numpy().tolist(),     # [3 values]
    "eps": float(bn.eps),
    "shape": [float(C), float(N)]  # [C, N] layout - PyTorch convention
}

# Save JSON
output_dir = Path("test/batchnorm")
output_dir.mkdir(parents=True, exist_ok=True)

json_path = output_dir / "reference.json"
with open(json_path, 'w') as f:
    json.dump(data, f, indent=2)

print("BatchNorm1D Reference Generated (JSON)")
print(f"  Input:  {input_cn.shape} -> {input_cn.size} values [C, N]")
print(f"  Output: {output_cn.shape} -> {output_cn.size} values [C, N]")
print(f"  Mean:   {bn.running_mean.shape}")
print(f"  Var:    {bn.running_var.shape}")
print(f"  Gamma:  {bn.weight.shape}")
print(f"  Beta:   {bn.bias.shape}")
print(f"  Eps:    {bn.eps}")
print(f"\n✓ Saved to {json_path}")

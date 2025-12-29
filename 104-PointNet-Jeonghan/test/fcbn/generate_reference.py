#!/usr/bin/env python3
"""
Generate reference data for FCBNNode (FC + BatchNorm + ReLU) test
Tests: FullyConnected + BatchNorm + ReLU block matching PointNet paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from safetensors.torch import save_file

# Fixed seed
torch.manual_seed(42)
np.random.seed(42)

# Config
I, O = 128, 256  # 128 input features, 256 output features

print("="*60)
print("Generating FCBNNode Reference (FC + BN + ReLU)")
print("="*60)

# Create FC + BN layers (no ReLU wrapper, we'll apply it manually)
fc = nn.Linear(I, O)
bn = nn.BatchNorm1d(O)
fc.eval()
bn.eval()

# Input: [1, I] - batch dimension for BatchNorm
input_data = torch.randn(1, I)

print(f"Input shape:  [1, {I}]")
print(f"Output shape: [1, {O}]")

# Forward: FC -> BN -> ReLU
with torch.no_grad():
    x = fc(input_data)           # [1, I] -> [1, O]
    x = bn(x)                    # BatchNorm: [1, O] -> [1, O]
    output_data = F.relu(x)      # ReLU: [1, O]

print(f"FC output range:   [{x.min():.4f}, {x.max():.4f}]")
print(f"Final output range: [{output_data.min():.4f}, {output_data.max():.4f}]")

# Convert to numpy
input_np = input_data.numpy()      # [1, 128]
output_np = output_data.numpy()    # [1, 256]

# Extract weights and biases
# FC: Keep PyTorch format [O, I] (no transpose!)
fc_weight = fc.weight.detach().numpy()  # [256, 128]
fc_bias = fc.bias.detach().numpy()       # [256]

# BN: BatchNorm parameters
bn_mean = bn.running_mean.numpy()     # [256]
bn_var = bn.running_var.numpy()       # [256]
bn_gamma = bn.weight.detach().numpy() # [256]
bn_beta = bn.bias.detach().numpy()    # [256]

print(f"\nWeights:")
print(f"  FC weight: [{O}, {I}] (PyTorch format)")
print(f"  FC bias:   [{O}]")
print(f"  BN mean:   [{O}]")
print(f"  BN var:    [{O}]")
print(f"  BN gamma:  [{O}]")
print(f"  BN beta:   [{O}]")

# Create JSON structure
data = {
    "input": input_np.flatten().tolist(),          # [128 values] - flatten [1, 128]
    "expected": output_np.flatten().tolist(),      # [256 values] - flatten [1, 256]
    "weight": fc_weight.flatten().tolist(),        # [32768 values]
    "bias": fc_bias.tolist(),                      # [256 values]
    "mean": bn_mean.tolist(),                      # [256 values]
    "var": bn_var.tolist(),                        # [256 values]
    "gamma": bn_gamma.tolist(),                    # [256 values]
    "beta": bn_beta.tolist(),                      # [256 values]
    "shape": [float(I), float(O)]                  # [128, 256]
}

# Save JSON (backward compatibility)
output_dir = Path("test/fcbn")
output_dir.mkdir(parents=True, exist_ok=True)

json_path = output_dir / "reference.json"
with open(json_path, 'w') as f:
    json.dump(data, f)

# Save SafeTensors (preferred format)
tensors = {
    "input": torch.from_numpy(input_np).contiguous(),
    "expected": torch.from_numpy(output_np).contiguous(),
    "weight": torch.from_numpy(fc_weight).contiguous(),
    "bias": torch.from_numpy(fc_bias).contiguous(),
    "mean": torch.from_numpy(bn_mean).contiguous(),
    "var": torch.from_numpy(bn_var).contiguous(),
    "gamma": torch.from_numpy(bn_gamma).contiguous(),
    "beta": torch.from_numpy(bn_beta).contiguous(),
    "shape": torch.tensor([I, O], dtype=torch.float32)
}

safetensors_path = output_dir / "reference.safetensors"
save_file(tensors, str(safetensors_path))

print(f"\n✅ Saved to:")
print(f"  - {json_path} (legacy)")
print(f"  - {safetensors_path} (preferred)")
print("="*60)

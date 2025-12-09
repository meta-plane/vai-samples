#!/usr/bin/env python3
"""
Generate reference data for FCBNSequence test
Tests: Sequence of FC+BN+ReLU blocks (last block is FC only, no BN+ReLU)
Matches PointNet paper STN3d fc1, fc2, fc3 layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path

# Fixed seed
torch.manual_seed(42)
np.random.seed(42)

# Config: [512] -> [256] -> [128] -> [9]
# First two blocks have BN+ReLU, last block is FC only
dims = [512, 256, 128, 9]

print("="*60)
print("Generating FCBNSequence Reference (3 blocks)")
print("="*60)
print(f"Architecture: {dims[0]} -> {dims[1]} -> {dims[2]} -> {dims[3]}")
print(f"  Block 0: FC+BN+ReLU ({dims[0]} -> {dims[1]})")
print(f"  Block 1: FC+BN+ReLU ({dims[1]} -> {dims[2]})")
print(f"  Block 2: FC only    ({dims[2]} -> {dims[3]})")

# Create layers
fc1 = nn.Linear(dims[0], dims[1])
bn1 = nn.BatchNorm1d(dims[1])
fc2 = nn.Linear(dims[1], dims[2])
bn2 = nn.BatchNorm1d(dims[2])
fc3 = nn.Linear(dims[2], dims[3])  # Last FC: no BN, no ReLU

fc1.eval()
bn1.eval()
fc2.eval()
bn2.eval()
fc3.eval()

# Input: [512] - 1D (after global pooling in PointNet)
input_data = torch.randn(dims[0])

print(f"\nInput shape:  [{dims[0]}]")
print(f"Output shape: [{dims[3]}]")

# Forward: FC1+BN1+ReLU -> FC2+BN2+ReLU -> FC3
with torch.no_grad():
    # Block 0: FC+BN+ReLU
    x = fc1(input_data)      # [512] -> [256]
    x = bn1(x.unsqueeze(0))  # Add batch: [1, 256] -> BN -> [1, 256]
    x = x.squeeze(0)         # Remove batch: [256]
    x = F.relu(x)            # [256]
    
    # Block 1: FC+BN+ReLU
    x = fc2(x)               # [256] -> [128]
    x = bn2(x.unsqueeze(0))  # Add batch: [1, 128] -> BN -> [1, 128]
    x = x.squeeze(0)         # Remove batch: [128]
    x = F.relu(x)            # [128]
    
    # Block 2: FC only (no BN, no ReLU)
    output_data = fc3(x)     # [128] -> [9]

print(f"Final output range: [{output_data.min():.4f}, {output_data.max():.4f}]")

# Convert to numpy
input_np = input_data.numpy()      # [512]
output_np = output_data.numpy()    # [9]

# Extract weights and biases
# Block 0
fc1_weight = fc1.weight.detach().transpose(0, 1).numpy()  # [512, 256]
fc1_bias = fc1.bias.detach().numpy()                       # [256]
bn1_mean = bn1.running_mean.numpy()                        # [256]
bn1_var = bn1.running_var.numpy()                          # [256]
bn1_gamma = bn1.weight.detach().numpy()                    # [256]
bn1_beta = bn1.bias.detach().numpy()                       # [256]

# Block 1
fc2_weight = fc2.weight.detach().transpose(0, 1).numpy()  # [256, 128]
fc2_bias = fc2.bias.detach().numpy()                       # [128]
bn2_mean = bn2.running_mean.numpy()                        # [128]
bn2_var = bn2.running_var.numpy()                          # [128]
bn2_gamma = bn2.weight.detach().numpy()                    # [128]
bn2_beta = bn2.bias.detach().numpy()                       # [128]

# Block 2 (last block: FC only)
fc3_weight = fc3.weight.detach().transpose(0, 1).numpy()  # [128, 9]
fc3_bias = fc3.bias.detach().numpy()                       # [9]

print(f"\nWeights:")
print(f"  Block 0 FC:  [{dims[0]}, {dims[1]}]")
print(f"  Block 0 BN:  [{dims[1]}] x 4")
print(f"  Block 1 FC:  [{dims[1]}, {dims[2]}]")
print(f"  Block 1 BN:  [{dims[2]}] x 4")
print(f"  Block 2 FC:  [{dims[2]}, {dims[3]}]")

# Create JSON structure
data = {
    "input": input_np.tolist(),                        # [512 values]
    "expected": output_np.tolist(),                    # [9 values]
    
    # Block 0 (FC+BN+ReLU)
    "block0.weight": fc1_weight.flatten().tolist(),    # [131072 values]
    "block0.bias": fc1_bias.tolist(),                  # [256 values]
    "block0.mean": bn1_mean.tolist(),                  # [256 values]
    "block0.var": bn1_var.tolist(),                    # [256 values]
    "block0.gamma": bn1_gamma.tolist(),                # [256 values]
    "block0.beta": bn1_beta.tolist(),                  # [256 values]
    
    # Block 1 (FC+BN+ReLU)
    "block1.weight": fc2_weight.flatten().tolist(),    # [32768 values]
    "block1.bias": fc2_bias.tolist(),                  # [128 values]
    "block1.mean": bn2_mean.tolist(),                  # [128 values]
    "block1.var": bn2_var.tolist(),                    # [128 values]
    "block1.gamma": bn2_gamma.tolist(),                # [128 values]
    "block1.beta": bn2_beta.tolist(),                  # [128 values]
    
    # Block 2 (FC only)
    "lastBlock.weight": fc3_weight.flatten().tolist(), # [1152 values]
    "lastBlock.bias": fc3_bias.tolist(),               # [9 values]
    
    "shape": [float(d) for d in dims]                  # [512, 256, 128, 9]
}

# Save JSON
output_dir = Path("test/fcbn_seq")
output_dir.mkdir(parents=True, exist_ok=True)

json_path = output_dir / "reference.json"
with open(json_path, 'w') as f:
    json.dump(data, f)

print(f"\n✓ Saved to {json_path}")
print("="*60)

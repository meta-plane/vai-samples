#!/usr/bin/env python3
"""
Generate reference data for FullyConnectedNode test in JSON format.
Tests: Linear layer (Fully Connected)
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
I, O = 128, 256  # 128 input features, 256 output features

# Create FC layer
fc = nn.Linear(I, O)
fc.eval()

# Input: [I]
input_data = torch.randn(I)

# Forward
with torch.no_grad():
    output_data = fc(input_data)

# Convert to numpy
input_np = input_data.numpy()      # [128]
output_np = output_data.numpy()    # [256]

# Extract weights and biases
# Linear weight: [O, I] - keep PyTorch format
weight = fc.weight.detach().numpy()  # [256, 128] - no transpose!
bias = fc.bias.detach().numpy()       # [256]

# Create JSON structure (all floats for parseNDArray compatibility)
data = {
    "input": input_np.tolist(),           # [128 values]
    "expected": output_np.tolist(),       # [256 values]
    "weight": weight.flatten().tolist(),  # [32768 values] (128*256)
    "bias": bias.tolist(),                # [256 values]
    "shape": [float(I), float(O)]         # [128, 256]
}

# Prepare output directory
output_dir = Path("test/fc")
output_dir.mkdir(parents=True, exist_ok=True)

# Save JSON (backward compatibility)
json_path = output_dir / "reference.json"
with open(json_path, 'w') as f:
    json.dump(data, f, indent=2)

# Save SafeTensors (preferred format)
tensors = {
    "input": torch.from_numpy(input_np).contiguous(),      # [128]
    "expected": torch.from_numpy(output_np).contiguous(),  # [256]
    "weight": torch.from_numpy(weight).contiguous(),       # [256, 128]
    "bias": torch.from_numpy(bias).contiguous(),           # [256]
    "shape": torch.tensor([I, O], dtype=torch.float32)     # [2]
}

safetensors_path = output_dir / "reference.safetensors"
save_file(tensors, str(safetensors_path))

print("FullyConnected Reference Generated (PyTorch Convention: [O, I])")
print(f"  Input shape:  ({I},) -> {I} values")
print(f"  Output shape: ({O},) -> {O} values")
print(f"  Weight:       ({O}, {I}) - PyTorch format")
print(f"  Bias:         ({O},)")
print(f"\n✅ Saved to:")
print(f"  - {json_path} (legacy)")
print(f"  - {safetensors_path} (preferred)")

#!/usr/bin/env python3
"""
Generate reference data for FCSequence test in JSON format.
Tests: Sequential Fully Connected layers (e.g., 128 -> 256 -> 512 -> 64)
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path

# Fixed seed
torch.manual_seed(42)
np.random.seed(42)

# Config: 3-layer FC sequence
channels = [128, 256, 512, 64]  # 3 FC layers: 128->256, 256->512, 512->64

# Create FC sequence
class FCSequence(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(channels[i], channels[i+1]) 
            for i in range(len(channels)-1)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Create model
fc_seq = FCSequence(channels)
fc_seq.eval()

# Input: [I] = [128]
input_data = torch.randn(channels[0])

# Forward
with torch.no_grad():
    output_data = fc_seq(input_data)

# Convert to numpy
input_np = input_data.numpy()      # [128]
output_np = output_data.numpy()    # [64]

# Extract weights and biases for each layer
weights = []
biases = []
for i, layer in enumerate(fc_seq.layers):
    # Linear weight: [O, I] -> transpose to [I, O] for GEMM
    weight = layer.weight.detach().transpose(0, 1).numpy()
    bias = layer.bias.detach().numpy()
    
    weights.append(weight.flatten().tolist())
    biases.append(bias.tolist())
    
    print(f"FC{i}: [{channels[i]}, {channels[i+1]}] weight shape: {weight.shape}")

# Create JSON structure
data = {
    "input": input_np.tolist(),           # [128 values]
    "expected": output_np.tolist(),       # [64 values]
    "channels": [float(c) for c in channels],  # [128, 256, 512, 64]
    
    # Weights for each FC layer
    "fc0.weight": weights[0],  # [128, 256]
    "fc0.bias": biases[0],     # [256]
    "fc1.weight": weights[1],  # [256, 512]
    "fc1.bias": biases[1],     # [512]
    "fc2.weight": weights[2],  # [512, 64]
    "fc2.bias": biases[2],     # [64]
}

# Save JSON
output_dir = Path("test/fcseq")
output_dir.mkdir(parents=True, exist_ok=True)

json_path = output_dir / "reference.json"
with open(json_path, 'w') as f:
    json.dump(data, f)

print("\nFCSequence Reference Generated (JSON)")
print(f"  Architecture: {' -> '.join(map(str, channels))}")
print(f"  Input shape:  ({channels[0]},) -> {channels[0]} values")
print(f"  Output shape: ({channels[-1]},) -> {channels[-1]} values")
print(f"  Layers: {len(channels)-1}")
print()
print(f"✓ Saved to {json_path}")

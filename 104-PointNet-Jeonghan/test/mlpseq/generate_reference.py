import torch
import torch.nn as nn
import json
import numpy as np
from pathlib import Path
from safetensors.torch import save_file

torch.manual_seed(42)

# Test configuration: 3-layer MLP sequence
# Each MLP = Conv1d(1x1) + BatchNorm1d + ReLU
channels = [3, 64, 128, 256]  # 3 MLP layers
N = 8  # Number of points

print(f"\nMLPSequence Test Configuration:")
print(f"  Architecture: {' -> '.join(map(str, channels))}")
print(f"  Layers: {len(channels) - 1} MLP blocks")
print(f"  Points: {N}")
print()

# Create PyTorch MLP blocks (Conv1d + BatchNorm1d + ReLU)
class MLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# Create 3 MLP blocks
mlp0 = MLPBlock(channels[0], channels[1])
mlp1 = MLPBlock(channels[1], channels[2])
mlp2 = MLPBlock(channels[2], channels[3])

# Set to eval mode to use running stats
mlp0.eval()
mlp1.eval()
mlp2.eval()

# Generate random input: [C_in, N] format (PyTorch convention)
# PyTorch Conv1d needs [1, C_in, N]
input_data = torch.randn(channels[0], N)
input_torch = input_data.unsqueeze(0)  # [1, C_in, N]

print(f"Input shape: [C_in={channels[0]}, N={N}] -> PyTorch: {input_torch.shape}")

# Forward pass through 3 MLP blocks
with torch.no_grad():
    out0 = mlp0(input_torch)
    out1 = mlp1(out0)
    output = mlp2(out1)

print(f"Output shape: PyTorch {output.shape} -> Vulkan [C_out={channels[-1]}, N={N}]")
print()

# Keep [C_out, N] layout (no transpose!)
output_data = output.squeeze(0)

# Prepare JSON data
json_data = {
    "shape": [float(channels[0]), float(N)],  # [C_in, N] order
    "input": input_data.flatten().tolist(),
    "expected": output_data.flatten().tolist(),
}

# Save weights for each MLP block
for i, mlp in enumerate([mlp0, mlp1, mlp2]):
    prefix = f"mlp{i}"
    
    # Conv weight: Keep PyTorch format [C_out, C_in] (no transpose!)
    conv_weight = mlp.conv.weight.squeeze(-1)
    conv_bias = mlp.conv.bias
    
    # BatchNorm parameters
    bn_mean = mlp.bn.running_mean
    bn_var = mlp.bn.running_var
    bn_gamma = mlp.bn.weight
    bn_beta = mlp.bn.bias
    
    print(f"{prefix.upper()}: [{channels[i]}, {channels[i+1]}]")
    print(f"  conv.weight shape: {conv_weight.shape} (PyTorch: [C_out, C_in])")
    print(f"  conv.bias shape: {conv_bias.shape}")
    print(f"  bn parameters shape: {bn_mean.shape}")
    
    json_data[f"{prefix}.conv.weight"] = conv_weight.flatten().tolist()
    json_data[f"{prefix}.conv.bias"] = conv_bias.tolist()
    json_data[f"{prefix}.bn.mean"] = bn_mean.tolist()
    json_data[f"{prefix}.bn.var"] = bn_var.tolist()
    json_data[f"{prefix}.bn.gamma"] = bn_gamma.tolist()
    json_data[f"{prefix}.bn.beta"] = bn_beta.tolist()

# Prepare output directory
output_dir = Path("test/mlpseq")
output_dir.mkdir(parents=True, exist_ok=True)

# Save to JSON (backward compatibility)
json_path = output_dir / "reference.json"
with open(json_path, 'w') as f:
    json.dump(json_data, f, indent=2)

# Save SafeTensors (preferred format)
tensors = {
    "input": input_data.contiguous(),
    "expected": output_data.contiguous(),
    "shape": torch.tensor([channels[0], N], dtype=torch.float32),
}

# Add MLP weights
for i, mlp in enumerate([mlp0, mlp1, mlp2]):
    prefix = f"mlp{i}"
    conv_weight = mlp.conv.weight.squeeze(-1).contiguous()
    conv_bias = mlp.conv.bias.contiguous()
    bn_mean = mlp.bn.running_mean.contiguous()
    bn_var = mlp.bn.running_var.contiguous()
    bn_gamma = mlp.bn.weight.contiguous()
    bn_beta = mlp.bn.bias.contiguous()
    
    tensors[f"{prefix}.conv.weight"] = conv_weight
    tensors[f"{prefix}.conv.bias"] = conv_bias
    tensors[f"{prefix}.bn.mean"] = bn_mean
    tensors[f"{prefix}.bn.var"] = bn_var
    tensors[f"{prefix}.bn.gamma"] = bn_gamma
    tensors[f"{prefix}.bn.beta"] = bn_beta

safetensors_path = output_dir / "reference.safetensors"
save_file(tensors, str(safetensors_path))

print()
print(f"MLPSequence Reference Generated (PyTorch Convention)")
print(f"  Architecture: {' -> '.join(map(str, channels))}")
print(f"  Input shape:  [C_in={channels[0]}, N={N}] -> {channels[0] * N} values")
print(f"  Output shape: [C_out={channels[-1]}, N={N}] -> {channels[-1] * N} values")
print(f"  Layers: {len(channels) - 1}")
print(f"\n✅ Saved to:")
print(f"  - {json_path} (legacy)")
print(f"  - {safetensors_path} (preferred)")
print()

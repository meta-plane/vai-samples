import torch
import torch.nn as nn
import json
import numpy as np

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

# Generate random input: [N, C_in] in Vulkan format
# PyTorch needs [1, C_in, N] for Conv1d
input_data = torch.randn(N, channels[0])
input_torch = input_data.unsqueeze(0).transpose(1, 2)  # [1, C_in, N]

print(f"Input shape: {input_data.shape} -> PyTorch: {input_torch.shape}")

# Forward pass through 3 MLP blocks
with torch.no_grad():
    out0 = mlp0(input_torch)
    out1 = mlp1(out0)
    output = mlp2(out1)

print(f"Output shape: PyTorch {output.shape} -> Vulkan [{N}, {channels[-1]}]")
print()

# Convert output: [1, C_out, N] -> [N, C_out]
output_data = output.squeeze(0).transpose(0, 1)

# Prepare JSON data
json_data = {
    "shape": [float(N), float(channels[0])],
    "input": input_data.flatten().tolist(),
    "expected": output_data.flatten().tolist(),
}

# Save weights for each MLP block
for i, mlp in enumerate([mlp0, mlp1, mlp2]):
    prefix = f"mlp{i}"
    
    # Conv weight: [C_out, C_in, 1] -> transpose to [C_in, C_out] for GEMM
    conv_weight = mlp.conv.weight.squeeze(-1).transpose(0, 1)
    conv_bias = mlp.conv.bias
    
    # BatchNorm parameters
    bn_mean = mlp.bn.running_mean
    bn_var = mlp.bn.running_var
    bn_gamma = mlp.bn.weight
    bn_beta = mlp.bn.bias
    
    print(f"{prefix.upper()}: [{channels[i]}, {channels[i+1]}]")
    print(f"  conv.weight shape: {conv_weight.shape}")
    print(f"  conv.bias shape: {conv_bias.shape}")
    print(f"  bn parameters shape: {bn_mean.shape}")
    
    json_data[f"{prefix}.conv.weight"] = conv_weight.flatten().tolist()
    json_data[f"{prefix}.conv.bias"] = conv_bias.tolist()
    json_data[f"{prefix}.bn.mean"] = bn_mean.tolist()
    json_data[f"{prefix}.bn.var"] = bn_var.tolist()
    json_data[f"{prefix}.bn.gamma"] = bn_gamma.tolist()
    json_data[f"{prefix}.bn.beta"] = bn_beta.tolist()

# Save to JSON
output_file = "test/mlpseq/reference.json"
with open(output_file, 'w') as f:
    json.dump(json_data, f, indent=2)

print()
print(f"MLPSequence Reference Generated (JSON)")
print(f"  Architecture: {' -> '.join(map(str, channels))}")
print(f"  Input shape:  ({N}, {channels[0]}) -> {N * channels[0]} values")
print(f"  Output shape: ({N}, {channels[-1]}) -> {N * channels[-1]} values")
print(f"  Layers: {len(channels) - 1}")
print()
print(f"✓ Saved to {output_file}")

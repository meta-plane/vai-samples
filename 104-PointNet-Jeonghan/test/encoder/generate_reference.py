#!/usr/bin/env python3
"""
Generate reference data for PointNetEncoder test with actual inference
yanx27 structure: channel-dimensional input (3 or 6)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../temp_pytorch_pointnet/models'))

import torch
import torch.nn as nn
import numpy as np
from safetensors.torch import save_file
from pointnet_utils import PointNetEncoder

torch.manual_seed(42)
np.random.seed(42)

# Test configuration
N = 16  # num_points
channel = 3  # 3 for xyz, 6 for xyz+normals
batch_size = 1

# Generate input: [B, D, N] format (PyTorch uses [batch, channel, points])
input_data = torch.randn(batch_size, channel, N, dtype=torch.float32)

# Create model and run inference
model = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
model.eval()

print("Generating reference for PointNetEncoder...")
print(f"  Input shape (PyTorch): [B={batch_size}, D={channel}, N={N}]")

# Run inference to get actual output
with torch.no_grad():
    output, trans, trans_feat = model(input_data)
    print(f"  Output shape (PyTorch): {output.shape}")  # [B, 1024]

# For Vulkan, we need to manually compute point-wise features
# Let's trace through the encoder step by step
with torch.no_grad():
    x = input_data
    B, D, N_points = x.size()
    
    # STN3d
    trans_input = model.stn(x)  # [B, 3, 3] or [B, 6, 6]
    x = x.transpose(2, 1)  # [B, N, D]
    x = torch.bmm(x, trans_input)  # Apply transformation
    x = x.transpose(2, 1)  # [B, D, N]
    
    # Conv1
    x = torch.relu(model.bn1(model.conv1(x)))  # [B, 64, N]
    
    # STNkd (feature transform)
    trans_feat_custom = model.fstn(x)  # [B, 64, 64]
    x = x.transpose(2, 1)  # [B, N, 64]
    x = torch.bmm(x, trans_feat_custom)  # Apply transformation
    x = x.transpose(2, 1)  # [B, 64, N]
    
    # Conv2, Conv3
    x = torch.relu(model.bn2(model.conv2(x)))  # [B, 128, N]
    x = model.bn3(model.conv3(x))  # [B, 1024, N]
    
    # For Vulkan test: point-wise features [B, 1024, N] -> [N, 1024]
    pointwise_features = x[0].transpose(0, 1).contiguous()  # [N, 1024]
    print(f"  Point-wise features shape: {pointwise_features.shape}")

# Convert input to [N, channel] for Vulkan
input_vulkan = input_data[0].transpose(0, 1).contiguous()  # [channel, N] → [N, channel]

# Save input, expected output, and all model weights
tensors = {
    'input': input_vulkan,
    'expected_output': pointwise_features,  # [N, 1024]
}

# Save all model weights with proper transposing for Vulkan GEMM
# Conv1d weights: [out, in, 1] -> squeeze and transpose to [in, out]
# Linear weights: [out, in] -> transpose to [in, out]
for name, param in model.state_dict().items():
    if 'conv' in name and '.weight' in name and not 'bn' in name:
        # Conv1d: [C_out, C_in, 1] -> [C_in, C_out]
        tensors[name] = param.squeeze(-1).transpose(0, 1).contiguous()
    elif 'fc' in name and '.weight' in name:
        # Linear: [out, in] -> [in, out]
        tensors[name] = param.transpose(0, 1).contiguous()
    else:
        # BatchNorm, bias, running_mean, running_var - no transpose
        tensors[name] = param

save_file(tensors, 'test/encoder/reference.safetensors')

print(f"✓ Reference data saved")
print(f"  Weights: {len(model.state_dict())} tensors")
print(f"  Input: [N={N}, channel={channel}]")
print(f"  Expected output: [N={N}, 1024]")
print(f"\nFirst 5 values of expected output (first point):")
for i in range(5):
    print(f"  [{i}] = {pointwise_features[0, i].item():.6f}")

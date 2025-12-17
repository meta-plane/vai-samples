#!/usr/bin/env python3
"""
Generate reference data for PointNetSegment test
yanx27 structure: 9-dimensional input, 13 semantic classes
"""

import torch
import json
import numpy as np
from safetensors.torch import save_file

torch.manual_seed(42)
np.random.seed(42)

N = 16  # num_points
NUM_CLASSES = 13  # semantic segmentation classes
CHANNEL = 9  # input channels

# Generate 9-dim input: [N, 9] (x,y,z + RGB + normalized coords)
input_data = torch.randn(N, CHANNEL, dtype=torch.float32)

# Generate expected output: [N, 13] (semantic class scores per point)
expected = torch.randn(N, NUM_CLASSES, dtype=torch.float32)

# Generate dummy weights for all layers
# Format: transposed for Vulkan GEMM (input_dim, output_dim)
tensors = {
    'input_shape': torch.tensor([N], dtype=torch.float32),
    'input': input_data,
    'expected_output': expected
}

# Encoder weights (STN3d - 9x9 transform)
# STN MLP: 9 -> 64 -> 128 -> 1024
tensors['feat.stn.conv1.weight'] = torch.randn(CHANNEL, 64).contiguous()
tensors['feat.stn.conv1.bias'] = torch.randn(64).contiguous()
tensors['feat.stn.bn1.weight'] = torch.ones(64).contiguous()
tensors['feat.stn.bn1.bias'] = torch.zeros(64).contiguous()
tensors['feat.stn.bn1.running_mean'] = torch.zeros(64).contiguous()
tensors['feat.stn.bn1.running_var'] = torch.ones(64).contiguous()

tensors['feat.stn.conv2.weight'] = torch.randn(64, 128).contiguous()
tensors['feat.stn.conv2.bias'] = torch.randn(128).contiguous()
tensors['feat.stn.bn2.weight'] = torch.ones(128).contiguous()
tensors['feat.stn.bn2.bias'] = torch.zeros(128).contiguous()
tensors['feat.stn.bn2.running_mean'] = torch.zeros(128).contiguous()
tensors['feat.stn.bn2.running_var'] = torch.ones(128).contiguous()

tensors['feat.stn.conv3.weight'] = torch.randn(128, 1024).contiguous()
tensors['feat.stn.conv3.bias'] = torch.randn(1024).contiguous()
tensors['feat.stn.bn3.weight'] = torch.ones(1024).contiguous()
tensors['feat.stn.bn3.bias'] = torch.zeros(1024).contiguous()
tensors['feat.stn.bn3.running_mean'] = torch.zeros(1024).contiguous()
tensors['feat.stn.bn3.running_var'] = torch.ones(1024).contiguous()

# STN FC layers
tensors['feat.stn.fc1.weight'] = torch.randn(1024, 512).contiguous()
tensors['feat.stn.fc1.bias'] = torch.randn(512).contiguous()
tensors['feat.stn.bn4.weight'] = torch.ones(512).contiguous()
tensors['feat.stn.bn4.bias'] = torch.zeros(512).contiguous()
tensors['feat.stn.bn4.running_mean'] = torch.zeros(512).contiguous()
tensors['feat.stn.bn4.running_var'] = torch.ones(512).contiguous()

tensors['feat.stn.fc2.weight'] = torch.randn(512, 256).contiguous()
tensors['feat.stn.fc2.bias'] = torch.randn(256).contiguous()
tensors['feat.stn.bn5.weight'] = torch.ones(256).contiguous()
tensors['feat.stn.bn5.bias'] = torch.zeros(256).contiguous()
tensors['feat.stn.bn5.running_mean'] = torch.zeros(256).contiguous()
tensors['feat.stn.bn5.running_var'] = torch.ones(256).contiguous()

tensors['feat.stn.fc3.weight'] = torch.randn(256, CHANNEL * CHANNEL).contiguous()
tensors['feat.stn.fc3.bias'] = torch.randn(CHANNEL * CHANNEL).contiguous()

# Conv1: 9 -> 64
tensors['feat.conv1.mlp0.weight'] = torch.randn(CHANNEL, 64).contiguous()
tensors['feat.conv1.mlp0.bias'] = torch.randn(64).contiguous()
tensors['feat.conv1.mlp0.bn_weight'] = torch.ones(64).contiguous()
tensors['feat.conv1.mlp0.bn_bias'] = torch.zeros(64).contiguous()
tensors['feat.conv1.mlp0.bn_mean'] = torch.zeros(64).contiguous()
tensors['feat.conv1.mlp0.bn_var'] = torch.ones(64).contiguous()

# FSTN (64x64 transform)
tensors['feat.fstn.conv1.weight'] = torch.randn(64, 64).contiguous()
tensors['feat.fstn.conv1.bias'] = torch.randn(64).contiguous()
tensors['feat.fstn.bn1.weight'] = torch.ones(64).contiguous()
tensors['feat.fstn.bn1.bias'] = torch.zeros(64).contiguous()
tensors['feat.fstn.bn1.running_mean'] = torch.zeros(64).contiguous()
tensors['feat.fstn.bn1.running_var'] = torch.ones(64).contiguous()

tensors['feat.fstn.conv2.weight'] = torch.randn(64, 128).contiguous()
tensors['feat.fstn.conv2.bias'] = torch.randn(128).contiguous()
tensors['feat.fstn.bn2.weight'] = torch.ones(128).contiguous()
tensors['feat.fstn.bn2.bias'] = torch.zeros(128).contiguous()
tensors['feat.fstn.bn2.running_mean'] = torch.zeros(128).contiguous()
tensors['feat.fstn.bn2.running_var'] = torch.ones(128).contiguous()

tensors['feat.fstn.conv3.weight'] = torch.randn(128, 1024).contiguous()
tensors['feat.fstn.conv3.bias'] = torch.randn(1024).contiguous()
tensors['feat.fstn.bn3.weight'] = torch.ones(1024).contiguous()
tensors['feat.fstn.bn3.bias'] = torch.zeros(1024).contiguous()
tensors['feat.fstn.bn3.running_mean'] = torch.zeros(1024).contiguous()
tensors['feat.fstn.bn3.running_var'] = torch.ones(1024).contiguous()

tensors['feat.fstn.fc1.weight'] = torch.randn(1024, 512).contiguous()
tensors['feat.fstn.fc1.bias'] = torch.randn(512).contiguous()
tensors['feat.fstn.bn4.weight'] = torch.ones(512).contiguous()
tensors['feat.fstn.bn4.bias'] = torch.zeros(512).contiguous()
tensors['feat.fstn.bn4.running_mean'] = torch.zeros(512).contiguous()
tensors['feat.fstn.bn4.running_var'] = torch.ones(512).contiguous()

tensors['feat.fstn.fc2.weight'] = torch.randn(512, 256).contiguous()
tensors['feat.fstn.fc2.bias'] = torch.randn(256).contiguous()
tensors['feat.fstn.bn5.weight'] = torch.ones(256).contiguous()
tensors['feat.fstn.bn5.bias'] = torch.zeros(256).contiguous()
tensors['feat.fstn.bn5.running_mean'] = torch.zeros(256).contiguous()
tensors['feat.fstn.bn5.running_var'] = torch.ones(256).contiguous()

tensors['feat.fstn.fc3.weight'] = torch.randn(256, 64 * 64).contiguous()
tensors['feat.fstn.fc3.bias'] = torch.randn(64 * 64).contiguous()

# Conv2: 64 -> 128
tensors['feat.conv2.mlp0.weight'] = torch.randn(64, 128).contiguous()
tensors['feat.conv2.mlp0.bias'] = torch.randn(128).contiguous()
tensors['feat.conv2.mlp0.bn_weight'] = torch.ones(128).contiguous()
tensors['feat.conv2.mlp0.bn_bias'] = torch.zeros(128).contiguous()
tensors['feat.conv2.mlp0.bn_mean'] = torch.zeros(128).contiguous()
tensors['feat.conv2.mlp0.bn_var'] = torch.ones(128).contiguous()

# Conv3: 128 -> 1024 (no ReLU)
tensors['feat.conv3.weight'] = torch.randn(128, 1024).contiguous()
tensors['feat.conv3.bias'] = torch.randn(1024).contiguous()
tensors['feat.conv3.bn_weight'] = torch.ones(1024).contiguous()
tensors['feat.conv3.bn_bias'] = torch.zeros(1024).contiguous()
tensors['feat.conv3.bn_mean'] = torch.zeros(1024).contiguous()
tensors['feat.conv3.bn_var'] = torch.ones(1024).contiguous()

# Segmentation head: 1088 -> 512 -> 256 -> 128 -> 13
# Note: Keys map to segHead.mlp0-3.* internally via operator[]
tensors['conv1.weight'] = torch.randn(1088, 512).contiguous()
tensors['conv1.bias'] = torch.randn(512).contiguous()
tensors['conv1.bn_weight'] = torch.ones(512).contiguous()
tensors['conv1.bn_bias'] = torch.zeros(512).contiguous()
tensors['conv1.bn_mean'] = torch.zeros(512).contiguous()
tensors['conv1.bn_var'] = torch.ones(512).contiguous()

tensors['conv2.weight'] = torch.randn(512, 256).contiguous()
tensors['conv2.bias'] = torch.randn(256).contiguous()
tensors['conv2.bn_weight'] = torch.ones(256).contiguous()
tensors['conv2.bn_bias'] = torch.zeros(256).contiguous()
tensors['conv2.bn_mean'] = torch.zeros(256).contiguous()
tensors['conv2.bn_var'] = torch.ones(256).contiguous()

tensors['conv3.weight'] = torch.randn(256, 128).contiguous()
tensors['conv3.bias'] = torch.randn(128).contiguous()
tensors['conv3.bn_weight'] = torch.ones(128).contiguous()
tensors['conv3.bn_bias'] = torch.zeros(128).contiguous()
tensors['conv3.bn_mean'] = torch.zeros(128).contiguous()
tensors['conv3.bn_var'] = torch.ones(128).contiguous()

tensors['conv4.weight'] = torch.randn(128, NUM_CLASSES).contiguous()
tensors['conv4.bias'] = torch.randn(NUM_CLASSES).contiguous()
tensors['conv4.bn_weight'] = torch.ones(NUM_CLASSES).contiguous()
tensors['conv4.bn_bias'] = torch.zeros(NUM_CLASSES).contiguous()
tensors['conv4.bn_mean'] = torch.zeros(NUM_CLASSES).contiguous()
tensors['conv4.bn_var'] = torch.ones(NUM_CLASSES).contiguous()

save_file(tensors, 'test/segment/reference.safetensors')

print(f"✓ Reference data saved (9-dim input, 13 classes)")
print(f"  Input: [N={N}, {CHANNEL}]")
print(f"  Output: [N={N}, {NUM_CLASSES}]")
print(f"  Weights: {len(tensors) - 3} tensors")

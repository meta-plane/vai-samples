#!/usr/bin/env python3
"""
Generate reference data for PointNetSegment test.
Uses exact yanx27 structure with PyTorch state_dict keys preserved.

Output: reference.safetensors containing:
  - input: [C, N] tensor (C=9 for xyz+rgb+normalized)
  - expected_output: [num_classes, N] tensor
  - All model weights with PyTorch keys
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from safetensors.torch import save_file

torch.manual_seed(42)
np.random.seed(42)


class STN3d(nn.Module):
    """Spatial Transformer Network for 3D points - outputs 3x3 matrix"""
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(
            np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)
        )).view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    """Spatial Transformer Network for k-dim features"""
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(
            np.eye(self.k).flatten().astype(np.float32)
        )).view(1, self.k * self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    """PointNet Encoder"""
    def __init__(self, global_feat=False, feature_transform=True, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)

        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x  # [B, 64, N]

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetSegment(nn.Module):
    """PointNet Semantic Segmentation"""
    def __init__(self, num_classes=13, channel=9):
        super(PointNetSegment, self).__init__()
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=channel)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, num_classes, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)  # No activation for final layer
        return x, trans, trans_feat


# Test configuration
N = 16
channel = 9  # xyz + rgb + normalized coords
num_classes = 13
batch_size = 1

# Generate input: [B, C, N] format
input_data = torch.randn(batch_size, channel, N, dtype=torch.float32)

# Create model
model = PointNetSegment(num_classes=num_classes, channel=channel)
model.eval()

print("Generating reference for PointNetSegment (PyTorch keys preserved)...")
print(f"  Input shape (PyTorch): [B={batch_size}, C={channel}, N={N}]")
print(f"  Classes: {num_classes}")

# Run inference
with torch.no_grad():
    output, trans, trans_feat = model(input_data)
    print(f"  Output shape (PyTorch): {output.shape}")  # [1, 13, 16]

# Extract data for Vulkan (remove batch dimension)
input_vulkan = input_data[0].contiguous()   # [C, N]
output_vulkan = output[0].contiguous()       # [num_classes, N]

print(f"  Input (Vulkan): {input_vulkan.shape} - [C, N] layout")
print(f"  Output (Vulkan): {output_vulkan.shape} - [num_classes, N] layout")

# Save tensors with PyTorch keys
tensors = {
    'input': input_vulkan,
    'expected_output': output_vulkan,
}

# Save all model weights with PyTorch state_dict keys
print("\nExtracting weights (PyTorch keys)...")
for name, param in model.state_dict().items():
    if param.dim() == 3 and param.size(2) == 1:
        # Conv1d: [C_out, C_in, 1] -> [C_out, C_in]
        tensors[name] = param.squeeze(-1).contiguous()
        print(f"  {name}: {param.shape} -> {tensors[name].shape}")
    else:
        tensors[name] = param.contiguous()
        if 'weight' in name or 'bias' in name:
            print(f"  {name}: {param.shape}")

from pathlib import Path
output_dir = Path(__file__).parent
output_file = output_dir / 'reference.safetensors'
save_file(tensors, str(output_file))

print(f"\n✓ Reference saved to {output_file}")
print(f"  Total tensors: {len(tensors)}")

# Print first few values for verification
print(f"\nFirst 5 values of output [0:5, 0]:")
for i in range(min(5, num_classes)):
    print(f"  output[{i}, 0] = {output_vulkan[i, 0].item():.6f}")

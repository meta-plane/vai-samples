"""
Generate reference data for T-Net Transform Path test
Tests: MLP -> MaxPool -> FC -> Reshape -> AddIdentity
This is the path that generates the transformation matrix
"""

import torch
import torch.nn as nn
import json

# Fixed random seed for reproducibility
torch.manual_seed(42)

# Test parameters
N = 8   # Number of points
K = 3   # Input dimension

print(f"=== T-Net Transform Path Reference Generator ===")
print(f"N={N}, K={K}")

# Generate random input
input_data = torch.randn(N, K)
print(f"Input shape: {input_data.shape}")

# ============================================================
# MLP: K -> 64 -> 128 -> 1024
# ============================================================
print("\n=== Building MLP layers ===")

# MLP0: K -> 64
mlp0_weight = torch.randn(K, 64) * 0.1
mlp0_bias = torch.randn(64) * 0.1
mlp0_mean = torch.zeros(64)
mlp0_var = torch.ones(64)
mlp0_gamma = torch.ones(64)
mlp0_beta = torch.zeros(64)

x = input_data @ mlp0_weight + mlp0_bias
x = (x - mlp0_mean) / torch.sqrt(mlp0_var + 1e-5) * mlp0_gamma + mlp0_beta
x = torch.relu(x)
mlp0_out = x.clone()
print(f"MLP0 output: {x.shape}")

# MLP1: 64 -> 128
mlp1_weight = torch.randn(64, 128) * 0.1
mlp1_bias = torch.randn(128) * 0.1
mlp1_mean = torch.zeros(128)
mlp1_var = torch.ones(128)
mlp1_gamma = torch.ones(128)
mlp1_beta = torch.zeros(128)

x = x @ mlp1_weight + mlp1_bias
x = (x - mlp1_mean) / torch.sqrt(mlp1_var + 1e-5) * mlp1_gamma + mlp1_beta
x = torch.relu(x)
mlp1_out = x.clone()
print(f"MLP1 output: {x.shape}")

# MLP2: 128 -> 1024
mlp2_weight = torch.randn(128, 1024) * 0.1
mlp2_bias = torch.randn(1024) * 0.1
mlp2_mean = torch.zeros(1024)
mlp2_var = torch.ones(1024)
mlp2_gamma = torch.ones(1024)
mlp2_beta = torch.zeros(1024)

x = x @ mlp2_weight + mlp2_bias
x = (x - mlp2_mean) / torch.sqrt(mlp2_var + 1e-5) * mlp2_gamma + mlp2_beta
x = torch.relu(x)
mlp_out = x.clone()
print(f"MLP output: {x.shape}")

# ============================================================
# MaxPool: [N, 1024] -> [1024]
# ============================================================
print("\n=== MaxPool ===")
pooled = torch.max(x, dim=0)[0]  # [1024]
print(f"Pooled shape: {pooled.shape}")
print(f"Pooled (first 5): {pooled[:5].numpy()}")

# ============================================================
# FC: 1024 -> 512 -> 256 -> K*K
# ============================================================
print("\n=== Building FC layers ===")

# FC0: 1024 -> 512 (with BN + ReLU)
fc0_weight = torch.randn(1024, 512) * 0.1
fc0_bias = torch.randn(512) * 0.1
fc0_mean = torch.zeros(512)
fc0_var = torch.ones(512)
fc0_gamma = torch.ones(512)
fc0_beta = torch.zeros(512)

x = pooled @ fc0_weight + fc0_bias  # [512]
fc0_fc_only = x.clone()  # Save pure FC output (before BN+ReLU)
# For BatchNorm on 1D, reshape to [1, 512]
x_2d = x.unsqueeze(0)  # [1, 512]
x_2d = (x_2d - fc0_mean) / torch.sqrt(fc0_var + 1e-5) * fc0_gamma + fc0_beta
x = x_2d.squeeze(0)  # [512]
x = torch.relu(x)
fc0_out = x.clone()
print(f"FC0 output: {x.shape}")

# FC1: 512 -> 256 (with BN + ReLU)
fc1_weight = torch.randn(512, 256) * 0.1
fc1_bias = torch.randn(256) * 0.1
fc1_mean = torch.zeros(256)
fc1_var = torch.ones(256)
fc1_gamma = torch.ones(256)
fc1_beta = torch.zeros(256)

x = x @ fc1_weight + fc1_bias  # [256]
x_2d = x.unsqueeze(0)  # [1, 256]
x_2d = (x_2d - fc1_mean) / torch.sqrt(fc1_var + 1e-5) * fc1_gamma + fc1_beta
x = x_2d.squeeze(0)  # [256]
x = torch.relu(x)
fc1_out = x.clone()
print(f"FC1 output: {x.shape}")

# FC2: 256 -> K*K (no BN, no ReLU)
fc2_weight = torch.randn(256, K*K) * 0.1
fc2_bias = torch.randn(K*K) * 0.1

x = x @ fc2_weight + fc2_bias  # [K*K]
fc2_out = x.clone()
print(f"FC2 output: {x.shape}")
print(f"FC2 output values: {x.numpy()}")

# ============================================================
# Reshape: [K*K] -> [K, K]
# ============================================================
print("\n=== Reshape ===")
transform_no_id = x.view(K, K)
print(f"Transform (no identity):\n{transform_no_id.numpy()}")

# ============================================================
# AddIdentity: [K, K] + I -> [K, K]
# ============================================================
print("\n=== AddIdentity ===")
identity = torch.eye(K)
transform = transform_no_id + identity
print(f"Transform (with identity):\n{transform.numpy()}")

# ============================================================
# Save reference data
# ============================================================
reference = {
    'shape': [float(N), float(K)],
    'input': input_data.numpy().flatten().tolist(),
    
    # Expected outputs at each stage
    'expected_mlp_out': mlp_out.numpy().flatten().tolist(),
    'expected_pooled': pooled.numpy().flatten().tolist(),
    'expected_fc0_fc_only': fc0_fc_only.numpy().flatten().tolist(),  # Pure FC output
    'expected_fc0_out': fc0_out.numpy().flatten().tolist(),
    'expected_fc1_out': fc1_out.numpy().flatten().tolist(),
    'expected_fc2_out': fc2_out.numpy().flatten().tolist(),
    'expected_transform_no_id': transform_no_id.numpy().flatten().tolist(),
    'expected_transform': transform.numpy().flatten().tolist(),
    
    # MLP weights
    'mlp.mlp0.weight': mlp0_weight.numpy().flatten().tolist(),
    'mlp.mlp0.bias': mlp0_bias.numpy().flatten().tolist(),
    'mlp.mlp0.mean': mlp0_mean.numpy().flatten().tolist(),
    'mlp.mlp0.var': mlp0_var.numpy().flatten().tolist(),
    'mlp.mlp0.gamma': mlp0_gamma.numpy().flatten().tolist(),
    'mlp.mlp0.beta': mlp0_beta.numpy().flatten().tolist(),
    
    'mlp.mlp1.weight': mlp1_weight.numpy().flatten().tolist(),
    'mlp.mlp1.bias': mlp1_bias.numpy().flatten().tolist(),
    'mlp.mlp1.mean': mlp1_mean.numpy().flatten().tolist(),
    'mlp.mlp1.var': mlp1_var.numpy().flatten().tolist(),
    'mlp.mlp1.gamma': mlp1_gamma.numpy().flatten().tolist(),
    'mlp.mlp1.beta': mlp1_beta.numpy().flatten().tolist(),
    
    'mlp.mlp2.weight': mlp2_weight.numpy().flatten().tolist(),
    'mlp.mlp2.bias': mlp2_bias.numpy().flatten().tolist(),
    'mlp.mlp2.mean': mlp2_mean.numpy().flatten().tolist(),
    'mlp.mlp2.var': mlp2_var.numpy().flatten().tolist(),
    'mlp.mlp2.gamma': mlp2_gamma.numpy().flatten().tolist(),
    'mlp.mlp2.beta': mlp2_beta.numpy().flatten().tolist(),
    
    # FC weights
    'fc.block0.weight': fc0_weight.numpy().flatten().tolist(),
    'fc.block0.bias': fc0_bias.numpy().flatten().tolist(),
    'fc.block0.mean': fc0_mean.numpy().flatten().tolist(),
    'fc.block0.var': fc0_var.numpy().flatten().tolist(),
    'fc.block0.gamma': fc0_gamma.numpy().flatten().tolist(),
    'fc.block0.beta': fc0_beta.numpy().flatten().tolist(),
    
    'fc.block1.weight': fc1_weight.numpy().flatten().tolist(),
    'fc.block1.bias': fc1_bias.numpy().flatten().tolist(),
    'fc.block1.mean': fc1_mean.numpy().flatten().tolist(),
    'fc.block1.var': fc1_var.numpy().flatten().tolist(),
    'fc.block1.gamma': fc1_gamma.numpy().flatten().tolist(),
    'fc.block1.beta': fc1_beta.numpy().flatten().tolist(),
    
    'fc.lastBlock.weight': fc2_weight.numpy().flatten().tolist(),
    'fc.lastBlock.bias': fc2_bias.numpy().flatten().tolist(),
}

output_path = 'test/tnet_transform_path/reference.json'
import os
os.makedirs('test/tnet_transform_path', exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(reference, f, indent=2)

print(f"\n✓ Reference data saved to {output_path}")

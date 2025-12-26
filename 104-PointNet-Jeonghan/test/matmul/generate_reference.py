import torch
import json
import numpy as np

torch.manual_seed(42)

# Test configuration: Matrix multiplication in [C, N] layout
# For [C, N] layout: A[K, N] @ B[M, K]^T  is equivalent to  B[M, K] @ A[K, N]
# Result: C[M, N]
N, K, M = 8, 16, 12  # N: points, K: input dim, M: output dim

print(f"\nMatMul Test Configuration ([C, N] layout):")
print(f"  A: [{K}, {N}] - input features")
print(f"  B: [{M}, {K}] - transformation matrix")
print(f"  C = B @ A: [{M}, {N}] - transformed features")
print()

# Generate random matrices in [C, N] layout
A = torch.randn(K, N)  # [16, 8] - input features
B = torch.randn(M, K)  # [12, 16] - transformation

# Compute matmul: C[M, N] = B[M, K] @ A[K, N]
with torch.no_grad():
    C = torch.matmul(B, A)

print(f"A shape: {A.shape} -> {A.numel()} values")
print(f"B shape: {B.shape} -> {B.numel()} values")
print(f"C shape: {C.shape} -> {C.numel()} values")
print()

# Prepare JSON data
json_data = {
    "shape": [float(M), float(K), float(N)],  # [M, K, N] order
    "A": A.flatten().tolist(),
    "B": B.flatten().tolist(),
    "C": C.flatten().tolist(),
}

# Save to JSON
from pathlib import Path
output_dir = Path(__file__).parent
output_file = output_dir / "reference.json"
with open(output_file, 'w') as f:
    json.dump(json_data, f, indent=2)

print(f"MatMul Reference Generated (JSON)")
print(f"  A: [{K}, {N}] = {K * N} values [K, N]")
print(f"  B: [{M}, {K}] = {M * K} values [M, K]")
print(f"  C: [{M}, {N}] = {M * N} values [M, N]")
print()
print(f"✓ Saved to {output_file}")

# Save SafeTensors (preferred format)
from safetensors.torch import save_file
tensors = {
    "A": A.flatten().contiguous(),
    "B": B.flatten().contiguous(),
    "C": C.flatten().contiguous(),
    "shape": torch.tensor([M, K, N], dtype=torch.float32)
}
safetensors_file = output_dir / "reference.safetensors"
save_file(tensors, str(safetensors_file))
print(f"✓ Saved SafeTensors to {safetensors_file}")

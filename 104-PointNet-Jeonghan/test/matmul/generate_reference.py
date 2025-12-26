import torch
import json
import numpy as np

torch.manual_seed(42)

# Test configuration: Matrix multiplication
# A: [N, K], B: [K, M] -> C: [N, M]
N = 8   # Number of rows in A
K = 16  # Columns in A, Rows in B
M = 12  # Columns in B

print(f"\nMatMul Test Configuration:")
print(f"  A: [{N}, {K}]")
print(f"  B: [{K}, {M}]")
print(f"  C = A @ B: [{N}, {M}]")
print()

# Generate random matrices
A = torch.randn(N, K)
B = torch.randn(K, M)

# Compute matmul
with torch.no_grad():
    C = torch.matmul(A, B)

print(f"A shape: {A.shape} -> {A.numel()} values")
print(f"B shape: {B.shape} -> {B.numel()} values")
print(f"C shape: {C.shape} -> {C.numel()} values")
print()

# Prepare JSON data
json_data = {
    "shape": [float(N), float(K), float(M)],  # All dims in one array
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

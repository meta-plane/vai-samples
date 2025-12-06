"""
Generate test data for Linear layer using PyTorch GPU
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn
from json_exporter import export_test_data, set_seed

# Check CUDA
if not torch.cuda.is_available():
    print("ERROR: CUDA not available")
    sys.exit(1)

# Set seed for reproducibility
set_seed(42)

# Create test input
batch_size = 2
seq_len = 4
d_in = 768
d_out = 768

input_data = torch.randn(batch_size, seq_len, d_in, dtype=torch.float32) * 0.02

# Create Linear layer and move to GPU
linear = nn.Linear(d_in, d_out, bias=True)

# Initialize with small random weights
with torch.no_grad():
    linear.weight.data = torch.randn(d_out, d_in, dtype=torch.float32) * 0.02
    linear.bias.data = torch.randn(d_out, dtype=torch.float32) * 0.01

# Move to GPU
linear = linear.cuda()
input_gpu = input_data.cuda()

# Forward pass on GPU
with torch.no_grad():
    output_gpu = linear(input_gpu)

# Move back to CPU for export
output_data = output_gpu.cpu()
weight_data = linear.weight.cpu()
bias_data = linear.bias.cpu()

# Export test data
export_test_data(
    input_data=input_data,
    output_data=output_data,
    parameters={
        "weight": weight_data,
        "bias": bias_data
    },
    output_path="../../../assets/test_data/linear_test.json"
)

print("\nLinear test data generated with PyTorch GPU!")
print(f"Sample input:  {input_data.flatten()[:5].tolist()}")
print(f"Sample output: {output_data.flatten()[:5].tolist()}")

"""
Generate test data for Add (residual connection) using PyTorch GPU
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from json_exporter import export_test_data, set_seed

# Check CUDA
if not torch.cuda.is_available():
    print("ERROR: CUDA not available")
    sys.exit(1)

# Set seed for reproducibility
set_seed(42)

# Create test inputs
batch_size = 2
seq_len = 4
d_model = 768

input0 = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32) * 0.02
input1 = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32) * 0.02

# Move to GPU
input0_gpu = input0.cuda()
input1_gpu = input1.cuda()

# Element-wise addition on GPU
output_gpu = input0_gpu + input1_gpu

# Move back to CPU for export
output_data = output_gpu.cpu()

# Export test data
export_test_data(
    input_data=input0,
    output_data=output_data,
    parameters={
        "in1": input1
    },
    output_path="../../../assets/test_data/add_test.json"
)

print("\nAdd test data generated with PyTorch GPU!")
print(f"Sample input0: {input0.flatten()[:5].tolist()}")
print(f"Sample input1: {input1.flatten()[:5].tolist()}")
print(f"Sample output: {output_data.flatten()[:5].tolist()}")

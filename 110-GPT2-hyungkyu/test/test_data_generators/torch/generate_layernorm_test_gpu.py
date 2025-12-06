"""
Generate test data for LayerNorm using PyTorch GPU
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from torch_layers import LayerNorm
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
d_model = 768

input_data = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32) * 0.02

# Create LayerNorm and move to GPU
layernorm = LayerNorm(d_model)

# Initialize parameters
with torch.no_grad():
    layernorm.scale.data = torch.randn(d_model, dtype=torch.float32) * 0.02 + 1.0
    layernorm.shift.data = torch.randn(d_model, dtype=torch.float32) * 0.01

# Move to GPU
layernorm = layernorm.cuda()
input_gpu = input_data.cuda()

# Forward pass on GPU
with torch.no_grad():
    output_gpu = layernorm(input_gpu)

# Move back to CPU for export
output_data = output_gpu.cpu()
scale_data = layernorm.scale.cpu()
shift_data = layernorm.shift.cpu()

# Export test data
export_test_data(
    input_data=input_data,
    output_data=output_data,
    parameters={
        "scale": scale_data,
        "shift": shift_data
    },
    output_path="../../../assets/test_data/layernorm_test.json"
)

print("\nLayerNorm test data generated with PyTorch GPU!")
print(f"Sample input:  {input_data.flatten()[:5].tolist()}")
print(f"Sample output: {output_data.flatten()[:5].tolist()}")

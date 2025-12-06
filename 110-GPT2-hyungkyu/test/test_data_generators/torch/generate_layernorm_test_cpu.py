"""
Generate test data for LayerNorm using PyTorch
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from torch_layers import LayerNorm
from json_exporter import export_test_data, set_seed

# Set seed for reproducibility
set_seed(42)

# Create test input
batch_size = 2
seq_len = 4
d_model = 768

input_data = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32) * 0.02

# Create LayerNorm
layernorm = LayerNorm(d_model)

# Initialize parameters
with torch.no_grad():
    layernorm.scale.data = torch.randn(d_model, dtype=torch.float32) * 0.02 + 1.0
    layernorm.shift.data = torch.randn(d_model, dtype=torch.float32) * 0.01

# Forward pass
with torch.no_grad():
    output_data = layernorm(input_data)

# Export test data (use 'scale' and 'shift' to match LayerNormNode slot names)
export_test_data(
    input_data=input_data,
    output_data=output_data,
    parameters={
        "scale": layernorm.scale.data,
        "shift": layernorm.shift.data
    },
    output_path="../../../assets/test_data/layernorm_test.json"
)

print("\nLayerNorm test data generated successfully!")
print(f"Sample input:  {input_data.flatten()[:5].tolist()}")
print(f"Sample output: {output_data.flatten()[:5].tolist()}")

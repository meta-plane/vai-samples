"""
Generate test data for GELU activation using PyTorch CPU
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from torch_layers import GELU
from json_exporter import export_test_data, set_seed

# Set seed for reproducibility
set_seed(42)

# Create test input
batch_size = 2
seq_len = 3
d_model = 8

input_data = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)

# Create GELU layer
gelu = GELU()

# Forward pass
with torch.no_grad():
    output_data = gelu(input_data)

# Export test data
export_test_data(
    input_data=input_data,
    output_data=output_data,
    output_path="../../../assets/test_data/gelu_test.json"
)

print("\nGELU test data generated successfully!")
print(f"Sample input:  {input_data.flatten()[:5].tolist()}")
print(f"Sample output: {output_data.flatten()[:5].tolist()}")

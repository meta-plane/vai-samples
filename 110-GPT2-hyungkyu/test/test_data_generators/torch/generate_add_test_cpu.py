"""
Generate test data for Add (residual connection) using PyTorch
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from json_exporter import export_test_data, set_seed

# Set seed for reproducibility
set_seed(42)

# Create test inputs
batch_size = 2
seq_len = 4
d_model = 768

input0 = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32) * 0.02
input1 = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32) * 0.02

# Element-wise addition
output_data = input0 + input1

# Export test data (input0 is the main input, input1 is a parameter)
export_test_data(
    input_data=input0,
    output_data=output_data,
    parameters={
        "in1": input1  # Second input as parameter
    },
    output_path="../../../assets/test_data/add_test.json"
)

print("\nAdd test data generated successfully!")
print(f"Sample input0: {input0.flatten()[:5].tolist()}")
print(f"Sample input1: {input1.flatten()[:5].tolist()}")
print(f"Sample output: {output_data.flatten()[:5].tolist()}")

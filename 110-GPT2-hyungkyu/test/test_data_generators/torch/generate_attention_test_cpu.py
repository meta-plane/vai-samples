"""
Generate test data for MultiHeadAttention using PyTorch
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from torch_layers import MultiHeadAttention
from json_exporter import export_test_data, set_seed

# Set seed for reproducibility
set_seed(42)

# Create test input
batch_size = 1
seq_len = 4
d_model = 768
num_heads = 12

input_data = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32) * 0.02

# Create MultiHeadAttention
attn = MultiHeadAttention(
    d_in=d_model,
    d_out=d_model,
    context_length=seq_len,
    dropout=0.0,  # No dropout for testing
    num_heads=num_heads,
    qkv_bias=True
)

# Initialize parameters with small random weights
with torch.no_grad():
    attn.W_query.weight.data = torch.randn(d_model, d_model, dtype=torch.float32) * 0.02
    attn.W_query.bias.data = torch.randn(d_model, dtype=torch.float32) * 0.01
    attn.W_key.weight.data = torch.randn(d_model, d_model, dtype=torch.float32) * 0.02
    attn.W_key.bias.data = torch.randn(d_model, dtype=torch.float32) * 0.01
    attn.W_value.weight.data = torch.randn(d_model, d_model, dtype=torch.float32) * 0.02
    attn.W_value.bias.data = torch.randn(d_model, dtype=torch.float32) * 0.01
    attn.out_proj.weight.data = torch.randn(d_model, d_model, dtype=torch.float32) * 0.02
    attn.out_proj.bias.data = torch.randn(d_model, dtype=torch.float32) * 0.01

# Forward pass
attn.eval()  # Disable dropout
with torch.no_grad():
    output_data = attn(input_data)

# Export test data
export_test_data(
    input_data=input_data,
    output_data=output_data,
    parameters={
        "W_query": attn.W_query.weight.data,
        "B_query": attn.W_query.bias.data,
        "W_key": attn.W_key.weight.data,
        "B_key": attn.W_key.bias.data,
        "W_value": attn.W_value.weight.data,
        "B_value": attn.W_value.bias.data,
        "W_out": attn.out_proj.weight.data,
        "B_out": attn.out_proj.bias.data
    },
    output_path="../../../assets/test_data/attention_test.json"
)

print("\nMultiHeadAttention test data generated successfully!")
print(f"Number of heads: {num_heads}")
print(f"Sample input:  {input_data.flatten()[:5].tolist()}")
print(f"Sample output: {output_data.flatten()[:5].tolist()}")

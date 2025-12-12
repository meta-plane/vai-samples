"""
Generate sequence test data for MultiHeadAttention with KV cache simulation
This tests autoregressive token-by-token generation
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import json
import numpy as np
from torch_layers import MultiHeadAttention
from json_exporter import set_seed

# Set seed for reproducibility
set_seed(42)

# Configuration
batch_size = 1
num_tokens = 4  # Generate 4 tokens sequentially
d_model = 768
num_heads = 12
max_cache_length = 1024

# Create MultiHeadAttention
attn = MultiHeadAttention(
    d_in=d_model,
    d_out=d_model,
    context_length=max_cache_length,
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

# Generate random input tokens
all_tokens = torch.randn(batch_size, num_tokens, d_model, dtype=torch.float32) * 0.02

# Simulate autoregressive generation
# In real KV cache: process tokens one by one, reusing previous K, V
# In PyTorch without KV cache: process progressively longer sequences
attn.eval()
steps = []

for i in range(1, num_tokens + 1):
    # Process tokens [0:i]
    input_sequence = all_tokens[:, :i, :]

    with torch.no_grad():
        output_sequence = attn(input_sequence)

    # For sequence test: we want each step to test incremental processing
    # Step i processes token i-1 and should match output of full sequence at position i-1
    step_data = {
        "input": all_tokens[:, i-1:i, :].numpy().tolist(),  # Single token input
        "output": output_sequence[:, i-1:i, :].numpy().tolist(),  # Output for that token
        "cache_length": i - 1  # How many tokens are already in cache
    }
    steps.append(step_data)

# Prepare JSON output
test_data = {
    "mode": "sequence",
    "config": {
        "batch_size": batch_size,
        "d_model": d_model,
        "num_heads": num_heads,
        "num_steps": num_tokens,
        "use_kv_cache": True,
        "max_cache_length": max_cache_length
    },
    "parameters": {
        "W_query": attn.W_query.weight.data.numpy().tolist(),
        "B_query": attn.W_query.bias.data.numpy().tolist(),
        "W_key": attn.W_key.weight.data.numpy().tolist(),
        "B_key": attn.W_key.bias.data.numpy().tolist(),
        "W_value": attn.W_value.weight.data.numpy().tolist(),
        "B_value": attn.W_value.bias.data.numpy().tolist(),
        "W_out": attn.out_proj.weight.data.numpy().tolist(),
        "B_out": attn.out_proj.bias.data.numpy().tolist()
    },
    "steps": steps
}

# Save to JSON
output_path = "../../../assets/test_data/attention_sequence_test.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(test_data, f, indent=2)

print(f"\nAttention sequence test data generated successfully!")
print(f"Number of steps: {num_tokens}")
print(f"Number of heads: {num_heads}")
print(f"Saved to: {output_path}")
print(f"\nStep summary:")
for i, step in enumerate(steps):
    print(f"  Step {i+1}: cache_length={step['cache_length']}, "
          f"input_shape=[{batch_size}, 1, {d_model}], "
          f"output_shape=[{batch_size}, 1, {d_model}]")

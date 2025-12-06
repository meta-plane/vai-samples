"""
Generate test data for FeedForward (MLP) using PyTorch
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from torch_layers import FeedForward
from json_exporter import export_test_data, set_seed

# Set seed for reproducibility
set_seed(42)

# Create test input
batch_size = 2
seq_len = 4
d_model = 768

input_data = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32) * 0.02

# Create FeedForward
cfg = {"emb_dim": d_model}
ff = FeedForward(cfg)

# Initialize parameters with small random weights
with torch.no_grad():
    # First linear layer: d_model -> 4*d_model
    ff.layers[0].weight.data = torch.randn(4 * d_model, d_model, dtype=torch.float32) * 0.02
    ff.layers[0].bias.data = torch.randn(4 * d_model, dtype=torch.float32) * 0.01

    # Second linear layer: 4*d_model -> d_model
    ff.layers[2].weight.data = torch.randn(d_model, 4 * d_model, dtype=torch.float32) * 0.02
    ff.layers[2].bias.data = torch.randn(d_model, dtype=torch.float32) * 0.01

# Forward pass
with torch.no_grad():
    output_data = ff(input_data)

# Export test data
export_test_data(
    input_data=input_data,
    output_data=output_data,
    parameters={
        "weight1": ff.layers[0].weight.data,
        "bias1": ff.layers[0].bias.data,
        "weight2": ff.layers[2].weight.data,
        "bias2": ff.layers[2].bias.data
    },
    output_path="../../../assets/test_data/feedforward_test.json"
)

print("\nFeedForward test data generated successfully!")
print(f"Hidden dimension: {4 * d_model}")
print(f"Sample input:  {input_data.flatten()[:5].tolist()}")
print(f"Sample output: {output_data.flatten()[:5].tolist()}")

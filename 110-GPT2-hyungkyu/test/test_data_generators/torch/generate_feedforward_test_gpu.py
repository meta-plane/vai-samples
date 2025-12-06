"""
Generate test data for FeedForward (MLP) using PyTorch GPU
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from torch_layers import FeedForward
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

# Create FeedForward
cfg = {"emb_dim": d_model}
ff = FeedForward(cfg)

# Initialize parameters
with torch.no_grad():
    ff.layers[0].weight.data = torch.randn(4 * d_model, d_model, dtype=torch.float32) * 0.02
    ff.layers[0].bias.data = torch.randn(4 * d_model, dtype=torch.float32) * 0.01
    ff.layers[2].weight.data = torch.randn(d_model, 4 * d_model, dtype=torch.float32) * 0.02
    ff.layers[2].bias.data = torch.randn(d_model, dtype=torch.float32) * 0.01

# Move to GPU
ff = ff.cuda()
input_gpu = input_data.cuda()

# Forward pass on GPU
with torch.no_grad():
    output_gpu = ff(input_gpu)

# Move back to CPU for export
output_data = output_gpu.cpu()

# Export test data
export_test_data(
    input_data=input_data,
    output_data=output_data,
    parameters={
        "weight1": ff.layers[0].weight.cpu(),
        "bias1": ff.layers[0].bias.cpu(),
        "weight2": ff.layers[2].weight.cpu(),
        "bias2": ff.layers[2].bias.cpu()
    },
    output_path="../../../assets/test_data/feedforward_test.json"
)

print("\nFeedForward test data generated with PyTorch GPU!")
print(f"Hidden dimension: {4 * d_model}")
print(f"Sample input:  {input_data.flatten()[:5].tolist()}")
print(f"Sample output: {output_data.flatten()[:5].tolist()}")

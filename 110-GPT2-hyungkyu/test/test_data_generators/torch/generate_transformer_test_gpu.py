"""
Generate test data for TransformerBlock using PyTorch GPU
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from torch_layers import TransformerBlock
from json_exporter import export_test_data, set_seed

# Check CUDA
if not torch.cuda.is_available():
    print("ERROR: CUDA not available")
    sys.exit(1)

# Set seed for reproducibility
set_seed(42)

# Create test input
batch_size = 1
seq_len = 4
d_model = 768
num_heads = 12

input_data = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32) * 0.02

# Create TransformerBlock
cfg = {
    "emb_dim": d_model,
    "context_length": seq_len,
    "n_heads": num_heads,
    "drop_rate": 0.0,
    "qkv_bias": True
}
transformer = TransformerBlock(cfg)

# Initialize all parameters
with torch.no_grad():
    # Attention Q, K, V projections
    transformer.att.W_query.weight.data = torch.randn(d_model, d_model, dtype=torch.float32) * 0.02
    transformer.att.W_query.bias.data = torch.randn(d_model, dtype=torch.float32) * 0.01
    transformer.att.W_key.weight.data = torch.randn(d_model, d_model, dtype=torch.float32) * 0.02
    transformer.att.W_key.bias.data = torch.randn(d_model, dtype=torch.float32) * 0.01
    transformer.att.W_value.weight.data = torch.randn(d_model, d_model, dtype=torch.float32) * 0.02
    transformer.att.W_value.bias.data = torch.randn(d_model, dtype=torch.float32) * 0.01

    # Attention output projection
    transformer.att.out_proj.weight.data = torch.randn(d_model, d_model, dtype=torch.float32) * 0.02
    transformer.att.out_proj.bias.data = torch.randn(d_model, dtype=torch.float32) * 0.01

    # First LayerNorm
    transformer.norm1.scale.data = torch.randn(d_model, dtype=torch.float32) * 0.02 + 1.0
    transformer.norm1.shift.data = torch.randn(d_model, dtype=torch.float32) * 0.01

    # FeedForward layer 1 (expansion)
    transformer.ff.layers[0].weight.data = torch.randn(4 * d_model, d_model, dtype=torch.float32) * 0.02
    transformer.ff.layers[0].bias.data = torch.randn(4 * d_model, dtype=torch.float32) * 0.01

    # FeedForward layer 2 (projection)
    transformer.ff.layers[2].weight.data = torch.randn(d_model, 4 * d_model, dtype=torch.float32) * 0.02
    transformer.ff.layers[2].bias.data = torch.randn(d_model, dtype=torch.float32) * 0.01

    # Second LayerNorm
    transformer.norm2.scale.data = torch.randn(d_model, dtype=torch.float32) * 0.02 + 1.0
    transformer.norm2.shift.data = torch.randn(d_model, dtype=torch.float32) * 0.01

# Move to GPU
transformer = transformer.cuda()
transformer.eval()
input_gpu = input_data.cuda()

# Forward pass on GPU
with torch.no_grad():
    output_gpu = transformer(input_gpu)

# Move back to CPU for export
output_data = output_gpu.cpu()

# Export test data (16 parameters total)
# Use TransformerBlock's slot names
export_test_data(
    input_data=input_data,
    output_data=output_data,
    parameters={
        # First LayerNorm parameters (2)
        "norm1_scale": transformer.norm1.scale.cpu(),
        "norm1_shift": transformer.norm1.shift.cpu(),

        # Attention parameters (8)
        "attn_wq": transformer.att.W_query.weight.cpu(),
        "attn_bq": transformer.att.W_query.bias.cpu(),
        "attn_wk": transformer.att.W_key.weight.cpu(),
        "attn_bk": transformer.att.W_key.bias.cpu(),
        "attn_wv": transformer.att.W_value.weight.cpu(),
        "attn_bv": transformer.att.W_value.bias.cpu(),
        "attn_wout": transformer.att.out_proj.weight.cpu(),
        "attn_bout": transformer.att.out_proj.bias.cpu(),

        # Second LayerNorm parameters (2)
        "norm2_scale": transformer.norm2.scale.cpu(),
        "norm2_shift": transformer.norm2.shift.cpu(),

        # FeedForward parameters (4)
        "ff_w1": transformer.ff.layers[0].weight.cpu(),
        "ff_b1": transformer.ff.layers[0].bias.cpu(),
        "ff_w2": transformer.ff.layers[2].weight.cpu(),
        "ff_b2": transformer.ff.layers[2].bias.cpu()
    },
    output_path="../../../assets/test_data/transformer_test.json"
)

print("\nTransformerBlock test data generated with PyTorch GPU!")
print(f"Total parameters: 16 (8 attention + 2 norm1 + 4 feedforward + 2 norm2)")
print(f"Sample input:  {input_data.flatten()[:5].tolist()}")
print(f"Sample output: {output_data.flatten()[:5].tolist()}")

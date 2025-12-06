"""
Generate test data for TransformerBlock using PyTorch
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from torch_layers import TransformerBlock
from json_exporter import export_test_data, set_seed

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
    "drop_rate": 0.0,  # No dropout for testing
    "qkv_bias": True
}
transformer = TransformerBlock(cfg)

# Initialize all parameters with small random weights
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

# Forward pass
transformer.eval()  # Disable dropout
with torch.no_grad():
    output_data = transformer(input_data)

# Export test data (16 parameters total)
# Use TransformerBlock's slot names
export_test_data(
    input_data=input_data,
    output_data=output_data,
    parameters={
        # First LayerNorm parameters (2)
        "norm1_scale": transformer.norm1.scale.data,
        "norm1_shift": transformer.norm1.shift.data,

        # Attention parameters (8)
        "attn_wq": transformer.att.W_query.weight.data,
        "attn_bq": transformer.att.W_query.bias.data,
        "attn_wk": transformer.att.W_key.weight.data,
        "attn_bk": transformer.att.W_key.bias.data,
        "attn_wv": transformer.att.W_value.weight.data,
        "attn_bv": transformer.att.W_value.bias.data,
        "attn_wout": transformer.att.out_proj.weight.data,
        "attn_bout": transformer.att.out_proj.bias.data,

        # Second LayerNorm parameters (2)
        "norm2_scale": transformer.norm2.scale.data,
        "norm2_shift": transformer.norm2.shift.data,

        # FeedForward parameters (4)
        "ff_w1": transformer.ff.layers[0].weight.data,
        "ff_b1": transformer.ff.layers[0].bias.data,
        "ff_w2": transformer.ff.layers[2].weight.data,
        "ff_b2": transformer.ff.layers[2].bias.data
    },
    output_path="../../../assets/test_data/transformer_test.json"
)

print("\nTransformerBlock test data generated successfully!")
print(f"Total parameters: 16 (8 attention + 2 norm1 + 4 feedforward + 2 norm2)")
print(f"Sample input:  {input_data.flatten()[:5].tolist()}")
print(f"Sample output: {output_data.flatten()[:5].tolist()}")

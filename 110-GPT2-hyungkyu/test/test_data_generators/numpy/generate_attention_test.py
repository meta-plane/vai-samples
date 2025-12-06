"""
Generate MultiHeadAttentionNode test data using NumPy
Implements multi-head self-attention mechanism
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from json_exporter import export_test_data


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention
    Q, K, V: [batch, num_heads, seq_len, head_dim]
    """
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    attention_weights = softmax(scores, axis=-1)
    output = np.matmul(attention_weights, V)
    return output


def softmax(x, axis=-1):
    """Numerically stable softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def multi_head_attention(input_data, W_q, B_q, W_k, B_k, W_v, B_v, W_out, B_out, num_heads):
    """
    Multi-head self-attention
    input_data: [batch, seq_len, d_model]
    """
    batch_size, seq_len, d_model = input_data.shape
    head_dim = d_model // num_heads

    # Linear projections: [batch, seq_len, d_model]
    Q = input_data @ W_q.T + B_q
    K = input_data @ W_k.T + B_k
    V = input_data @ W_v.T + B_v

    # Reshape to [batch, num_heads, seq_len, head_dim]
    Q = Q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

    # Causal mask (for autoregressive generation)
    mask = np.triu(np.ones((seq_len, seq_len)) * -1e10, k=1)
    mask = mask[np.newaxis, np.newaxis, :, :]  # [1, 1, seq_len, seq_len]

    # Scaled dot-product attention
    attn_output = scaled_dot_product_attention(Q, K, V, mask)

    # Concatenate heads: [batch, seq_len, d_model]
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

    # Output projection
    output = attn_output @ W_out.T + B_out

    return output


def main():
    np.random.seed(42)

    # Configuration
    batch_size = 1
    seq_len = 4
    d_model = 768
    num_heads = 12
    head_dim = d_model // num_heads

    # Generate input data
    input_data = np.random.randn(batch_size, seq_len, d_model).astype(np.float32) * 0.02

    # Generate weight matrices [d_model, d_model]
    W_query = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    W_key = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    W_value = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    W_out = np.random.randn(d_model, d_model).astype(np.float32) * 0.02

    # Generate bias vectors [d_model]
    B_query = np.zeros(d_model, dtype=np.float32)
    B_key = np.zeros(d_model, dtype=np.float32)
    B_value = np.zeros(d_model, dtype=np.float32)
    B_out = np.zeros(d_model, dtype=np.float32)

    # Compute attention output
    output_data = multi_head_attention(
        input_data,
        W_query, B_query,
        W_key, B_key,
        W_value, B_value,
        W_out, B_out,
        num_heads
    )

    # Export test data with actual slot names
    output_path = "../../assets/test_data/attention_test.json"
    export_test_data(
        output_path=output_path,
        input_data=input_data,
        output_data=output_data,
        parameters={
            "W_query": W_query,
            "B_query": B_query,
            "W_key": W_key,
            "B_key": B_key,
            "W_value": W_value,
            "B_value": B_value,
            "W_out": W_out,
            "B_out": B_out
        }
    )

    print(f"\nMultiHeadAttentionNode configuration:")
    print(f"  d_in (d_model):   {d_model}")
    print(f"  d_out (d_model):  {d_model}")
    print(f"  num_heads:        {num_heads}")
    print(f"  head_dim:         {head_dim}")
    print(f"  Input shape:      {input_data.shape}")
    print(f"  Output shape:     {output_data.shape}")
    print(f"  Weight shape:     {W_query.shape}")
    print(f"  Bias shape:       {B_query.shape}")


if __name__ == "__main__":
    main()

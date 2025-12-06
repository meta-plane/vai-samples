"""
Generate TransformerBlock test data using NumPy
Implements full Transformer block: x = x + Attention(LayerNorm(x))
                                    x = x + FeedForward(LayerNorm(x))
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from json_exporter import export_test_data


def layer_norm(x, gamma, beta, eps=1e-5):
    """LayerNorm: normalize over the last dimension"""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


def gelu(x):
    """GELU activation function"""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x, axis=-1):
    """Numerically stable softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled dot-product attention"""
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    attention_weights = softmax(scores, axis=-1)
    output = np.matmul(attention_weights, V)
    return output


def multi_head_attention(input_data, W_q, B_q, W_k, B_k, W_v, B_v, W_out, B_out, num_heads):
    """Multi-head self-attention"""
    batch_size, seq_len, d_model = input_data.shape
    head_dim = d_model // num_heads

    # Linear projections
    Q = input_data @ W_q.T + B_q
    K = input_data @ W_k.T + B_k
    V = input_data @ W_v.T + B_v

    # Reshape to [batch, num_heads, seq_len, head_dim]
    Q = Q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

    # Causal mask
    mask = np.triu(np.ones((seq_len, seq_len)) * -1e10, k=1)
    mask = mask[np.newaxis, np.newaxis, :, :]

    # Attention
    attn_output = scaled_dot_product_attention(Q, K, V, mask)

    # Concatenate heads
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

    # Output projection
    output = attn_output @ W_out.T + B_out

    return output


def feed_forward(input_data, weight1, bias1, weight2, bias2):
    """Feed-forward network: Linear → GELU → Linear"""
    hidden = input_data @ weight1.T + bias1
    gelu_out = gelu(hidden)
    output = gelu_out @ weight2.T + bias2
    return output


def transformer_block(x,
                     norm1_scale, norm1_shift,
                     W_q, B_q, W_k, B_k, W_v, B_v, W_out, B_out, num_heads,
                     norm2_scale, norm2_shift,
                     ff_w1, ff_b1, ff_w2, ff_b2):
    """
    Full Transformer block with Pre-LayerNorm architecture
    x = x + Attention(LayerNorm(x))
    x = x + FeedForward(LayerNorm(x))
    """
    # First sub-block: LayerNorm -> Attention -> Residual
    normed1 = layer_norm(x, norm1_scale, norm1_shift)
    attn_out = multi_head_attention(normed1, W_q, B_q, W_k, B_k, W_v, B_v, W_out, B_out, num_heads)
    x = x + attn_out  # First residual connection

    # Second sub-block: LayerNorm -> FeedForward -> Residual
    normed2 = layer_norm(x, norm2_scale, norm2_shift)
    ff_out = feed_forward(normed2, ff_w1, ff_b1, ff_w2, ff_b2)
    x = x + ff_out  # Second residual connection

    return x


def main():
    np.random.seed(42)

    # Configuration
    batch_size = 1
    seq_len = 4
    d_model = 768
    num_heads = 12
    hidden_dim = 4 * d_model  # 3072

    # Generate input data
    input_data = np.random.randn(batch_size, seq_len, d_model).astype(np.float32) * 0.02

    # LayerNorm 1 parameters
    norm1_scale = np.ones(d_model, dtype=np.float32)
    norm1_shift = np.zeros(d_model, dtype=np.float32)

    # Attention parameters
    W_query = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    W_key = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    W_value = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    W_out = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    B_query = np.zeros(d_model, dtype=np.float32)
    B_key = np.zeros(d_model, dtype=np.float32)
    B_value = np.zeros(d_model, dtype=np.float32)
    B_out = np.zeros(d_model, dtype=np.float32)

    # LayerNorm 2 parameters
    norm2_scale = np.ones(d_model, dtype=np.float32)
    norm2_shift = np.zeros(d_model, dtype=np.float32)

    # FeedForward parameters
    ff_w1 = np.random.randn(hidden_dim, d_model).astype(np.float32) * 0.02
    ff_b1 = np.zeros(hidden_dim, dtype=np.float32)
    ff_w2 = np.random.randn(d_model, hidden_dim).astype(np.float32) * 0.02
    ff_b2 = np.zeros(d_model, dtype=np.float32)

    # Compute transformer block output
    output_data = transformer_block(
        input_data,
        norm1_scale, norm1_shift,
        W_query, B_query, W_key, B_key, W_value, B_value, W_out, B_out, num_heads,
        norm2_scale, norm2_shift,
        ff_w1, ff_b1, ff_w2, ff_b2
    )

    # Export test data with TransformerBlock's parameter naming
    output_path = "../../assets/test_data/transformer_test.json"
    export_test_data(
        output_path=output_path,
        input_data=input_data,
        output_data=output_data,
        parameters={
            "norm1_scale": norm1_scale,
            "norm1_shift": norm1_shift,
            "attn_wq": W_query,
            "attn_bq": B_query,
            "attn_wk": W_key,
            "attn_bk": B_key,
            "attn_wv": W_value,
            "attn_bv": B_value,
            "attn_wout": W_out,
            "attn_bout": B_out,
            "norm2_scale": norm2_scale,
            "norm2_shift": norm2_shift,
            "ff_w1": ff_w1,
            "ff_b1": ff_b1,
            "ff_w2": ff_w2,
            "ff_b2": ff_b2
        }
    )

    print(f"\nTransformerBlock configuration:")
    print(f"  d_model:          {d_model}")
    print(f"  num_heads:        {num_heads}")
    print(f"  hidden_dim:       {hidden_dim} (4 * d_model)")
    print(f"  Input shape:      {input_data.shape}")
    print(f"  Output shape:     {output_data.shape}")
    print(f"  Architecture:     Pre-LayerNorm Transformer Block")
    print(f"                    x = x + Attention(LayerNorm(x))")
    print(f"                    x = x + FeedForward(LayerNorm(x))")
    print(f"  Parameters:       16 tensors (2 LayerNorms + 1 Attention + 1 FeedForward)")


if __name__ == "__main__":
    main()

"""
Generate FeedForwardNode test data using NumPy
Implements MLP: Linear → GELU → Linear
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from json_exporter import export_test_data


def gelu(x):
    """
    GELU activation function
    """
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def feed_forward(input_data, weight1, bias1, weight2, bias2):
    """
    Feed-forward network: Linear → GELU → Linear
    input_data: [batch, seq_len, d_model]
    weight1: [hidden_dim, d_model] for Linear1
    bias1: [hidden_dim]
    weight2: [d_model, hidden_dim] for Linear2
    bias2: [d_model]
    """
    # Linear1: [batch, seq_len, d_model] @ [d_model, hidden_dim] = [batch, seq_len, hidden_dim]
    hidden = input_data @ weight1.T + bias1

    # GELU activation
    gelu_out = gelu(hidden)

    # Linear2: [batch, seq_len, hidden_dim] @ [hidden_dim, d_model] = [batch, seq_len, d_model]
    output = gelu_out @ weight2.T + bias2

    return output


def main():
    np.random.seed(42)

    # Configuration
    batch_size = 2
    seq_len = 4
    d_model = 768
    hidden_dim = 4 * d_model  # 3072

    # Generate input data
    input_data = np.random.randn(batch_size, seq_len, d_model).astype(np.float32) * 0.02

    # Generate weight matrices
    # weight1: [hidden_dim, d_model] for d_model -> hidden_dim expansion
    weight1 = np.random.randn(hidden_dim, d_model).astype(np.float32) * 0.02
    bias1 = np.zeros(hidden_dim, dtype=np.float32)

    # weight2: [d_model, hidden_dim] for hidden_dim -> d_model projection
    weight2 = np.random.randn(d_model, hidden_dim).astype(np.float32) * 0.02
    bias2 = np.zeros(d_model, dtype=np.float32)

    # Compute feed-forward output
    output_data = feed_forward(input_data, weight1, bias1, weight2, bias2)

    # Export test data with actual slot names
    output_path = "../../assets/test_data/feedforward_test.json"
    export_test_data(
        output_path=output_path,
        input_data=input_data,
        output_data=output_data,
        parameters={
            "weight1": weight1,
            "bias1": bias1,
            "weight2": weight2,
            "bias2": bias2
        }
    )

    print(f"\nFeedForwardNode configuration:")
    print(f"  d_model:          {d_model}")
    print(f"  hidden_dim:       {hidden_dim} (4 * d_model)")
    print(f"  Input shape:      {input_data.shape}")
    print(f"  Output shape:     {output_data.shape}")
    print(f"  Weight1 shape:    {weight1.shape} ({d_model} -> {hidden_dim})")
    print(f"  Bias1 shape:      {bias1.shape}")
    print(f"  Weight2 shape:    {weight2.shape} ({hidden_dim} -> {d_model})")
    print(f"  Bias2 shape:      {bias2.shape}")
    print(f"  Architecture:     Linear1 → GELU → Linear2")


if __name__ == "__main__":
    main()

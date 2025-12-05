"""
Generate LayerNorm test data using NumPy
"""
import sys
import os
import numpy as np

# Add current directory to path to import json_exporter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from json_exporter import export_test_data


def layer_norm(x, gamma, beta, eps=1e-5):
    """
    LayerNorm: normalize over the last dimension
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


def main():
    np.random.seed(42)

    # Test configuration
    batch_size = 2
    seq_len = 4
    normalized_shape = 768
    eps = 1e-5

    # Generate input data
    input_data = np.random.randn(batch_size, seq_len, normalized_shape).astype(np.float32)

    # Generate gamma (weight) and beta (bias)
    gamma = np.ones(normalized_shape, dtype=np.float32)
    beta = np.zeros(normalized_shape, dtype=np.float32)

    # Compute LayerNorm output
    output_data = layer_norm(input_data, gamma, beta, eps)

    # Export test data
    output_path = "../../assets/test_data/layernorm_test.json"
    export_test_data(
        output_path=output_path,
        input_data=input_data,
        output_data=output_data,
        parameters={
            "weight": gamma,
            "bias": beta
        }
    )

    print(f"\nLayerNorm configuration:")
    print(f"  normalized_shape: {normalized_shape}")
    print(f"  eps:              {eps}")
    print(f"  Input shape:      {input_data.shape}")
    print(f"  Output shape:     {output_data.shape}")
    print(f"  Weight shape:     {gamma.shape}")
    print(f"  Bias shape:       {beta.shape}")


if __name__ == "__main__":
    main()

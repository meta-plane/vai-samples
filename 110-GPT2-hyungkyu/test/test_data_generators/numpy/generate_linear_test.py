"""
Generate Linear layer test data using NumPy
"""
import sys
import os
import numpy as np

# Add current directory to path to import json_exporter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from json_exporter import export_test_data


def main():
    np.random.seed(42)

    # Test configuration
    batch_size = 1
    seq_len = 4
    in_features = 768
    out_features = 3072

    # Generate input data
    input_data = np.random.randn(batch_size, seq_len, in_features).astype(np.float32)

    # Generate weight and bias
    # Weight shape: [out_features, in_features] for GPU (stored as transposed)
    weight = np.random.randn(out_features, in_features).astype(np.float32) * 0.02
    bias = np.zeros(out_features, dtype=np.float32)

    # Compute output: Y = X @ W^T + b
    output_data = input_data @ weight.T + bias

    # Export test data
    output_path = "../../assets/test_data/linear_test.json"
    export_test_data(
        output_path=output_path,
        input_data=input_data,
        output_data=output_data,
        parameters={
            "weight": weight,
            "bias": bias
        }
    )

    print(f"\nLinear layer configuration:")
    print(f"  in_features:  {in_features}")
    print(f"  out_features: {out_features}")
    print(f"  Input shape:  {input_data.shape}")
    print(f"  Output shape: {output_data.shape}")
    print(f"  Weight shape: {weight.shape}")
    print(f"  Bias shape:   {bias.shape}")


if __name__ == "__main__":
    main()

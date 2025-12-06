"""
Generate GELU test data using the json_exporter utility
"""
import sys
import os
import numpy as np

# Add current directory to path to import json_exporter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from json_exporter import export_test_data


def gelu(x):
    """
    GELU activation function
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    sqrt_2_over_pi = 0.7978845608
    coeff = 0.044715
    inner = sqrt_2_over_pi * (x + coeff * x * x * x)
    return 0.5 * x * (1.0 + np.tanh(inner))


def main():
    np.random.seed(42)

    # Test configuration
    batch_size = 2
    seq_len = 3
    d_model = 8

    # Generate input data
    input_data = np.random.randn(batch_size, seq_len, d_model).astype(np.float32) * 2.0

    # Compute GELU
    output_data = gelu(input_data)

    # Export using the standard utility
    output_path = "../../assets/test_data/gelu_test.json"
    export_test_data(
        output_path=output_path,
        input_data=input_data,
        output_data=output_data
        # GELU has no parameters - omit parameters argument
    )

    print(f"\nSample values:")
    print(f"  Input[0]:  {input_data.flatten()[:5]}")
    print(f"  Output[0]: {output_data.flatten()[:5]}")


if __name__ == "__main__":
    main()

"""
Generate AddNode test data using NumPy
AddNode performs element-wise addition: output = in0 + in1
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
    batch_size = 2
    seq_len = 4
    d_model = 768

    # Generate two input tensors with same shape
    input0 = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
    input1 = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)

    # Element-wise addition
    output_data = input0 + input1

    # For AddNode, we need to store both inputs
    # We'll use input for in0 and parameters for in1
    output_path = "../../assets/test_data/add_test.json"

    # Export with input as in0, and in1 as a "parameter"
    export_test_data(
        output_path=output_path,
        input_data=input0,
        output_data=output_data,
        parameters={
            "in1": input1  # Second input stored as parameter
        }
    )

    print(f"\nAddNode configuration:")
    print(f"  Input shape (in0): {input0.shape}")
    print(f"  Input shape (in1): {input1.shape}")
    print(f"  Output shape:      {output_data.shape}")
    print(f"  Operation:         out = in0 + in1")


if __name__ == "__main__":
    main()

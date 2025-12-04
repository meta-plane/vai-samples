import json
import numpy as np
import os
from pathlib import Path

# Test configuration
batch_size = 2
seq_len = 3
in_features = 4
out_features = 5

# Get the project root directory (2 levels up from this script)
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
output_dir = project_root / "assets" / "test_data"

# Generate input data - simple pattern for verification
input_data = np.zeros((batch_size, seq_len, in_features), dtype=np.float32)
for b in range(batch_size):
    for s in range(seq_len):
        for i in range(in_features):
            input_data[b, s, i] = b * 100.0 + s * 10.0 + i * 1.0

# Generate weight data - simple pattern
weight_data = np.zeros((out_features, in_features), dtype=np.float32)
for o in range(out_features):
    for i in range(in_features):
        weight_data[o, i] = (o * 10.0 + i) * 0.1

# Compute expected output: Y = X @ W^T
# input: [B, S, in_features] = [2, 3, 4]
# weight: [out_features, in_features] = [5, 4]
# output: [B, S, out_features] = [2, 3, 5]
output_data = np.matmul(input_data, weight_data.T)

# Create test data dictionary
test_data = {
    "config": {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "in_features": in_features,
        "out_features": out_features
    },
    "input": input_data.tolist(),
    "weight": weight_data.tolist(),
    "expected_output": output_data.tolist()
}

# Ensure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

# Save to JSON file
output_file = output_dir / "linear_test_data.json"
with open(output_file, 'w') as f:
    json.dump(test_data, f, indent=2)

print(f"Test data saved to {output_file.absolute()}")
print(f"\nInput shape: {input_data.shape}")
print(f"Weight shape: {weight_data.shape}")
print(f"Output shape: {output_data.shape}")
print(f"\nSample input[0,0,:]: {input_data[0,0,:]}")
print(f"Sample weight[0,:]: {weight_data[0,:]}")
print(f"Sample output[0,0,:]: {output_data[0,0,:]}")

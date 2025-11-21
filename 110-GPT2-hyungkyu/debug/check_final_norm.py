"""
Check Final LayerNorm weights
"""
import numpy as np
import struct

# Read binary weights file
weights_file = "assets/weights/124M/gpt2_weights.bin"

with open(weights_file, "rb") as f:
    # First 4 bytes: num_tensors
    num_tensors = struct.unpack('i', f.read(4))[0]
    print(f"Number of tensors: {num_tensors}\n")

    # Find final_norm tensors
    for i in range(num_tensors):
        # Read name length
        name_len = struct.unpack('i', f.read(4))[0]
        # Read name
        name = f.read(name_len).decode('utf-8')
        # Read ndim
        ndim = struct.unpack('i', f.read(4))[0]
        # Read shape
        shape = list(struct.unpack(f'{ndim}i', f.read(4 * ndim)))
        # Read data size
        data_size = np.prod(shape)

        if "final_norm" in name:
            print(f"Found tensor: {name}")
            print(f"  Shape: {shape}")

            # Read data
            data = struct.unpack(f'{data_size}f', f.read(4 * data_size))

            print(f"  Min: {min(data):.6f}")
            print(f"  Max: {max(data):.6f}")
            print(f"  Mean: {np.mean(data):.6f}")
            print(f"  Std: {np.std(data):.6f}")
            print(f"  First 10 values: {data[:10]}")
            print()
        else:
            # Skip data
            f.seek(4 * data_size, 1)

print("=" * 60)
print("Final LayerNorm should have:")
print("  scale: all values close to 1.0")
print("  shift: all values close to 0.0")
print("=" * 60)

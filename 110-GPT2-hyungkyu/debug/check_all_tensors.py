"""
List all tensors in the weights file to find LM head
"""
import numpy as np
import struct

# Read binary weights file
weights_file = "assets/weights/124M/gpt2_weights.bin"

with open(weights_file, "rb") as f:
    # First 4 bytes: num_tensors
    num_tensors = struct.unpack('i', f.read(4))[0]
    print(f"Total tensors: {num_tensors}\n")

    print("All tensor names:")
    print("=" * 60)

    # List all tensor names
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

        # Skip data
        f.seek(4 * data_size, 1)

        # Print name and shape
        print(f"{i+1:3d}. {name:50s} {str(shape):20s}")

"""
Check if GPT-2 weights are properly loaded by comparing first few values
with PyTorch model
"""
import numpy as np
import struct

# Read binary weights file
weights_file = "assets/weights/124M/gpt2_weights.bin"

with open(weights_file, "rb") as f:
    # Read token embedding (wte)
    # First 4 bytes: num_tensors (should be 197)
    num_tensors = struct.unpack('i', f.read(4))[0]
    print(f"Number of tensors: {num_tensors}")

    # Find wte tensor
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

        if name == "wte":
            print(f"\nFound {name}: shape={shape}")
            vocab_size, d_model = shape

            # Read only first token (token 0)
            print(f"\nReading first 10 values of token 0:")
            token0_data = struct.unpack(f'{d_model}f', f.read(4 * d_model))
            print(token0_data[:10])

            # Skip to token 284
            tokens_to_skip = 283
            f.seek(4 * d_model * tokens_to_skip, 1)

            # Read token 284
            print(f"\nReading first 10 values of token 284 (' to'):")
            token284_data = struct.unpack(f'{d_model}f', f.read(4 * d_model))
            print(token284_data[:10])

            # Check statistics of what we read
            all_values = list(token0_data) + list(token284_data)
            print(f"\nStatistics (tokens 0 and 284):")
            print(f"  Mean: {np.mean(all_values):.6f}")
            print(f"  Std: {np.std(all_values):.6f}")
            print(f"  Min: {np.min(all_values):.6f}")
            print(f"  Max: {np.max(all_values):.6f}")
            break
        else:
            # Skip data
            f.seek(4 * data_size, 1)

print("\n" + "="*60)
print("If these values look normal (not all zeros or very large),")
print("then weight loading is probably correct.")
print("="*60)

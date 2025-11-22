"""
Check the shape of positional embedding in weights file
"""
import struct
import sys

weights_file = "110-GPT2-hyungkyu/assets/weights/124M/gpt2_weights.bin"

try:
    with open(weights_file, "rb") as f:
        # Read number of tensors
        num_tensors = struct.unpack('i', f.read(4))[0]
        print(f"Number of tensors: {num_tensors}")

        # Find wpe tensor (positional embedding)
        for i in range(num_tensors):
            # Read name length
            name_len = struct.unpack('i', f.read(4))[0]
            # Read name
            name = f.read(name_len).decode('utf-8')
            # Read ndim
            ndim = struct.unpack('i', f.read(4))[0]
            # Read shape
            shape = list(struct.unpack(f'{ndim}i', f.read(4 * ndim)))
            # Calculate data size
            data_size = 1
            for dim in shape:
                data_size *= dim

            print(f"Tensor {i}: {name}, shape={shape}")

            if name == "wpe":
                print(f"\n*** POSITIONAL EMBEDDING FOUND ***")
                print(f"Shape: {shape}")
                print(f"Max position: {shape[0] - 1}")
                print(f"Embedding dim: {shape[1]}")

                # Read first position and last position
                pos0 = struct.unpack(f'{shape[1]}f', f.read(4 * shape[1]))
                print(f"\nFirst 5 values of position 0: {pos0[:5]}")

                # Skip to last position
                positions_to_skip = shape[0] - 2
                f.seek(4 * shape[1] * positions_to_skip, 1)

                pos_last = struct.unpack(f'{shape[1]}f', f.read(4 * shape[1]))
                print(f"First 5 values of position {shape[0]-1}: {pos_last[:5]}")

                break
            else:
                # Skip data
                f.seek(4 * data_size, 1)

except FileNotFoundError:
    print(f"ERROR: File not found: {weights_file}")
    sys.exit(1)

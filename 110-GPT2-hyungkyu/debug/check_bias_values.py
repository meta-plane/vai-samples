"""
Check bias values to see if they are significant
"""
import numpy as np
import struct

# Read binary weights file
weights_file = "assets/weights/124M/gpt2_weights.bin"

with open(weights_file, "rb") as f:
    # First 4 bytes: num_tensors
    num_tensors = struct.unpack('i', f.read(4))[0]

    print("Checking bias values:")
    print("=" * 80)

    bias_stats = []

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

        if "bias" in name:
            # Read data
            data = struct.unpack(f'{data_size}f', f.read(4 * data_size))

            min_val = min(data)
            max_val = max(data)
            mean_val = np.mean(data)
            abs_max = max(abs(min_val), abs(max_val))

            bias_stats.append((name, abs_max, mean_val))

            # Print only first few layers
            if "h.0." in name or "h.1." in name or "h.11." in name:
                print(f"{name:50s} | abs_max={abs_max:8.4f} | mean={mean_val:8.4f}")
        else:
            # Skip data
            f.seek(4 * data_size, 1)

    print("=" * 80)
    print("\nSummary:")
    print(f"Total bias tensors: {len(bias_stats)}")

    # Find max abs bias
    max_bias = max(bias_stats, key=lambda x: x[1])
    print(f"Largest abs bias: {max_bias[0]} = {max_bias[1]:.4f}")

    # Calculate average abs_max
    avg_abs_max = np.mean([x[1] for x in bias_stats])
    print(f"Average abs_max across all biases: {avg_abs_max:.4f}")

    if avg_abs_max > 0.1:
        print("\n✗ Biases are SIGNIFICANT and cannot be ignored!")
        print("  They must be added to the model.")
    else:
        print("\n✓ Biases are small and may be negligible.")

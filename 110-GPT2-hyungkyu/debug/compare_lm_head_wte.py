"""
Compare lm_head.weight and wte to see if they are the same
"""
import numpy as np
import struct

# Read binary weights file
weights_file = "assets/weights/124M/gpt2_weights.bin"

lm_head_weight = None
wte = None

with open(weights_file, "rb") as f:
    # First 4 bytes: num_tensors
    num_tensors = struct.unpack('i', f.read(4))[0]

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

        if name == "lm_head.weight":
            print(f"Reading lm_head.weight...")
            # Read only first 1000 values to save memory
            sample_size = min(1000, data_size)
            lm_head_weight = struct.unpack(f'{sample_size}f', f.read(4 * sample_size))
            # Skip rest
            f.seek(4 * (data_size - sample_size), 1)
            print(f"  First 10 values: {lm_head_weight[:10]}")

        elif name == "wte":
            print(f"Reading wte...")
            # Read only first 1000 values to save memory
            sample_size = min(1000, data_size)
            wte = struct.unpack(f'{sample_size}f', f.read(4 * sample_size))
            # Skip rest
            f.seek(4 * (data_size - sample_size), 1)
            print(f"  First 10 values: {wte[:10]}")
        else:
            # Skip data
            f.seek(4 * data_size, 1)

print("\n" + "=" * 60)
if lm_head_weight and wte:
    # Compare first 1000 values
    diff = [abs(a - b) for a, b in zip(lm_head_weight, wte)]
    max_diff = max(diff)
    mean_diff = np.mean(diff)

    print(f"Comparison (first 1000 values):")
    print(f"  Max difference: {max_diff:.10f}")
    print(f"  Mean difference: {mean_diff:.10f}")

    if max_diff < 1e-6:
        print("\n✓ lm_head.weight and wte are IDENTICAL")
        print("  Weight tying is correct!")
    else:
        print("\n✗ lm_head.weight and wte are DIFFERENT!")
        print("  We should use lm_head.weight instead of wte!")
else:
    print("Error: Could not read both tensors")
print("=" * 60)

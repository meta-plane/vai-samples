#!/usr/bin/env python3
"""
Convert PyTorch PointNet weights to SafeTensors format.

This script performs minimal transformation - only squeezing Conv1d weights.
PyTorch state_dict keys are preserved as-is for direct use by C++ WeightLoader.

Usage:
    python convert_to_safetensors.py checkpoint.pth [output.safetensors]

    # From project root with default output
    python utils/convert_to_safetensors.py weights/best_model.pth
"""

import torch
import sys
from pathlib import Path

try:
    from safetensors.torch import save_file
except ImportError:
    print("Error: safetensors package required")
    print("Install: pip install safetensors")
    sys.exit(1)


def convert(checkpoint_path: str, output_path: str = None):
    """
    Convert PyTorch checkpoint to SafeTensors with minimal transformation.

    Transformations:
    - Conv1d weights: [C_out, C_in, 1] → [C_out, C_in] (squeeze kernel dim)
    - All other tensors: unchanged

    PyTorch keys are preserved exactly as they appear in state_dict.
    """

    print(f"Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract state_dict from various checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove common prefixes (module., model.) if present
    cleaned_dict = {}
    for key, value in state_dict.items():
        clean_key = key
        for prefix in ['module.', 'model.']:
            if key.startswith(prefix):
                clean_key = key[len(prefix):]
                break
        cleaned_dict[clean_key] = value

    state_dict = cleaned_dict
    print(f"Found {len(state_dict)} tensors")

    # Convert with minimal transformation
    tensors = {}
    conv_count = 0

    for key, value in state_dict.items():
        # Conv1d weights: [C_out, C_in, 1] → [C_out, C_in]
        if value.dim() == 3 and value.size(2) == 1:
            tensors[key] = value.squeeze(-1).contiguous()
            conv_count += 1
        else:
            tensors[key] = value.contiguous()

    # Determine output path
    if output_path is None:
        input_path = Path(checkpoint_path)
        output_path = input_path.with_suffix('.safetensors')

    # Save
    save_file(tensors, str(output_path))

    file_size = Path(output_path).stat().st_size / 1024 / 1024
    print(f"\nConverted {len(tensors)} tensors ({conv_count} Conv1d squeezed)")
    print(f"Output: {output_path} ({file_size:.2f} MB)")

    # Print key summary
    print("\nKey summary (first 10):")
    for i, key in enumerate(sorted(tensors.keys())[:10]):
        shape = list(tensors[key].shape)
        print(f"  {key}: {shape}")
    if len(tensors) > 10:
        print(f"  ... and {len(tensors) - 10} more")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python convert_to_safetensors.py checkpoint.pth [output.safetensors]")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    convert(checkpoint_path, output_path)

#!/usr/bin/env python3
"""
Generate reference data for MaxPooling1D test
Tests max pooling along dimension 1: [C, N] -> [C]
"""

import torch
import json
from pathlib import Path

def main():
    torch.manual_seed(42)
    
    # Test configuration
    N = 16   # Number of points
    C = 128  # Number of channels
    
    print(f"Generating MaxPooling1D reference data...")
    print(f"  Input: [{N}, {C}]")
    print(f"  Output: [{C}]")
    
    # Generate random input [C, N] - PyTorch native format
    x = torch.randn(C, N)
    
    # MaxPool along dimension 1 (points dimension)
    output, indices = torch.max(x, dim=1)
    
    print(f"  Input range: [{x.min():.6f}, {x.max():.6f}]")
    print(f"  Output range: [{output.min():.6f}, {output.max():.6f}]")
    
    # Verify: output should contain max values
    for c in range(min(5, C)):
        expected_max = max(x[c, :].tolist())
        actual_max = output[c].item()
        assert abs(expected_max - actual_max) < 1e-6, f"Channel {c}: {expected_max} != {actual_max}"
    
    # Prepare data
    data = {
        'shape': [float(C), float(N)],  # [C, N] layout
        'input': x.numpy().flatten().tolist(),
        'output': output.numpy().tolist(),
        'indices': indices.numpy().tolist(),  # For debugging
    }
    
    # Save to file
    output_dir = Path(__file__).parent
    output_file = output_dir / 'reference.json'
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Saved to {output_file}")
    print(f"  Input values: {C * N}")
    print(f"  Output values: {C}")
    
    # Show some sample values
    print(f"\nSample values:")
    print(f"  Input[0, :5]: {x[0, :5].tolist()}")
    print(f"  Input[1, :5]: {x[1, :5].tolist()}")
    print(f"  Output[:5]: {output[:5].tolist()}")
    print(f"  Indices[:5]: {indices[:5].tolist()} (which row had max)")

if __name__ == '__main__':
    main()

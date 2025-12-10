#!/usr/bin/env python3
"""
Test AddIdentityNode
Input: [K, K] matrix
Output: Input + I (identity matrix)
"""

import torch
import json

def main():
    torch.manual_seed(42)
    
    K = 3
    
    print(f"Generating AddIdentity reference...")
    print(f"  Input: [{K}, {K}]")
    print(f"  Output: [{K}, {K}]")
    
    # Generate random matrix
    x = torch.randn(K, K)
    
    # Add identity
    identity = torch.eye(K)
    output = x + identity
    
    print(f"\nInput matrix:")
    for i in range(K):
        print(f"  {x[i].tolist()}")
    
    print(f"\nIdentity matrix:")
    for i in range(K):
        print(f"  {identity[i].tolist()}")
    
    print(f"\nOutput matrix:")
    for i in range(K):
        print(f"  {output[i].tolist()}")
    
    data = {
        'K': [float(K)],  # Wrap in array for parseNDArray()
        'input': x.numpy().flatten().tolist(),
        'output': output.numpy().flatten().tolist(),
    }
    
    output_file = 'test/add_identity/reference.json'
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✓ Saved to {output_file}")

if __name__ == '__main__':
    main()

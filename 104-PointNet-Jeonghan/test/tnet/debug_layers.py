#!/usr/bin/env python3
"""
Debug script to check intermediate layer outputs
Compares Vulkan outputs with PyTorch reference
"""

import json
import sys

def check_layer_diff(name, vulkan_vals, pytorch_vals, tolerance=1e-4):
    """Compare two arrays and report differences"""
    if len(vulkan_vals) != len(pytorch_vals):
        print(f"✗ {name}: SIZE MISMATCH! Vulkan={len(vulkan_vals)}, PyTorch={len(pytorch_vals)}")
        return False
    
    diffs = [abs(v - p) for v, p in zip(vulkan_vals, pytorch_vals)]
    max_diff = max(diffs)
    avg_diff = sum(diffs) / len(diffs)
    mismatches = sum(1 for d in diffs if d > tolerance)
    
    status = "✓" if mismatches == 0 else "✗"
    print(f"{status} {name}:")
    print(f"    Values: {len(vulkan_vals)}")
    print(f"    Max diff: {max_diff:.6f}")
    print(f"    Avg diff: {avg_diff:.6f}")
    print(f"    Mismatches: {mismatches}/{len(vulkan_vals)}")
    
    if mismatches > 0 and mismatches < 10:
        print(f"    First few mismatches:")
        count = 0
        for i, (v, p, d) in enumerate(zip(vulkan_vals, pytorch_vals, diffs)):
            if d > tolerance:
                print(f"      [{i}] Expected: {p:.6f}, Got: {v:.6f}, Diff: {d:.6f}")
                count += 1
                if count >= 5:
                    break
    
    return mismatches == 0

def main():
    # Load reference
    with open('test/tnet/reference.json', 'r') as f:
        ref = json.load(f)
    
    print("=" * 60)
    print("TNet Layer-by-Layer Debugging")
    print("=" * 60)
    print()
    
    # Check if we have Vulkan outputs (need to generate these)
    # For now, just show PyTorch reference values
    
    print("PyTorch Reference Intermediate Values:")
    print("-" * 60)
    
    if 'debug_pooled' in ref:
        pooled = ref['debug_pooled']
        print(f"After MaxPool: [{len(pooled)}]")
        print(f"  Range: [{min(pooled):.6f}, {max(pooled):.6f}]")
        print(f"  First 5: {[f'{v:.6f}' for v in pooled[:5]]}")
        print()
    
    if 'debug_fc_out0' in ref:
        fc0 = ref['debug_fc_out0']
        print(f"After FC+BN+ReLU 0: [{len(fc0)}]")
        print(f"  Range: [{min(fc0):.6f}, {max(fc0):.6f}]")
        print(f"  First 5: {[f'{v:.6f}' for v in fc0[:5]]}")
        print()
    
    if 'debug_fc_out1' in ref:
        fc1 = ref['debug_fc_out1']
        print(f"After FC+BN+ReLU 1: [{len(fc1)}]")
        print(f"  Range: [{min(fc1):.6f}, {max(fc1):.6f}]")
        print(f"  First 5: {[f'{v:.6f}' for v in fc1[:5]]}")
        print()
    
    if 'debug_fc_out2' in ref:
        fc2 = ref['debug_fc_out2']
        print(f"After FC (last, no BN/ReLU): [{len(fc2)}]")
        print(f"  Range: [{min(fc2):.6f}, {max(fc2):.6f}]")
        print(f"  First 5: {[f'{v:.6f}' for v in fc2[:5]]}")
        print()
    
    if 'debug_transform_no_id' in ref:
        transform_no_id = ref['debug_transform_no_id']
        print(f"Transform (before identity): [3x3] = {len(transform_no_id)} values")
        print(f"  Range: [{min(transform_no_id):.6f}, {max(transform_no_id):.6f}]")
        for i in range(3):
            row = transform_no_id[i*3:(i+1)*3]
            print(f"  Row {i}: {[f'{v:.6f}' for v in row]}")
        print()
    
    if 'transform' in ref:
        transform = ref['transform']
        print(f"Transform (with identity): [3x3] = {len(transform)} values")
        print(f"  Range: [{min(transform):.6f}, {max(transform):.6f}]")
        for i in range(3):
            row = transform[i*3:(i+1)*3]
            print(f"  Row {i}: {[f'{v:.6f}' for v in row]}")
        print()
        
        # Check diagonal (should be close to 1 due to identity)
        print("  Diagonal values (should be ~1.0 due to identity):")
        for i in range(3):
            val = transform[i*3 + i]
            print(f"    [{i},{i}] = {val:.6f}")

if __name__ == '__main__':
    main()

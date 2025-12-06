#!/usr/bin/env python3
"""
Convert PyTorch PointNet weights to JSON format for Vulkan inference.

Based on: https://github.com/yanx27/Pointnet_Pointnet2_pytorch

Usage:
    python convert_pytorch_weights.py --checkpoint <path_to_pytorch_checkpoint> --output <output_json>
"""

import torch
import json
import numpy as np
import argparse
from pathlib import Path


def convert_tensor_to_list(tensor):
    """Convert PyTorch tensor to nested list."""
    return tensor.detach().cpu().numpy().tolist()


def convert_pointnet_weights(checkpoint_path, output_path):
    """
    Convert PointNet segmentation weights from PyTorch to JSON.
    
    Expected PyTorch model structure (from reference repo):
    - Input Transform Network (TNet1): 3x3
    - MLP1: [3, 64, 64]
    - Feature Transform Network (TNet2): 64x64
    - MLP2: [64, 128, 1024]
    - Segmentation Head: [2048, 512, 256, num_classes]
    """
    
    print(f"Loading PyTorch checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    print(f"Found {len(state_dict)} parameters")
    
    # Initialize output dictionary
    weights = {}
    
    # Mapping from PyTorch keys to our JSON keys
    key_mapping = {
        # TNet1 (input transformation)
        'tnet1.conv1.weight': 'tnet1.mlp.0.weight',
        'tnet1.conv1.bias': 'tnet1.mlp.0.bias',
        'tnet1.conv2.weight': 'tnet1.mlp.1.weight',
        'tnet1.conv2.bias': 'tnet1.mlp.1.bias',
        'tnet1.conv3.weight': 'tnet1.mlp.2.weight',
        'tnet1.conv3.bias': 'tnet1.mlp.2.bias',
        
        'tnet1.fc1.weight': 'tnet1.fc.0.weight',
        'tnet1.fc1.bias': 'tnet1.fc.0.bias',
        'tnet1.fc2.weight': 'tnet1.fc.1.weight',
        'tnet1.fc2.bias': 'tnet1.fc.1.bias',
        'tnet1.fc3.weight': 'tnet1.fc.2.weight',
        'tnet1.fc3.bias': 'tnet1.fc.2.bias',
        'tnet1.fc4.weight': 'tnet1.fc.3.weight',
        'tnet1.fc4.bias': 'tnet1.fc.3.bias',
        
        # MLP1 (first feature extraction)
        'mlp1.conv1.weight': 'mlp1.0.weight',
        'mlp1.conv1.bias': 'mlp1.0.bias',
        'mlp1.conv2.weight': 'mlp1.1.weight',
        'mlp1.conv2.bias': 'mlp1.1.bias',
        
        # TNet2 (feature transformation)
        'tnet2.conv1.weight': 'tnet2.mlp.0.weight',
        'tnet2.conv1.bias': 'tnet2.mlp.0.bias',
        'tnet2.conv2.weight': 'tnet2.mlp.1.weight',
        'tnet2.conv2.bias': 'tnet2.mlp.1.bias',
        'tnet2.conv3.weight': 'tnet2.mlp.2.weight',
        'tnet2.conv3.bias': 'tnet2.mlp.2.bias',
        
        'tnet2.fc1.weight': 'tnet2.fc.0.weight',
        'tnet2.fc1.bias': 'tnet2.fc.0.bias',
        'tnet2.fc2.weight': 'tnet2.fc.1.weight',
        'tnet2.fc2.bias': 'tnet2.fc.1.bias',
        'tnet2.fc3.weight': 'tnet2.fc.2.weight',
        'tnet2.fc3.bias': 'tnet2.fc.2.bias',
        'tnet2.fc4.weight': 'tnet2.fc.3.weight',
        'tnet2.fc4.bias': 'tnet2.fc.3.bias',
        
        # MLP2 (second feature extraction)
        'mlp2.conv1.weight': 'mlp2.0.weight',
        'mlp2.conv1.bias': 'mlp2.0.bias',
        'mlp2.conv2.weight': 'mlp2.1.weight',
        'mlp2.conv2.bias': 'mlp2.1.bias',
        
        # Segmentation head
        'seg_head.conv1.weight': 'segHead.0.weight',
        'seg_head.conv1.bias': 'segHead.0.bias',
        'seg_head.conv2.weight': 'segHead.1.weight',
        'seg_head.conv2.bias': 'segHead.1.bias',
        'seg_head.conv3.weight': 'segHead.2.weight',
        'seg_head.conv3.bias': 'segHead.2.bias',
    }
    
    # Convert weights
    converted_count = 0
    for pytorch_key, json_key in key_mapping.items():
        # Try to find the key with various prefixes
        found = False
        for prefix in ['', 'module.', 'model.']:
            full_key = prefix + pytorch_key
            if full_key in state_dict:
                weights[json_key] = convert_tensor_to_list(state_dict[full_key])
                converted_count += 1
                found = True
                print(f"✓ Converted: {full_key} -> {json_key}")
                break
        
        if not found:
            print(f"⚠ Warning: Key not found: {pytorch_key}")
    
    print(f"\nConverted {converted_count} / {len(key_mapping)} parameters")
    
    # Save to JSON
    print(f"\nSaving to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(weights, f, indent=2)
    
    print(f"✓ Conversion complete!")
    print(f"  Output size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")


def create_random_weights(output_path, num_classes=10):
    """
    Create random weights for testing (when no pretrained model available).
    """
    print(f"Creating random weights for {num_classes} classes...")
    
    weights = {}
    
    # TNet1: 3 -> 64 -> 128 -> 1024, then 1024 -> 512 -> 256 -> 9
    # Note: C++ expects [input, output] shape, so transpose PyTorch [output, input]
    weights['tnet1.mlp.0.weight'] = np.random.randn(3, 64).astype(np.float32).tolist()
    weights['tnet1.mlp.0.bias'] = np.random.randn(64).astype(np.float32).tolist()
    weights['tnet1.mlp.1.weight'] = np.random.randn(64, 128).astype(np.float32).tolist()
    weights['tnet1.mlp.1.bias'] = np.random.randn(128).astype(np.float32).tolist()
    weights['tnet1.mlp.2.weight'] = np.random.randn(128, 1024).astype(np.float32).tolist()
    weights['tnet1.mlp.2.bias'] = np.random.randn(1024).astype(np.float32).tolist()
    
    weights['tnet1.fc.0.weight'] = np.random.randn(1024, 512).astype(np.float32).tolist()
    weights['tnet1.fc.0.bias'] = np.random.randn(512).astype(np.float32).tolist()
    weights['tnet1.fc.1.weight'] = np.random.randn(512, 256).astype(np.float32).tolist()
    weights['tnet1.fc.1.bias'] = np.random.randn(256).astype(np.float32).tolist()
    weights['tnet1.fc.2.weight'] = np.random.randn(256, 9).astype(np.float32).tolist()
    weights['tnet1.fc.2.bias'] = np.random.randn(9).astype(np.float32).tolist()
    
    # MLP1: 3 -> 64 -> 64
    weights['mlp1.0.weight'] = np.random.randn(3, 64).astype(np.float32).tolist()
    weights['mlp1.0.bias'] = np.random.randn(64).astype(np.float32).tolist()
    weights['mlp1.1.weight'] = np.random.randn(64, 64).astype(np.float32).tolist()
    weights['mlp1.1.bias'] = np.random.randn(64).astype(np.float32).tolist()
    
    # TNet2: 64 -> 64 -> 128 -> 1024, then 1024 -> 512 -> 256 -> 64*64
    weights['tnet2.mlp.0.weight'] = np.random.randn(64, 64).astype(np.float32).tolist()
    weights['tnet2.mlp.0.bias'] = np.random.randn(64).astype(np.float32).tolist()
    weights['tnet2.mlp.1.weight'] = np.random.randn(64, 128).astype(np.float32).tolist()
    weights['tnet2.mlp.1.bias'] = np.random.randn(128).astype(np.float32).tolist()
    weights['tnet2.mlp.2.weight'] = np.random.randn(128, 1024).astype(np.float32).tolist()
    weights['tnet2.mlp.2.bias'] = np.random.randn(1024).astype(np.float32).tolist()
    
    weights['tnet2.fc.0.weight'] = np.random.randn(1024, 512).astype(np.float32).tolist()
    weights['tnet2.fc.0.bias'] = np.random.randn(512).astype(np.float32).tolist()
    weights['tnet2.fc.1.weight'] = np.random.randn(512, 256).astype(np.float32).tolist()
    weights['tnet2.fc.1.bias'] = np.random.randn(256).astype(np.float32).tolist()
    weights['tnet2.fc.2.weight'] = np.random.randn(256, 4096).astype(np.float32).tolist()
    weights['tnet2.fc.2.bias'] = np.random.randn(4096).astype(np.float32).tolist()
    
    # MLP2: 64 -> 128 -> 1024
    weights['mlp2.0.weight'] = np.random.randn(64, 128).astype(np.float32).tolist()
    weights['mlp2.0.bias'] = np.random.randn(128).astype(np.float32).tolist()
    weights['mlp2.1.weight'] = np.random.randn(128, 1024).astype(np.float32).tolist()
    weights['mlp2.1.bias'] = np.random.randn(1024).astype(np.float32).tolist()
    
    # Segmentation head: 2048 -> 512 -> 256 -> num_classes
    weights['segHead.0.weight'] = np.random.randn(2048, 512).astype(np.float32).tolist()
    weights['segHead.0.bias'] = np.random.randn(512).astype(np.float32).tolist()
    weights['segHead.1.weight'] = np.random.randn(512, 256).astype(np.float32).tolist()
    weights['segHead.1.bias'] = np.random.randn(256).astype(np.float32).tolist()
    weights['segHead.2.weight'] = np.random.randn(256, num_classes).astype(np.float32).tolist()
    weights['segHead.2.bias'] = np.random.randn(num_classes).astype(np.float32).tolist()
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(weights, f, indent=2)
    
    print(f"✓ Random weights created!")
    print(f"  Output: {output_path}")
    print(f"  Size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Convert PointNet PyTorch weights to JSON')
    parser.add_argument('--checkpoint', type=str, help='Path to PyTorch checkpoint (.pth)')
    parser.add_argument('--output', type=str, default='assets/weights/pointnet_weights.json',
                        help='Output JSON path')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of segmentation classes')
    parser.add_argument('--random', action='store_true', help='Create random weights for testing')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    if args.random:
        create_random_weights(args.output, args.num_classes)
    else:
        if not args.checkpoint:
            print("Error: --checkpoint required (or use --random for testing)")
            return
        convert_pointnet_weights(args.checkpoint, args.output)


if __name__ == '__main__':
    main()


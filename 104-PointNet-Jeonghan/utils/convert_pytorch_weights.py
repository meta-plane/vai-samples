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
    
    Based on yanx27/Pointnet_Pointnet2_pytorch structure:
    - feat.stn (STN3d): Conv1d[9→64→128→1024] + FC[1024→512→256→9]
    - feat.conv1-3: Conv1d[9→64→128→1024]
    - feat.fstn (STNkd): Conv1d[64→64→128→1024] + FC[1024→512→256→4096]
    - Segmentation head: Conv1d[1088→512→256→128→13]
    """
    
    print(f"Loading PyTorch checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
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
    
    # Mapping from yanx27 PyTorch keys to our JSON keys
    # yanx27 structure: feat.stn, feat.conv1-3, feat.fstn, conv1-4, bn1-3
    key_mapping = {
        # feat.stn (STN3d): [9→64→128→1024] + FC[1024→512→256→9]
        'feat.stn.conv1.weight': 'feat.stn.mlp.mlp0.weight',
        'feat.stn.conv1.bias': 'feat.stn.mlp.mlp0.bias',
        'feat.stn.conv2.weight': 'feat.stn.mlp.mlp1.weight',
        'feat.stn.conv2.bias': 'feat.stn.mlp.mlp1.bias',
        'feat.stn.conv3.weight': 'feat.stn.mlp.mlp2.weight',
        'feat.stn.conv3.bias': 'feat.stn.mlp.mlp2.bias',
        
        'feat.stn.fc1.weight': 'feat.stn.fc.block0.weight',
        'feat.stn.fc1.bias': 'feat.stn.fc.block0.bias',
        'feat.stn.fc2.weight': 'feat.stn.fc.block1.weight',
        'feat.stn.fc2.bias': 'feat.stn.fc.block1.bias',
        'feat.stn.fc3.weight': 'feat.stn.fc.lastBlock.weight',
        'feat.stn.fc3.bias': 'feat.stn.fc.lastBlock.bias',
        
        # BatchNorm for STN
        'feat.stn.bn1.weight': 'feat.stn.mlp.mlp0.gamma',
        'feat.stn.bn1.bias': 'feat.stn.mlp.mlp0.beta',
        'feat.stn.bn1.running_mean': 'feat.stn.mlp.mlp0.mean',
        'feat.stn.bn1.running_var': 'feat.stn.mlp.mlp0.var',
        'feat.stn.bn2.weight': 'feat.stn.mlp.mlp1.gamma',
        'feat.stn.bn2.bias': 'feat.stn.mlp.mlp1.beta',
        'feat.stn.bn2.running_mean': 'feat.stn.mlp.mlp1.mean',
        'feat.stn.bn2.running_var': 'feat.stn.mlp.mlp1.var',
        'feat.stn.bn3.weight': 'feat.stn.mlp.mlp2.gamma',
        'feat.stn.bn3.bias': 'feat.stn.mlp.mlp2.beta',
        'feat.stn.bn3.running_mean': 'feat.stn.mlp.mlp2.mean',
        'feat.stn.bn3.running_var': 'feat.stn.mlp.mlp2.var',
        'feat.stn.bn4.weight': 'feat.stn.fc.block0.gamma',
        'feat.stn.bn4.bias': 'feat.stn.fc.block0.beta',
        'feat.stn.bn4.running_mean': 'feat.stn.fc.block0.mean',
        'feat.stn.bn4.running_var': 'feat.stn.fc.block0.var',
        'feat.stn.bn5.weight': 'feat.stn.fc.block1.gamma',
        'feat.stn.bn5.bias': 'feat.stn.fc.block1.beta',
        'feat.stn.bn5.running_mean': 'feat.stn.fc.block1.mean',
        'feat.stn.bn5.running_var': 'feat.stn.fc.block1.var',
        
        # feat.conv1-3 (main feature extraction): [9→64→128→1024]
        'feat.conv1.weight': 'feat.conv.mlp0.weight',
        'feat.conv1.bias': 'feat.conv.mlp0.bias',
        'feat.conv2.weight': 'feat.conv.mlp1.weight',
        'feat.conv2.bias': 'feat.conv.mlp1.bias',
        'feat.conv3.weight': 'feat.conv.mlp2.weight',
        'feat.conv3.bias': 'feat.conv.mlp2.bias',
        
        'feat.bn1.weight': 'feat.conv.mlp0.gamma',
        'feat.bn1.bias': 'feat.conv.mlp0.beta',
        'feat.bn1.running_mean': 'feat.conv.mlp0.mean',
        'feat.bn1.running_var': 'feat.conv.mlp0.var',
        'feat.bn2.weight': 'feat.conv.mlp1.gamma',
        'feat.bn2.bias': 'feat.conv.mlp1.beta',
        'feat.bn2.running_mean': 'feat.conv.mlp1.mean',
        'feat.bn2.running_var': 'feat.conv.mlp1.var',
        'feat.bn3.weight': 'feat.conv.mlp2.gamma',
        'feat.bn3.bias': 'feat.conv.mlp2.beta',
        'feat.bn3.running_mean': 'feat.conv.mlp2.mean',
        'feat.bn3.running_var': 'feat.conv.mlp2.var',
        
        # feat.fstn (STNkd): [64→64→128→1024] + FC[1024→512→256→4096]
        'feat.fstn.conv1.weight': 'feat.fstn.mlp.mlp0.weight',
        'feat.fstn.conv1.bias': 'feat.fstn.mlp.mlp0.bias',
        'feat.fstn.conv2.weight': 'feat.fstn.mlp.mlp1.weight',
        'feat.fstn.conv2.bias': 'feat.fstn.mlp.mlp1.bias',
        'feat.fstn.conv3.weight': 'feat.fstn.mlp.mlp2.weight',
        'feat.fstn.conv3.bias': 'feat.fstn.mlp.mlp2.bias',
        
        'feat.fstn.fc1.weight': 'feat.fstn.fc.block0.weight',
        'feat.fstn.fc1.bias': 'feat.fstn.fc.block0.bias',
        'feat.fstn.fc2.weight': 'feat.fstn.fc.block1.weight',
        'feat.fstn.fc2.bias': 'feat.fstn.fc.block1.bias',
        'feat.fstn.fc3.weight': 'feat.fstn.fc.lastBlock.weight',
        'feat.fstn.fc3.bias': 'feat.fstn.fc.lastBlock.bias',
        
        'feat.fstn.bn1.weight': 'feat.fstn.mlp.mlp0.gamma',
        'feat.fstn.bn1.bias': 'feat.fstn.mlp.mlp0.beta',
        'feat.fstn.bn1.running_mean': 'feat.fstn.mlp.mlp0.mean',
        'feat.fstn.bn1.running_var': 'feat.fstn.mlp.mlp0.var',
        'feat.fstn.bn2.weight': 'feat.fstn.mlp.mlp1.gamma',
        'feat.fstn.bn2.bias': 'feat.fstn.mlp.mlp1.beta',
        'feat.fstn.bn2.running_mean': 'feat.fstn.mlp.mlp1.mean',
        'feat.fstn.bn2.running_var': 'feat.fstn.mlp.mlp1.var',
        'feat.fstn.bn3.weight': 'feat.fstn.mlp.mlp2.gamma',
        'feat.fstn.bn3.bias': 'feat.fstn.mlp.mlp2.beta',
        'feat.fstn.bn3.running_mean': 'feat.fstn.mlp.mlp2.mean',
        'feat.fstn.bn3.running_var': 'feat.fstn.mlp.mlp2.var',
        'feat.fstn.bn4.weight': 'feat.fstn.fc.block0.gamma',
        'feat.fstn.bn4.bias': 'feat.fstn.fc.block0.beta',
        'feat.fstn.bn4.running_mean': 'feat.fstn.fc.block0.mean',
        'feat.fstn.bn4.running_var': 'feat.fstn.fc.block0.var',
        'feat.fstn.bn5.weight': 'feat.fstn.fc.block1.gamma',
        'feat.fstn.bn5.bias': 'feat.fstn.fc.block1.beta',
        'feat.fstn.bn5.running_mean': 'feat.fstn.fc.block1.mean',
        'feat.fstn.bn5.running_var': 'feat.fstn.fc.block1.var',
        
        # Segmentation head: [1088→512→256→128→13]
        'conv1.weight': 'conv1.weight',
        'conv1.bias': 'conv1.bias',
        'conv2.weight': 'conv2.weight',
        'conv2.bias': 'conv2.bias',
        'conv3.weight': 'conv3.weight',
        'conv3.bias': 'conv3.bias',
        'conv4.weight': 'conv4.weight',
        'conv4.bias': 'conv4.bias',
        
        'bn1.weight': 'bn1.weight',
        'bn1.bias': 'bn1.bias',
        'bn1.running_mean': 'bn1.mean',
        'bn1.running_var': 'bn1.var',
        'bn2.weight': 'bn2.weight',
        'bn2.bias': 'bn2.bias',
        'bn2.running_mean': 'bn2.mean',
        'bn2.running_var': 'bn2.var',
        'bn3.weight': 'bn3.weight',
        'bn3.bias': 'bn3.bias',
        'bn3.running_mean': 'bn3.mean',
        'bn3.running_var': 'bn3.var',
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


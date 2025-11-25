#!/usr/bin/env python3
"""
Convert PyTorch PointNet weights to SafeTensors format (recommended).

SafeTensors is faster and safer than JSON:
- Binary format (10x faster loading)
- Memory-mapped (efficient for large models)
- Safe against code injection
- Standard format used by Hugging Face

Usage:
    # From PyTorch checkpoint
    python convert_to_safetensors.py --checkpoint model.pth --output weights.safetensors
    
    # Create random weights for testing
    python convert_to_safetensors.py --random --num_classes 10
"""

import torch
import numpy as np
import argparse
from pathlib import Path

try:
    from safetensors.torch import save_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("⚠ Warning: safetensors not installed")
    print("  Install with: pip install safetensors")


def convert_pointnet_to_safetensors(checkpoint_path, output_path):
    """
    Convert PyTorch PointNet checkpoint to SafeTensors format.
    """
    
    if not SAFETENSORS_AVAILABLE:
        print("Error: safetensors package required")
        print("Install: pip install safetensors")
        return
    
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
    
    # Mapping from PyTorch keys to our keys
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
        
        # MLP1
        'mlp1.conv1.weight': 'mlp1.0.weight',
        'mlp1.conv1.bias': 'mlp1.0.bias',
        'mlp1.conv2.weight': 'mlp1.1.weight',
        'mlp1.conv2.bias': 'mlp1.1.bias',
        
        # TNet2
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
        
        # MLP2
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
    converted_weights = {}
    converted_count = 0
    
    for pytorch_key, our_key in key_mapping.items():
        found = False
        for prefix in ['', 'module.', 'model.']:
            full_key = prefix + pytorch_key
            if full_key in state_dict:
                tensor = state_dict[full_key].contiguous()
                converted_weights[our_key] = tensor
                converted_count += 1
                found = True
                print(f"✓ {full_key} -> {our_key}")
                break
        
        if not found:
            print(f"⚠ Warning: {pytorch_key} not found")
    
    print(f"\nConverted {converted_count} / {len(key_mapping)} parameters")
    
    # Save to SafeTensors
    print(f"\nSaving to SafeTensors: {output_path}")
    save_file(converted_weights, output_path)
    
    file_size = Path(output_path).stat().st_size / 1024 / 1024
    print(f"✓ Conversion complete!")
    print(f"  Output: {output_path}")
    print(f"  Size: {file_size:.2f} MB")
    print(f"  Format: SafeTensors (binary, memory-mapped)")


def create_random_safetensors(output_path, num_classes=10):
    """
    Create random weights in SafeTensors format for testing.
    """
    
    if not SAFETENSORS_AVAILABLE:
        print("Error: safetensors package required")
        print("Install: pip install safetensors")
        return
    
    print(f"Creating random weights for {num_classes} classes...")
    
    weights = {}
    
    # TNet1: 3 -> 64 -> 128 -> 1024, then 1024 -> 512 -> 256 -> 9
    weights['tnet1.mlp.0.weight'] = torch.randn(64, 3)
    weights['tnet1.mlp.0.bias'] = torch.randn(64)
    weights['tnet1.mlp.1.weight'] = torch.randn(128, 64)
    weights['tnet1.mlp.1.bias'] = torch.randn(128)
    weights['tnet1.mlp.2.weight'] = torch.randn(1024, 128)
    weights['tnet1.mlp.2.bias'] = torch.randn(1024)
    
    weights['tnet1.fc.0.weight'] = torch.randn(512, 1024)
    weights['tnet1.fc.0.bias'] = torch.randn(512)
    weights['tnet1.fc.1.weight'] = torch.randn(256, 512)
    weights['tnet1.fc.1.bias'] = torch.randn(256)
    weights['tnet1.fc.2.weight'] = torch.randn(9, 256)
    weights['tnet1.fc.2.bias'] = torch.randn(9)
    weights['tnet1.fc.3.weight'] = torch.eye(3).flatten()  # Identity
    weights['tnet1.fc.3.bias'] = torch.zeros(9)
    
    # MLP1: 3 -> 64 -> 64
    weights['mlp1.0.weight'] = torch.randn(64, 3)
    weights['mlp1.0.bias'] = torch.randn(64)
    weights['mlp1.1.weight'] = torch.randn(64, 64)
    weights['mlp1.1.bias'] = torch.randn(64)
    
    # TNet2: 64 -> 64 -> 128 -> 1024, then 1024 -> 512 -> 256 -> 64*64
    weights['tnet2.mlp.0.weight'] = torch.randn(64, 64)
    weights['tnet2.mlp.0.bias'] = torch.randn(64)
    weights['tnet2.mlp.1.weight'] = torch.randn(128, 64)
    weights['tnet2.mlp.1.bias'] = torch.randn(128)
    weights['tnet2.mlp.2.weight'] = torch.randn(1024, 128)
    weights['tnet2.mlp.2.bias'] = torch.randn(1024)
    
    weights['tnet2.fc.0.weight'] = torch.randn(512, 1024)
    weights['tnet2.fc.0.bias'] = torch.randn(512)
    weights['tnet2.fc.1.weight'] = torch.randn(256, 512)
    weights['tnet2.fc.1.bias'] = torch.randn(256)
    weights['tnet2.fc.2.weight'] = torch.randn(4096, 256)
    weights['tnet2.fc.2.bias'] = torch.randn(4096)
    weights['tnet2.fc.3.weight'] = torch.eye(64).flatten()  # Identity
    weights['tnet2.fc.3.bias'] = torch.zeros(4096)
    
    # MLP2: 64 -> 128 -> 1024
    weights['mlp2.0.weight'] = torch.randn(128, 64)
    weights['mlp2.0.bias'] = torch.randn(128)
    weights['mlp2.1.weight'] = torch.randn(1024, 128)
    weights['mlp2.1.bias'] = torch.randn(1024)
    
    # Segmentation head: 2048 -> 512 -> 256 -> num_classes
    weights['segHead.0.weight'] = torch.randn(512, 2048)
    weights['segHead.0.bias'] = torch.randn(512)
    weights['segHead.1.weight'] = torch.randn(256, 512)
    weights['segHead.1.bias'] = torch.randn(256)
    weights['segHead.2.weight'] = torch.randn(num_classes, 256)
    weights['segHead.2.bias'] = torch.randn(num_classes)
    
    # Save to SafeTensors
    save_file(weights, output_path)
    
    file_size = Path(output_path).stat().st_size / 1024 / 1024
    print(f"✓ Random weights created!")
    print(f"  Output: {output_path}")
    print(f"  Size: {file_size:.2f} MB")
    print(f"  Format: SafeTensors")


def main():
    parser = argparse.ArgumentParser(
        description='Convert PointNet weights to SafeTensors format'
    )
    parser.add_argument('--checkpoint', type=str, 
                        help='Path to PyTorch checkpoint (.pth)')
    parser.add_argument('--output', type=str, 
                        default='assets/weights/pointnet_weights.safetensors',
                        help='Output SafeTensors path')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of segmentation classes')
    parser.add_argument('--random', action='store_true',
                        help='Create random weights for testing')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    if args.random:
        create_random_safetensors(args.output, args.num_classes)
    else:
        if not args.checkpoint:
            print("Error: --checkpoint required (or use --random)")
            return
        convert_pointnet_to_safetensors(args.checkpoint, args.output)


if __name__ == '__main__':
    main()


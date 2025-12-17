#!/usr/bin/env python3
"""
PyTorch PointNet Inference Benchmark
Compare performance with Vulkan implementation
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
from pathlib import Path

# PointNet model definition (yanx27 architecture for S3DIS segmentation)
class TNetBlock(nn.Module):
    def __init__(self, k):
        super(TNetBlock, self).__init__()
        self.k = k
        self.mlp1 = nn.Sequential(
            nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k*k)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.mlp1(x)
        x = torch.max(x, 2, keepdim=False)[0]
        x = self.fc(x)
        
        identity = torch.eye(self.k, device=x.device).flatten().unsqueeze(0)
        x = x + identity
        x = x.view(batch_size, self.k, self.k)
        return x

class PointNetEncoder(nn.Module):
    def __init__(self, channel=9):
        super(PointNetEncoder, self).__init__()
        self.channel = channel
        
        # Input transformation
        self.input_transform = TNetBlock(k=channel)
        
        # MLP [64, 64]
        self.mlp1 = nn.Sequential(
            nn.Conv1d(channel, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Feature transformation
        self.feature_transform = TNetBlock(k=64)
        
        # MLP [64, 128, 1024]
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024)
        )

    def forward(self, x):
        num_points = x.size(2)
        
        # Input transformation
        trans = self.input_transform(x)
        x = torch.bmm(trans, x)
        
        # First MLP
        point_feat = self.mlp1(x)
        
        # Feature transformation
        trans_feat = self.feature_transform(point_feat)
        x = torch.bmm(trans_feat, point_feat)
        
        # Second MLP
        x = self.mlp2(x)
        
        # Global feature
        global_feat = torch.max(x, 2, keepdim=False)[0]
        
        # Broadcast to all points
        global_feat_expanded = global_feat.unsqueeze(2).expand(-1, -1, num_points)
        
        # Concatenate point features with global feature
        x = torch.cat([point_feat, global_feat_expanded], dim=1)
        
        return x

class PointNetSegment(nn.Module):
    def __init__(self, num_classes=13, channel=9):
        super(PointNetSegment, self).__init__()
        self.num_classes = num_classes
        self.channel = channel
        
        # Encoder
        self.encoder = PointNetEncoder(channel=channel)
        
        # Segmentation head: [1088, 512, 256, 128, num_classes]
        self.seg_head = nn.Sequential(
            nn.Conv1d(1088, 512, 1),  # 64 + 1024 = 1088
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.seg_head(x)
        return x


def load_modelnet40_sample(modelnet_path, num_points=1024, split='test'):
    """Load a sample from ModelNet40 dataset"""
    modelnet_path = Path(modelnet_path)
    
    # Find first available class
    for class_dir in sorted(modelnet_path.iterdir()):
        if not class_dir.is_dir():
            continue
        
        split_dir = class_dir / split
        if not split_dir.exists():
            continue
        
        # Load first .off file
        for off_file in sorted(split_dir.glob('*.off')):
            print(f"Loading: {off_file}")
            
            # Simple OFF file parser
            with open(off_file, 'r') as f:
                lines = f.readlines()
                
                # Skip header
                if lines[0].strip() == 'OFF':
                    header_idx = 1
                else:
                    header_idx = 0
                
                # Parse vertex/face counts
                n_verts, n_faces, _ = map(int, lines[header_idx].strip().split())
                
                # Read vertices
                vertices = []
                for i in range(n_verts):
                    vertex = list(map(float, lines[header_idx + 1 + i].strip().split()[:3]))
                    vertices.append(vertex)
                
                vertices = np.array(vertices, dtype=np.float32)
                
                # Sample points if needed
                if len(vertices) > num_points:
                    indices = np.random.choice(len(vertices), num_points, replace=False)
                    vertices = vertices[indices]
                elif len(vertices) < num_points:
                    # Duplicate points if insufficient
                    indices = np.random.choice(len(vertices), num_points, replace=True)
                    vertices = vertices[indices]
                
                return class_dir.name, vertices
    
    return None, None


def benchmark_pytorch(model, point_cloud, num_iterations=10, device='cuda'):
    """Benchmark PyTorch inference"""
    model.eval()
    
    # Prepare input: [B, C, N] = [1, 9, 1024]
    # Expand xyz to 9 channels (xyz + normals_placeholder)
    point_cloud_9ch = np.zeros((1, 9, point_cloud.shape[0]), dtype=np.float32)
    point_cloud_9ch[0, :3, :] = point_cloud.T  # xyz
    # Leave channels 3-8 as zeros (placeholder for normals/features)
    
    x = torch.from_numpy(point_cloud_9ch).to(device)
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
    
    # Benchmark
    print(f"\nRunning inference benchmark ({num_iterations} iterations)...")
    iteration_times = []
    
    with torch.no_grad():
        for i in range(num_iterations):
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            output = model(x)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            iteration_times.append(elapsed_ms)
            
            print(f"  Iteration {i+1}: {elapsed_ms:.2f} ms")
    
    # Statistics
    avg_time = np.mean(iteration_times)
    min_time = np.min(iteration_times)
    max_time = np.max(iteration_times)
    
    # Get predictions
    with torch.no_grad():
        output = model(x)
        preds = torch.argmax(output, dim=1).cpu().numpy()[0]
    
    return {
        'avg_time_ms': avg_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'iterations': num_iterations,
        'num_points': point_cloud.shape[0],
        'predictions': preds
    }


def main():
    print("╔════════════════════════════════════════════════════════╗")
    print("║    PyTorch PointNet Inference Benchmark               ║")
    print("║         S3DIS Semantic Segmentation                   ║")
    print("╚════════════════════════════════════════════════════════╝\n")
    
    # Configuration
    num_classes = 13
    channel = 9
    num_points = 1024
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")
    print(f"Points: {num_points}")
    print(f"Classes: {num_classes}\n")
    
    # Load ModelNet40 sample
    modelnet_path = "assets/datasets/ModelNet40"
    class_name, point_cloud = load_modelnet40_sample(modelnet_path, num_points)
    
    if point_cloud is None:
        print(f"Error: Could not load ModelNet40 sample from {modelnet_path}")
        return
    
    print(f"Loaded: {class_name} ({point_cloud.shape[0]} points)\n")
    
    # Create model
    print("Creating model...")
    model = PointNetSegment(num_classes=num_classes, channel=channel).to(device)
    
    # Load weights if available
    weights_path = "assets/weights/pointnet_yanx27.pth"
    if Path(weights_path).exists():
        print(f"Loading weights from {weights_path}...")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Weights not found at {weights_path}, using random weights")
    
    # Benchmark
    results = benchmark_pytorch(model, point_cloud, num_iterations=10, device=device)
    
    # Print results
    print("\n" + "="*56)
    print("Benchmark Results")
    print("="*56)
    print(f"Average time: {results['avg_time_ms']:.2f} ms")
    print(f"Min time:     {results['min_time_ms']:.2f} ms")
    print(f"Max time:     {results['max_time_ms']:.2f} ms")
    print(f"Throughput:   {results['num_points'] / (results['avg_time_ms'] / 1000.0):.0f} points/sec")
    
    # Show predicted classes
    preds = results['predictions']
    class_counts = np.bincount(preds, minlength=num_classes)
    top_classes = np.argsort(class_counts)[::-1][:3]
    
    print("\nTop semantic classes detected:")
    for cls in top_classes:
        percentage = 100.0 * class_counts[cls] / len(preds)
        print(f"  Class {cls}: {percentage:.1f}% ({class_counts[cls]} points)")
    
    print("="*56)
    
    # Save results
    output_file = "benchmark_results_pytorch.json"
    with open(output_file, 'w') as f:
        json.dump({
            'device': device,
            'avg_time_ms': float(results['avg_time_ms']),
            'min_time_ms': float(results['min_time_ms']),
            'max_time_ms': float(results['max_time_ms']),
            'throughput_points_per_sec': float(results['num_points'] / (results['avg_time_ms'] / 1000.0)),
            'num_points': int(results['num_points']),
            'num_iterations': int(results['iterations'])
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()

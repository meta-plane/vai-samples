#!/usr/bin/env python3
"""
PointNet Benchmark: Vulkan vs PyTorch Performance Comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import subprocess
import re
from pathlib import Path
from torch.autograd import Variable


# ============================================================
# PyTorch PointNet Model (yanx27 architecture)
# ============================================================

class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = Variable(torch.from_numpy(
            np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)
        )).view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = Variable(torch.from_numpy(
            np.eye(self.k).flatten().astype(np.float32)
        )).view(1, self.k * self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=False, feature_transform=True, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetSegment(nn.Module):
    def __init__(self, num_classes=13, channel=9):
        super(PointNetSegment, self).__init__()
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=channel)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, num_classes, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x


# ============================================================
# Benchmark Functions
# ============================================================

def load_modelnet40_sample(modelnet_path, num_points=1024):
    """Load a sample from ModelNet40"""
    modelnet_path = Path(modelnet_path)

    for class_dir in sorted(modelnet_path.iterdir()):
        if not class_dir.is_dir():
            continue
        split_dir = class_dir / 'test'
        if not split_dir.exists():
            continue

        for off_file in sorted(split_dir.glob('*.off')):
            with open(off_file, 'r') as f:
                lines = f.readlines()
                header_idx = 1 if lines[0].strip() == 'OFF' else 0
                n_verts = int(lines[header_idx].strip().split()[0])

                vertices = []
                for i in range(n_verts):
                    vertex = list(map(float, lines[header_idx + 1 + i].strip().split()[:3]))
                    vertices.append(vertex)

                vertices = np.array(vertices, dtype=np.float32)

                if len(vertices) > num_points:
                    indices = np.random.choice(len(vertices), num_points, replace=False)
                    vertices = vertices[indices]
                elif len(vertices) < num_points:
                    indices = np.random.choice(len(vertices), num_points, replace=True)
                    vertices = vertices[indices]

                return class_dir.name, vertices

    return None, None


def run_vulkan_benchmark():
    """Run Vulkan benchmark and parse results (uses release build)"""
    # Use release build for fair comparison with PyTorch
    vulkan_exe = '/home/jeong/workspace/vai-samples/bin/release/104-PointNet-Jeonghan'

    # Fallback to debug if release doesn't exist
    if not Path(vulkan_exe).exists():
        vulkan_exe = '/home/jeong/workspace/vai-samples/bin/debug/104-PointNet-Jeonghan'
        print(f"  Warning: Using debug build (run ./build.sh --release for fair comparison)")

    proc = subprocess.Popen(
        [vulkan_exe],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = proc.communicate(input='1\n', timeout=60)
    output = stdout + stderr

    # Parse results
    avg_match = re.search(r'Average time\s+│\s+([0-9.]+)', output)
    min_match = re.search(r'Min time\s+│\s+([0-9.]+)', output)
    max_match = re.search(r'Max time\s+│\s+([0-9.]+)', output)
    throughput_match = re.search(r'Throughput\s+│\s+([0-9]+)', output)

    return {
        'avg_ms': float(avg_match.group(1)) if avg_match else None,
        'min_ms': float(min_match.group(1)) if min_match else None,
        'max_ms': float(max_match.group(1)) if max_match else None,
        'throughput': int(throughput_match.group(1)) if throughput_match else None
    }


def run_pytorch_benchmark(model, point_cloud, num_iterations=10, device='cuda'):
    """Run PyTorch benchmark - includes data preprocessing for fair comparison with Vulkan"""
    model.eval()

    # Warmup (3 iterations) - same as Vulkan
    print("  Warmup (3 iterations)...")
    with torch.no_grad():
        for _ in range(3):
            # Include preprocessing in warmup too
            input_9ch = np.zeros((1, 9, point_cloud.shape[0]), dtype=np.float32)
            input_9ch[0, :3, :] = point_cloud.T
            input_9ch[0, 3:6, :] = 1.0
            input_9ch[0, 6:9, :] = point_cloud.T
            x = torch.from_numpy(input_9ch).to(device)
            _ = model(x)
    print("  Done")

    # Benchmark - include preprocessing time for fair comparison
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            if device == 'cuda':
                torch.cuda.synchronize()

            start = time.perf_counter()

            # Data preprocessing (same as Vulkan segment() function)
            input_9ch = np.zeros((1, 9, point_cloud.shape[0]), dtype=np.float32)
            input_9ch[0, :3, :] = point_cloud.T  # xyz
            input_9ch[0, 3:6, :] = 1.0  # rgb = 1
            input_9ch[0, 6:9, :] = point_cloud.T  # normalized xyz
            x = torch.from_numpy(input_9ch).to(device)

            # Forward pass
            output = model(x)

            if device == 'cuda':
                torch.cuda.synchronize()

            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

    return {
        'avg_ms': np.mean(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'throughput': int(point_cloud.shape[0] / (np.mean(times) / 1000.0))
    }


def main():
    print()
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║       PointNet Benchmark: Vulkan vs PyTorch Comparison        ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()

    # Configuration
    num_classes = 13
    channel = 9
    num_points = 1024
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Configuration:")
    print(f"  Points:  {num_points}")
    print(f"  Classes: {num_classes}")
    print(f"  Device:  {device}")
    print()

    # Load sample
    modelnet_path = "assets/datasets/ModelNet40"
    class_name, point_cloud = load_modelnet40_sample(modelnet_path, num_points)

    if point_cloud is None:
        print(f"Error: Could not load ModelNet40 from {modelnet_path}")
        return

    print(f"Sample: {class_name} ({point_cloud.shape[0]} points)")
    print()

    # ============================================================
    # Vulkan Benchmark
    # ============================================================
    print("Running Vulkan benchmark...")
    vulkan_results = run_vulkan_benchmark()
    print(f"  Done: {vulkan_results['avg_ms']:.2f} ms avg")
    print()

    # ============================================================
    # PyTorch Benchmark
    # ============================================================
    print("Running PyTorch benchmark...")

    # Load model with pretrained weights
    weights_path = Path("assets/weights/best_model.pth")
    model = PointNetSegment(num_classes=num_classes, channel=channel).to(device)

    if weights_path.exists():
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        # Handle both checkpoint format and direct state_dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"  Loaded weights: {weights_path}")
    else:
        print(f"  Warning: Using random weights (no pretrained model)")

    pytorch_results = run_pytorch_benchmark(model, point_cloud, num_iterations=10, device=device)
    print(f"  Done: {pytorch_results['avg_ms']:.2f} ms avg")
    print()

    # ============================================================
    # Comparison Results
    # ============================================================
    print("═" * 66)
    print("                        COMPARISON RESULTS")
    print("═" * 66)
    print()

    # Calculate speedup
    speedup = pytorch_results['avg_ms'] / vulkan_results['avg_ms'] if vulkan_results['avg_ms'] else 0

    # Fixed column width: 17 chars for value column content
    # time rows:  "    12345.67 ms" = 13 digits + " ms" = 16, +1 space = 17
    # tp rows:    "    123456 pts/s" = 10 digits + " pts/s" = 16, +1 space = 17
    W_VAL = 17

    print("┌────────────────────┬───────────────────┬───────────────────┬──────────┐")
    print("│ Metric             │ Vulkan            │ PyTorch           │ Speedup  │")
    print("├────────────────────┼───────────────────┼───────────────────┼──────────┤")

    # Average time
    v_avg = vulkan_results['avg_ms']
    p_avg = pytorch_results['avg_ms']
    v_str = f"{v_avg:>10.2f} ms"
    p_str = f"{p_avg:>10.2f} ms"
    print(f"│ Average time       │ {v_str:>{W_VAL}} │ {p_str:>{W_VAL}} │ {speedup:>6.2f}x  │")

    # Min time
    v_min = vulkan_results['min_ms']
    p_min = pytorch_results['min_ms']
    min_speedup = p_min / v_min if v_min else 0
    v_str = f"{v_min:>10.2f} ms"
    p_str = f"{p_min:>10.2f} ms"
    print(f"│ Min time           │ {v_str:>{W_VAL}} │ {p_str:>{W_VAL}} │ {min_speedup:>6.2f}x  │")

    # Max time
    v_max = vulkan_results['max_ms']
    p_max = pytorch_results['max_ms']
    max_speedup = p_max / v_max if v_max else 0
    v_str = f"{v_max:>10.2f} ms"
    p_str = f"{p_max:>10.2f} ms"
    print(f"│ Max time           │ {v_str:>{W_VAL}} │ {p_str:>{W_VAL}} │ {max_speedup:>6.2f}x  │")

    # Throughput
    v_tp = vulkan_results['throughput']
    p_tp = pytorch_results['throughput']
    tp_ratio = v_tp / p_tp if p_tp else 0
    v_str = f"{v_tp:>10} pts/s"
    p_str = f"{p_tp:>10} pts/s"
    print(f"│ Throughput         │ {v_str:>{W_VAL}} │ {p_str:>{W_VAL}} │ {tp_ratio:>6.2f}x  │")

    print("└────────────────────┴───────────────────┴───────────────────┴──────────┘")
    print()

    # Summary
    if speedup > 1:
        print(f"Result: Vulkan is {speedup:.2f}x faster than PyTorch ({device})")
    else:
        print(f"Result: PyTorch ({device}) is {1/speedup:.2f}x faster than Vulkan")
    print()


if __name__ == '__main__':
    main()

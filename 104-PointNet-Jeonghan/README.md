# PointNet Segmentation - Vulkan Implementation

Vulkan compute shader 기반 PointNet semantic segmentation 구현.

**Reference**: [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

## Build & Run

```bash
./build.sh
python3 benchmark_pytorch.py
```

## Performance

| Metric | Vulkan | PyTorch | Speedup |
|--------|--------|---------|---------|
| Average time | 6.51 ms | 1.34 ms | 0.21x |
| Min time | 6.10 ms | 1.07 ms | 0.18x |
| Max time | 6.84 ms | 2.05 ms | 0.30x |
| Throughput | 157,333 pts/s | 761,987 pts/s | 0.21x |

*Tested: 1024 points, 13 classes*

## Architecture

```
Input [9, N]
    ↓
PointNetEncoder
├── STN3d (9 → 3×3)
├── Conv1 (9 → 64) + BN + ReLU
├── FSTN (64 → 64×64)
├── Conv2 (64 → 128) + BN + ReLU
└── Conv3 (128 → 1024) + BN
    ↓
├── MaxPool → [1024]
│       ↓
│   Broadcast → [1024, N]
└── Concat → [1088, N]
        ↓
    SegmentationHead
    ├── Conv (1088 → 512) + BN + ReLU
    ├── Conv (512 → 256) + BN + ReLU
    ├── Conv (256 → 128) + BN + ReLU
    └── Conv (128 → 13)
        ↓
    Output [13, N]
```

## Available Nodes

### MLP / Convolution Nodes

| Node | Description |
|------|-------------|
| `PointWiseMLPNode` | Conv1x1 + BatchNorm + ReLU (3 kernels) |
| `FusedPointWiseMLPNode` | Conv1x1 + BatchNorm + ReLU (1 fused kernel) |
| `TiledFusedPointWiseMLPNode` | Shared memory tiled version (Cin,Cout ≥ 64) |
| `PointWiseConvNode` | Conv1x1 + BatchNorm (no ReLU, 2 kernels) |
| `FusedPointWiseConvNode` | Conv1x1 + BatchNorm (1 fused kernel) |
| `PointWiseLinearNode` | Conv1x1 only (final output layer) |

### Pooling Nodes

| Node | Description |
|------|-------------|
| `MaxPooling1DNode` | Global max pooling [C, N] → [C, 1] |
| `TreeMaxPooling1DNode` | Tree reduction max pooling O(log N) |
| `MaxPoolingNode` | Window-based max pooling |

### Tensor Operation Nodes

| Node | Description |
|------|-------------|
| `BroadcastNode` | [C, 1] → [C, N] |
| `ConcatNode` | [C1, N] + [C2, N] → [C1+C2, N] |
| `SliceNode` | [C, N] → [slice, N] (channel slice) |
| `MatMulNode` | [N, K] @ [K, M] → [N, M] |

### Other Nodes

| Node | Description |
|------|-------------|
| `BatchNorm1DNode` | Batch normalization |
| `ReluNode` | ReLU activation |
| `FullyConnectedNode` | Fully connected layer |
| `FlattenNode` | Flatten tensor |
| `ReShapeNode` | Reshape tensor |
| `AddIdentityNode` | Add identity matrix (TNet) |
| `IdentityNode` | Pass-through / signal split |

## Testing

```bash
./benchmark.sh              # All tests
python3 benchmark_pytorch.py  # Vulkan vs PyTorch
```

## References

- [PointNet Paper](https://arxiv.org/abs/1612.00593)
- [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

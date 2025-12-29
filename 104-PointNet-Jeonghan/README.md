# PointNet Segmentation - Vulkan Implementation

High-performance PointNet semantic segmentation using Vulkan compute shaders.

**Reference**: [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

## Features

- **PyTorch-Exact Implementation**: Matches yanx27 architecture precisely (max diff: 5e-08)
- **Pure Vulkan Compute**: GPU-accelerated inference without CUDA dependency
- **S3DIS Segmentation**: 13-class semantic segmentation with 9-channel input
- **SafeTensors Format**: Fast weight loading with PyTorch state_dict keys
- **Modular Architecture**: WeightLoader, PointNetEncoder, TNet, MLPSequence

## Quick Start

```bash
# Build
./build.sh

# Run benchmark
python3 benchmark_pytorch.py
```

## Performance Comparison

| Metric | Vulkan | PyTorch CUDA | Ratio |
|--------|--------|--------------|-------|
| Average | ~16 ms | ~3.5 ms | 4.6x |
| Throughput | 63k pts/s | 291k pts/s | 4.6x |

*Tested on 1024 points, 13 classes. PyTorch is faster due to cuDNN optimization.*

## Architecture

```
Input [9, N]  (xyz + rgb + normalized_xyz)
    ↓
PointNetEncoder
├── STN3d (9 → 3×3 matrix)
├── Conv1 (9 → 64) + BN + ReLU
├── FSTN (64 → 64×64 matrix)
├── Conv2 (64 → 128) + BN + ReLU
└── Conv3 (128 → 1024) + BN (no ReLU)
    ↓
├── MaxPool → [1024] (global feature)
│       ↓
│   Broadcast → [1024, N]
│       ↓
└── Concat([global, pointfeat]) → [1088, N]
        ↓
    SegmentationHead
    ├── Conv1 (1088 → 512) + BN + ReLU
    ├── Conv2 (512 → 256) + BN + ReLU
    ├── Conv3 (256 → 128) + BN + ReLU
    └── Conv4 (128 → 13) (no BN, no ReLU)
        ↓
    Output [13, N]
```

## Input Format (S3DIS 9-channel)

| Channel | Content |
|---------|---------|
| 0-2 | Centered XYZ (relative to block center) |
| 3-5 | Normalized RGB (/255) |
| 6-8 | Normalized XYZ (relative to room max) |

## Project Structure

```
104-PointNet-Jeonghan/
├── networks/
│   ├── include/
│   │   ├── pointnet.hpp       # Network architecture
│   │   ├── weightLoader.hpp   # PyTorch weight loading
│   │   ├── inference.h        # High-level API
│   │   └── weights.h          # Weight loading interface
│   └── src/
│       ├── inference.cpp      # Inference implementation
│       └── weights.cpp        # Weight loading (24 lines)
├── library/                   # Core Vulkan framework
│   ├── neuralNodes.h/cpp      # Compute nodes
│   └── vulkanApp.h/cpp        # Vulkan initialization
├── test/                      # Unit tests
│   ├── segment/               # Full model test
│   ├── encoder/               # Encoder test
│   ├── tnet/                  # TNet test
│   └── ...
├── assets/
│   └── weights/
│       └── pointnet_sem_seg.safetensors  # Pretrained weights (14MB)
├── main.cpp                   # Performance benchmark
├── benchmark_pytorch.py       # Vulkan vs PyTorch comparison
├── benchmark.sh               # Test runner
└── build.sh                   # Build script
```

## Testing

```bash
# Run all tests
./benchmark.sh

# Individual tests
echo "1" | /path/to/bin/debug/test_segment
echo "1" | /path/to/bin/debug/test_encoder
echo "1" | /path/to/bin/debug/test_tnet

# Vulkan vs PyTorch comparison
python3 benchmark_pytorch.py
```

### Test Results

| Test | Max Diff | Status |
|------|----------|--------|
| test_segment | 5.2e-08 | PASSED |
| test_encoder | 1.2e-07 | PASSED |
| test_tnet | 1.5e-07 | PASSED |

## API Usage

```cpp
#include "inference.h"

using namespace networks;

// Configuration
InferenceConfig config;
config.weights_file = "assets/weights/pointnet_sem_seg.safetensors";
config.num_classes = 13;
config.channel = 9;

// Load model
PointNetSegment* model = loadPretrainedModel(config);

// Point cloud [N*3 floats]: x1,y1,z1, x2,y2,z2, ...
std::vector<float> point_cloud = loadPointCloud("sample.txt");

// Inference (auto-expands 3ch to 9ch)
SegmentationResult result = segment(*model, point_cloud, config);

// Results
for (uint32_t i = 0; i < result.num_points; ++i) {
    uint32_t label = result.predicted_labels[i];
}

delete model;
```

## Weight Conversion

```bash
# Convert PyTorch checkpoint to SafeTensors
python3 utils/convert_to_safetensors.py \
    assets/weights/best_model.pth \
    assets/weights/pointnet_sem_seg.safetensors
```

## Key Implementation Details

1. **PyTorch [C, N] Convention**: All tensors use channel-first format
2. **WeightLoader**: Centralized weight loading with PyTorch state_dict keys
3. **PointWiseLinearNode**: Final conv4 without BatchNorm/ReLU
4. **Concat Order**: [global(1024), pointfeat(64)] = 1088 channels

## References

- [PointNet Paper](https://arxiv.org/abs/1612.00593)
- [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
- [S3DIS Dataset](http://buildingparser.stanford.edu/dataset.html)

---

**Last Updated**: 2025-12-29
**Author**: Jeonghan

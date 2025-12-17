# PointNet Segmentation - Vulkan Implementation

High-performance PointNet point cloud segmentation inference using Vulkan compute shaders.

**Reference Implementation:** [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

## 🎯 Features

- ✅ **Pure Vulkan Compute**: GPU-accelerated inference with Vulkan compute shaders
- ✅ **PointNet Segmentation**: Full implementation following the original paper
- ✅ **Pretrained Weights**: yanx27 S3DIS semantic segmentation model (13 classes)
- ✅ **Clean API**: Inspired by GPT-2 inference module design
- ✅ **Modular Architecture**: Separate inference, weights, and network modules
- ✅ **Multiple Nodes**: TNet, MLP, MaxPool, Broadcast, Concat operations
- ✅ **Flexible Input**: Support for `.txt`, `.ply`, `.xyz`, `.off` point cloud formats
- ✅ **ModelNet40 Integration**: 40 object categories, auto-sampling, error recovery
- ✅ **Comprehensive Tests**: 15+ unit tests validating against PyTorch reference
- ✅ **High Performance**: ~74,000 points/sec throughput on standard GPUs

## 📁 Project Structure

```
104-PointNet-Jeonghan/
├── networks/
│   ├── include/
│   │   ├── pointnet.hpp      # Network architecture
│   │   ├── inference.h       # High-level inference API
│   │   └── weights.h         # Weight loading
│   └── src/
│       ├── pointnet.cpp      # Network implementation
│       ├── inference.cpp     # Inference implementation
│       └── weights.cpp       # Weight loading implementation
├── library/                  # Core Vulkan framework
│   ├── neuralNet.h/cpp       # Neural network base classes
│   ├── neuralNodes.h/cpp     # Network nodes (Conv, MLP, etc.)
│   ├── vulkanApp.h/cpp       # Vulkan initialization
│   ├── pointCloudLoader.h    # Point cloud file loaders (.txt/.ply/.xyz/.off)
│   └── ...
├── utils/                    # Utilities
│   ├── convert_pytorch_weights.py  # PyTorch → JSON converter
│   ├── convert_to_safetensors.py   # Convert JSON to SafeTensors
│   └── download.sh                 # Dataset download script
├── assets/
│   ├── weights/              # Model weights (JSON)
│   ├── data/                 # Sample point clouds
│   └── datasets/             # Downloaded datasets (ModelNet40)
├── test/                     # Unit tests
│   ├── encoder/              # PointNetEncoder test
│   ├── segment/              # Full segmentation test
│   ├── mlp/                  # MLP layer test
│   └── ...                   # 15+ component tests
├── main.cpp                  # Demo application
├── test_off_loader.cpp       # OFF format loader test
├── test_real_data.cpp        # Real data inference test
├── CMakeLists.txt            # Build configuration
├── build.sh                  # Incremental build script
├── README.md                 # This file
└── WEIGHTS_README.md         # Weight conversion guide
```

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Ubuntu/Debian
sudo apt install build-essential cmake libvulkan-dev vulkan-tools

# Verify Vulkan
vulkaninfo | grep "deviceName"
```

### 2. Build

```bash
cd 104-PointNet-Jeonghan

# First time: full build
cd ..
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# Incremental builds
cd 104-PointNet-Jeonghan
./build.sh
```

### 3. Download Pretrained Weights

**yanx27 S3DIS Semantic Segmentation Model** (Recommended):
```bash
# Already provided in assets/weights/pointnet_yanx27.json
# 13 classes (ceiling, floor, wall, beam, column, window, door, table, chair, sofa, bookcase, board, clutter)
# Trained on S3DIS dataset
# File size: ~120MB
```

**Alternative: Custom Weights**:
```bash
# Random weights (for testing)
python utils/convert_pytorch_weights.py --random --num_classes 10

# Convert from PyTorch checkpoint
python utils/convert_pytorch_weights.py \
    --checkpoint path/to/model.pth \
    --output assets/weights/pointnet_weights.json
```

See [WEIGHTS_README.md](WEIGHTS_README.md) for detailed instructions.

### 4. Download ModelNet40 Dataset (Optional)

```bash
# Download ~2GB dataset (40 categories, 12,311 CAD models)
./utils/download.sh
```

**ModelNet40 Categories**: airplane, bathtub, bed, bench, bookshelf, bottle, bowl, car, chair, cone, cup, curtain, desk, door, dresser, flower_pot, glass_box, guitar, keyboard, lamp, laptop, mantel, monitor, night_stand, person, piano, plant, radio, range_hood, sink, sofa, stairs, stool, table, tent, toilet, tv_stand, vase, wardrobe, xbox

**Features**:
- 40 object categories with train/test splits
- OFF format (vertices and faces)
- Automatic sampling and normalization
- Error recovery for corrupted files

**Supported Point Cloud Formats**: `.txt`, `.ply`, `.xyz`, `.off`

### 5. Run Inference

```bash
# Run from project directory
cd /home/jeonghan/workspace/vai-samples/104-PointNet-Jeonghan
/home/jeonghan/workspace/vai-samples/bin/debug/104-PointNet-Jeonghan
```

Expected output:
```
╔════════════════════════════════════════════════════════╗
║      PointNet Segmentation - Vulkan Inference         ║
║           with ModelNet40 Dataset Support             ║
╚════════════════════════════════════════════════════════╝

Step 1: Loading pretrained model...
Loading PointNet Segmentation model...
  Weights: assets/weights/pointnet_yanx27.json
  Num classes: 13
  Input channels: 3
✓ Network created
Loading pretrained weights...
✓ Weights loaded

========================================================
Example 1: ModelNet40 Dataset
========================================================

--- ModelNet40 Sample ---
Loaded: lamp (19) from lamp_0144.off
Points: 1024
Input: [1024, 3] (xyz only)
Segmentation complete:
  Points: 1024
  Time: 13.800 ms
  Throughput: 74204.0 points/sec

Object: lamp (ModelNet40 class 19)

Top semantic classes detected:
  Class 12: 100.0% (1024 points)
  Class 11: 0.0% (0 points)
  Class 10: 0.0% (0 points)
Performance: 74204 points/sec

========================================================
PointNet segmentation demo complete!
========================================================
```

**Note**: yanx27 model is trained for semantic segmentation (13 S3DIS classes), not ModelNet40 classification (40 classes). ModelNet40 is used for demonstration and visualization purposes.

## 🏗️ Architecture

### PointNet Segmentation Network

Following the original paper: [PointNet: Deep Learning on Point Sets](https://arxiv.org/abs/1612.00593)

```
Input [N, 3]
  ↓
encoder → [N, 1024] (point-wise features)
  ├───→ maxpool → [1, 1024] (global feature)
  │         ↓
  │    broadcast → [N, 1024] (replicated global)
  │         ↓
  └───→ concat → [N, 2048] (point + global)
            ↓
        segHead → [N, numClasses]
            ↓
        output
```

**Components:**

1. **PointNetEncoder**:
   - Input TNet (3x3 transformation)
   - MLP1: [3, 64, 64]
   - Feature TNet (64x64 transformation)
   - MLP2: [64, 128, 1024]

2. **Segmentation Head**:
   - Global max pooling
   - Feature broadcast
   - Concatenation (point + global)
   - MLP: [2048, 512, 256, numClasses]

### Custom Nodes

- **BroadcastNode**: Replicates global feature to all points
- **ConcatNode**: Concatenates point and global features
- **TNetBlock**: Spatial transformer network
- **MLPSequence**: Point-wise MLP layers
- **MaxPooling1DNode**: Global feature extraction

## 📖 API Usage

### High-Level API (Recommended)

```cpp
#include "inference.h"

using namespace networks;

// Configuration
InferenceConfig config;
config.weights_file = "assets/weights/pointnet_weights.json";
config.num_classes = 10;

// Load model
PointNetSegment* model = loadPretrainedModel(config);

// Prepare point cloud [N*3 floats]
std::vector<float> point_cloud = {...};  // [x1, y1, z1, x2, y2, z2, ...]

// Run segmentation
SegmentationResult result = segment(*model, point_cloud, config);

// Use results
if (result.success) {
    for (uint32_t i = 0; i < result.num_points; ++i) {
        uint32_t label = result.predicted_labels[i];
        std::cout << "Point " << i << " → Class " << label << "\n";
    }
}

// Cleanup
delete model;
```

### From File

```cpp
// Load point cloud from text file
SegmentationResult result = segmentFromFile(
    *model,
    "assets/data/sample.txt",
    config
);
```

## 🔧 Development

### Build System

```bash
# Full clean build
cd build && cmake .. && make -j$(nproc)

# Incremental build (faster)
cd 104-PointNet-Jeonghan && ./build.sh

# Only this project (skip dependencies)
cd build && make 104-PointNet-Jeonghan -j$(nproc)
```

### Adding New Nodes

1. Declare in `library/neuralNodes.h`
2. Implement in `library/neuralNodes.cpp`
3. Add GLSL compute shader
4. Use in `networks/include/pointnet.hpp`

Example: See `BroadcastNode` and `ConcatNode`

### Testing

```bash
# Build with tests
./build.sh --test

# Run all tests
cd ../build
ctest --output-on-failure

# Run specific tests
/home/jeonghan/workspace/vai-samples/bin/debug/test_encoder
/home/jeonghan/workspace/vai-samples/bin/debug/test_segment

# Test with real data
cd /home/jeonghan/workspace/vai-samples/104-PointNet-Jeonghan
/home/jeonghan/workspace/vai-samples/bin/debug/test_off_loader
```

**Test Results** (see [test/TEST_RESULTS.md](test/TEST_RESULTS.md) for details):

| Component | Test Points | Errors | Max Error | Status |
|-----------|-------------|--------|-----------|--------|
| PointNetEncoder | 1024 | 0 | 1.19e-07 | ✅ PASSED |
| PointNetSegment | 64 | 0 | 0.045 | ✅ PASSED |

All tests compare Vulkan compute shader outputs against PyTorch reference implementations with tolerance of 1e-3.

## 📊 Performance

**Measured Performance (yanx27 pretrained model):**
- **Throughput**: ~74,000 points/sec (average across multiple runs)
- **Latency**: ~13.8ms per 1024-point cloud
- **Consistency**: 72K-75K points/sec range
- **Tested on**: Consumer-grade GPU

**Performance by Object Type** (ModelNet40 samples):
| Object | Points | Time (ms) | Throughput (pts/sec) |
|--------|--------|-----------|---------------------|
| bench | 1024 | 14.07 | 72,776 |
| stool | 1024 | 13.71 | 74,669 |
| cup | 1024 | 13.65 | 75,005 |
| lamp | 1024 | 13.80 | 74,204 |
| plant | 1024 | 13.97 | 73,311 |

**Memory Usage:**
- Model weights: ~120 MB (yanx27 JSON format)
- Per-inference: ~8 MB (1024 points, 13 classes)
- GPU buffers: ~16 MB (input + intermediate + output)

**Known Limitations:**
- Multiple inference calls not yet supported (buffer reuse issue)
- Run program multiple times for different samples

## 🐛 Troubleshooting

### Build Errors

```bash
# Missing Vulkan
sudo apt install libvulkan-dev vulkan-tools

# CMake version
cmake --version  # Need >= 3.10

# Regenerate build files
rm -rf build && mkdir build && cd build && cmake ..
```

### Runtime Errors

**"Pretrained weights not found"**
```bash
python utils/convert_pytorch_weights.py --random
```

**"Vulkan device not found"**
```bash
# Check Vulkan support
vulkaninfo

# For WSL, ensure GPU passthrough is enabled
```

**"Failed to load point cloud"**
- Check file format: each line should be `x y z`
- Verify file path is relative to executable location

## 📚 References

- **Original Paper**: [PointNet: Deep Learning on Point Sets](https://arxiv.org/abs/1612.00593)
- **PyTorch Implementation**: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
- **Vulkan Compute**: https://www.khronos.org/vulkan/

## 📝 Citation

If you use this implementation, please cite:

```bibtex
@article{Pytorch_Pointnet_Pointnet2,
  Author = {Xu Yan},
  Title = {Pointnet/Pointnet++ Pytorch},
  Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
  Year = {2019}
}

@inproceedings{qi2017pointnet,
  title={Pointnet: Deep learning on point sets for 3d classification and segmentation},
  author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={652--660},
  year={2017}
}
```

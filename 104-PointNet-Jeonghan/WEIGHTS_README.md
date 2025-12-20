# PointNet Segmentation - Weights Guide

This document explains how to obtain and convert PointNet weights for Vulkan inference.

## 📋 Quick Start

### Option 1: Download Pretrained Weights (Recommended)

Use the automated download script to get pretrained PointNet2 weights:

```bash
# Download and convert pretrained weights automatically
cd utils
./download_weight.sh
```

This script will:
- ✅ Download `best_model.pth` from [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) (~21MB)
- ✅ Automatically convert to SafeTensors format
- ✅ Save to `assets/weights/pointnet_sem_seg.safetensors`

**Model Details:**
- Source: yanx27/Pointnet_Pointnet2_pytorch
- Task: Part Segmentation (Multi-Scale Grouping)
- Pretrained: ShapeNet Part Dataset

### Option 2: Use Random Weights (For Testing)

```bash
# Generate random weights for quick testing
python utils/convert_pytorch_weights.py --random --num_classes 10

# This creates: assets/weights/pointnet_weights.json
```

### Option 3: Convert Your Own PyTorch Checkpoint

```bash
# If you have a trained PyTorch model
python utils/convert_pytorch_weights.py \
    --checkpoint path/to/model.pth \
    --output assets/weights/pointnet_weights.json \
    --num_classes 10
```

## 🏗️ Weight Format

Our JSON weight format follows this structure:

```json
{
  "tnet1.mlp.0.weight": [...],  // Input TNet MLP layers
  "tnet1.mlp.0.bias": [...],
  "tnet1.fc.0.weight": [...],   // Input TNet FC layers
  ...
  "mlp1.0.weight": [...],        // First feature extraction
  "mlp1.1.weight": [...],
  "tnet2.mlp.0.weight": [...],  // Feature TNet
  ...
  "mlp2.0.weight": [...],        // Second feature extraction
  "mlp2.1.weight": [...],
  "segHead.0.weight": [...],     // Segmentation head
  "segHead.1.weight": [...],
  "segHead.2.weight": [...]
}
```

### Architecture Details

**PointNet Segmentation Network:**

1. **Input Transform (TNet1)**: 3x3 transformation
   - MLP: [3, 64, 128, 1024]
   - FC: [1024, 512, 256, 9]
   
2. **Feature Extraction (MLP1)**: [3, 64, 64]

3. **Feature Transform (TNet2)**: 64x64 transformation
   - MLP: [64, 64, 128, 1024]
   - FC: [1024, 512, 256, 4096]

4. **Feature Extraction (MLP2)**: [64, 128, 1024]

5. **Global Feature**: Max pooling → [1, 1024]

6. **Segmentation Head**: [2048, 512, 256, num_classes]
   - Input: Concatenation of point features [N, 1024] + global feature [N, 1024]

## 🔧 Download & Conversion Scripts

### Automated Weight Download

```bash
# Download pretrained weights and convert automatically
cd utils
./download_weight.sh
```

Features:
- Downloads PointNet2 Part Seg pretrained weights from GitHub
- Interactive conversion to SafeTensors format
- Verifies file integrity
- Shows file size and location

### Manual Conversion

```bash
# Basic usage with random weights
python utils/convert_pytorch_weights.py --random

# Specify number of classes
python utils/convert_pytorch_weights.py --random --num_classes 50

# Convert from checkpoint
python utils/convert_pytorch_weights.py \
    --checkpoint model.pth \
    --output weights.json \
    --num_classes 10
```

### Supported PyTorch Model Structures

The converter supports models from:
- [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) ✅
- [fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch)

The script automatically handles:
- `model_state_dict` or `state_dict` keys
- Module prefixes (`module.`, `model.`)
- Different layer naming conventions

## 📊 Expected Weight Sizes

| Component       | Parameters | Approx. Size |
|-----------------|------------|--------------|
| TNet1 (MLP)     | ~270K      | 1.0 MB       |
| TNet1 (FC)      | ~1.8M      | 7.0 MB       |
| MLP1            | ~4K        | 0.02 MB      |
| TNet2 (MLP)     | ~270K      | 1.0 MB       |
| TNet2 (FC)      | ~5.5M      | 22.0 MB      |
| MLP2            | ~140K      | 0.5 MB       |
| Seg Head        | ~1.4M      | 5.5 MB       |
| **Total**       | **~9.4M**  | **~37 MB**   |

*Size may vary based on number of classes*

## 🎯 Verification

After conversion, verify the weights:

```bash
# Check file size
ls -lh assets/weights/pointnet_weights.json

# Check JSON structure
python -c "import json; w = json.load(open('assets/weights/pointnet_weights.json')); print(f'Found {len(w)} weight tensors')"

# Run inference to test
cd build/bin/debug
./104-PointNet-Jeonghan
```

## 📥 Dataset Download

Download ModelNet40 dataset for testing:

```bash
cd utils
./download.sh
```

This downloads and extracts ModelNet40 to `assets/datasets/ModelNet40/`.

## 📚 Additional Resources

- **Original Paper**: [PointNet: Deep Learning on Point Sets](https://arxiv.org/abs/1612.00593)
- **PyTorch Implementation**: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
- **Pretrained Models**: Available via `./utils/download_weight.sh`
- **ModelNet40 Dataset**: https://modelnet.cs.princeton.edu/
- **ShapeNet Dataset**: https://shapenet.org/

## ❓ Troubleshooting

### "Key not found" warnings during conversion
- Some PyTorch models use different naming conventions
- Check the model structure with: `print(model.state_dict().keys())`
- You may need to modify the `key_mapping` in the converter script

### JSON file too large
- Weight files are typically 30-50 MB (normal)
- Use compression if needed: `gzip pointnet_weights.json`

### Model not loading in inference
- Verify JSON is valid: `python -m json.tool weights.json > /dev/null`
- Check number of classes matches: `--num_classes` in converter
- Ensure all required keys are present

## 🚀 Complete Workflow

```bash
# 1. Download pretrained weights
cd utils
./download_weight.sh

# 2. Download ModelNet40 dataset (optional)
./download.sh

# 3. Build and run inference
cd ..
./build.sh
../bin/debug/104-PointNet-Jeonghan
```

## 🔄 Updates

Last updated: 2025-12-20

Automated scripts:
- `utils/download_weight.sh` - Download pretrained PointNet2 weights
- `utils/download.sh` - Download ModelNet40 dataset
- `utils/convert_pytorch_weights.py` - Convert PyTorch checkpoints


# PointNet Segmentation - Weights Guide

This document explains how to obtain and convert PointNet weights for Vulkan inference.

## 📋 Quick Start

### Option 1: Use Random Weights (For Testing)

```bash
# Generate random weights for quick testing
python utils/convert_pytorch_weights.py --random --num_classes 10

# This creates: assets/weights/pointnet_weights.json
```

### Option 2: Convert from PyTorch Checkpoint

```bash
# If you have a trained PyTorch model
python utils/convert_pytorch_weights.py \
    --checkpoint path/to/model.pth \
    --output assets/weights/pointnet_weights.json \
    --num_classes 10
```

### Option 3: Train Your Own Model

Follow the PyTorch reference implementation to train a model:

**Reference:** https://github.com/yanx27/Pointnet_Pointnet2_pytorch

```bash
# Clone the reference repo
git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git
cd Pointnet_Pointnet2_pytorch

# Install dependencies
conda create -n pointnet python=3.7
conda activate pointnet
conda install pytorch==1.6.0 cudatoolkit=10.1 -c pytorch

# Download ModelNet40 dataset
# (See their README for detailed instructions)

# Train PointNet for classification
python train_classification.py --model pointnet_cls --log_dir pointnet_cls

# Or train for part segmentation
python train_partseg.py --model pointnet_part_seg --log_dir pointnet_part_seg

# Convert the trained model
cd ../104-PointNet-Jeonghan
python utils/convert_pytorch_weights.py \
    --checkpoint ../Pointnet_Pointnet2_pytorch/log/pointnet_part_seg/checkpoints/best_model.pth \
    --output assets/weights/pointnet_weights.json
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

## 🔧 Conversion Script Usage

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
- [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
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

## 📚 Additional Resources

- **Original Paper**: [PointNet: Deep Learning on Point Sets](https://arxiv.org/abs/1612.00593)
- **PyTorch Implementation**: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
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

## 🔄 Updates

Last updated: 2025-01-23

For the latest version of this guide, see:
https://github.com/yanx27/Pointnet_Pointnet2_pytorch


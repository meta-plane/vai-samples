# AGENTS.md - PointNet Vulkan Implementation

## Project Context

PointNet Semantic Segmentation의 Vulkan Compute Shader 구현.
yanx27/Pointnet_Pointnet2_pytorch 모델을 PyTorch와 동일하게 구현.

**현재 상태**: PyTorch와 완전 일치 (max diff: 5e-08)

### Tech Stack

- Language: C++23
- Build: CMake 3.16+, GCC 13+
- GPU: Vulkan 1.3 SDK
- Weights: SafeTensors (PyTorch state_dict keys)
- Reference: PyTorch (yanx27/Pointnet_Pointnet2_pytorch)

### Architecture (PyTorch Convention)

```
Input [C, N] = [9, 1024]  (PyTorch channel-first)
    ↓
PointNetEncoder
├── STN3d (9ch → 3×3 matrix)
├── MatMul (xyz only)
├── Conv1 [9→64] + BN + ReLU
├── FSTN (64ch → 64×64 matrix)
├── MatMul
├── Conv2 [64→128] + BN + ReLU
└── Conv3 [128→1024] + BN (NO ReLU)
    ↓
├── MaxPool → [1024]
├── Broadcast → [1024, N]
└── Concat([global, pointfeat]) → [1088, N]
        ↓
    SegmentationHead
    ├── Conv1 [1088→512] + BN + ReLU
    ├── Conv2 [512→256] + BN + ReLU
    ├── Conv3 [256→128] + BN + ReLU
    └── Conv4 [128→13] (NO BN, NO ReLU)
        ↓
    Output [13, N]
```

## Operational Commands

```bash
# Build
./build.sh

# Run main benchmark (requires input "1" for GPU selection)
echo "1" | ../bin/debug/104-PointNet-Jeonghan

# Run Vulkan vs PyTorch comparison
python3 benchmark_pytorch.py

# Run test suite
./benchmark.sh

# Individual tests
echo "1" | ../bin/debug/test_segment
echo "1" | ../bin/debug/test_encoder
echo "1" | ../bin/debug/test_tnet

# Generate PyTorch reference
cd test/segment && python3 generate_reference.py
```

## Golden Rules

### PyTorch Convention (CRITICAL)

1. **Tensor Shape**: `[C, N]` (channel-first, NOT `[N, C]`)
2. **Weight Keys**: PyTorch state_dict keys preserved (e.g., `feat.stn.conv1.weight`)
3. **Concat Order**: `[global(1024), pointfeat(64)]` = 1088 channels
4. **Conv3 in Encoder**: BatchNorm but NO ReLU (matches PyTorch)
5. **Conv4 in SegHead**: NO BatchNorm, NO ReLU (matches PyTorch)

### Do's

- Node 추가 시 `neuralNodes.h/cpp`에 구현
- 새 기능 추가 전 PyTorch reference 먼저 작성
- WeightLoader 사용하여 weight 로딩
- 테스트 허용 오차: epsilon = 1e-3

### Don'ts

- `[N, C]` 형식 사용 금지 (PyTorch는 `[C, N]`)
- JSON weight format 사용 금지 (SafeTensors만 사용)
- test 없이 커밋 금지

## Key Files

| File | Purpose |
|------|---------|
| `networks/include/pointnet.hpp` | Network architecture |
| `networks/include/weightLoader.hpp` | PyTorch weight loading |
| `library/neuralNodes.cpp` | Compute shader nodes |
| `test/segment/generate_reference.py` | PyTorch reference |
| `benchmark_pytorch.py` | Vulkan vs PyTorch comparison |

## Weight Key Mapping (PyTorch → Vulkan)

```
# Encoder
feat.stn.conv1.*     → encoder.stn.mlp.mlp0.*
feat.stn.fc1.*       → encoder.stn.fc.block0.*
feat.conv1.*         → encoder.conv1.mlp0.*
feat.fstn.*          → encoder.fstn.*

# Segmentation Head
conv1.*              → segHead.mlp0.*
conv2.*              → segHead.mlp1.*
conv3.*              → segHead.mlp2.*
conv4.*              → conv4.* (PointWiseLinearNode)
```

## Performance

| Backend | Avg Time | Throughput |
|---------|----------|------------|
| Vulkan | ~16 ms | 63k pts/s |
| PyTorch CUDA | ~3.5 ms | 291k pts/s |

*PyTorch is ~4.6x faster due to cuDNN optimization*

## Context Map

- **[library/](./library/)** - Vulkan compute nodes
- **[networks/](./networks/)** - PointNet architecture
- **[test/](./test/)** - Unit tests with PyTorch references

---
Last Updated: 2025-12-29

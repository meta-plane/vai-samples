# AGENTS.md - PointNet Vulkan Implementation

## Project Context

PointNet Semantic Segmentation의 Vulkan Compute Shader 구현 프로젝트.
PyTorch 기반 yanx27/Pointnet_Pointnet2_pytorch 모델을 GPU 컴퓨트 파이프라인으로 포팅.

### Tech Stack

- Language: C++23
- Build: CMake 3.16+, GCC 13+
- GPU: Vulkan 1.3 SDK (LunarG 1.3.290.0)
- Dependencies: GLFW, glslang, SPIRV-Tools, SPIRV-Reflect, nlohmann/json
- Weights Format: SafeTensors (권장), JSON (레거시)
- Reference Framework: PyTorch (테스트 레퍼런스 생성용)

### Architecture

```
PointNetSegment
├── PointNetEncoder
│   ├── TNetBlock (STN: Spatial Transformer)
│   │   ├── MLPSequence<3>
│   │   ├── MaxPooling1D
│   │   ├── FCBNSequence<3>
│   │   └── AddIdentity
│   ├── MatMulNode
│   ├── MLPSequence (conv1-3)
│   └── TNetBlock (FSTN: Feature Transformer)
├── MaxPooling1D (global)
├── BroadcastNode
├── ConcatNode
└── MLPSequence<4> (segHead)
```

## Operational Commands

```bash
# Full project build (from vai-samples root)
cd .. && ./build.sh

# Incremental build (main only)
./build.sh

# Build with tests
./build.sh --test

# Release build
./build.sh --release

# Run main executable
../bin/debug/104-PointNet-Jeonghan

# Run all tests
cd ../build/104-PointNet-Jeonghan && ctest --output-on-failure

# Run specific test
../bin/debug/test_mlp
../bin/debug/test_encoder
../bin/debug/test_segment

# Generate PyTorch reference (from test subdirectory)
cd test/mlp && python generate_reference.py
```

## Golden Rules

### Immutable

- Vulkan SDK 1.3 이상 필수 (VK_API_VERSION_1_3 API 사용)
- 모든 Shader는 GLSL로 작성하고 런타임 컴파일 (glslang)
- 가중치 파일은 SafeTensors 형식 사용 (JSON은 레거시용)

### Do's

- Node 클래스 추가 시 `neuralNodes.h`에 선언, `neuralNodes.cpp`에 구현
- 새 연산 추가 전 반드시 해당 테스트 폴더 생성 및 PyTorch 레퍼런스 작성
- Tensor shape은 [N, C] 형식 (N: points, C: channels)
- 텐서 데이터는 row-major 순서로 저장
- 테스트 허용 오차: epsilon = 1e-4 (float32 precision)
- NodeGroup 상속 시 `defineSlot()`으로 in/out 슬롯 명시

### Don'ts

- `neuralNet.h`의 Node/NeuralNet 기본 구조 변경 금지
- 테스트 없이 커밋 금지 (최소한 관련 test_* 실행)
- Vulkan 리소스를 직접 해제하지 말 것 (Device::destroy() 사용)
- GPU 버퍼를 CPU에서 직접 접근하지 말 것 (staging buffer 사용)

## Standards

### Naming Conventions

- Files: `snake_case.cpp`, `snake_case.h` (library), `camelCase.hpp` (networks)
- Classes: `PascalCaseNode`, `PascalCaseSequence`
- Test files: `test_<component>.cpp`
- Reference files: `reference.safetensors`, `generate_reference.py`

### Weight Parameter Naming (yanx27 format)

```
feat.stn.*         -> encoder.stn.*
feat.fstn.*        -> encoder.fstn.*
feat.conv.mlp0-2.* -> encoder.conv1-3.*
conv1-4.*          -> segHead.mlp0-3.*
```

### Git Commit Format

```
[component] Brief description

- Detailed change 1
- Detailed change 2

Example:
[encoder] Fix FSTN matmul dimension mismatch
[test] Add TNetBlock unit test
[library] Implement AddIdentityNode for TNet
```

### Maintenance Policy

코드와 규칙 간 괴리 발생 시 AGENTS.md 업데이트를 제안하라.
새로운 Node 타입 추가 시 Context Map에 해당 테스트 경로 추가.

## Context Map

- **[Vulkan Core / Neural Nodes](./library/AGENTS.md)** - Compute shader 노드 및 Vulkan 파이프라인 수정 시.
- **[PointNet Networks](./networks/AGENTS.md)** - PointNet 아키텍처 및 네트워크 구현 수정 시.
- **[Component Tests](./test/AGENTS.md)** - 단위 테스트 작성 및 PyTorch 레퍼런스 생성 시.


# AGENTS.md - Component Tests

## Module Context

PointNet 구성 요소의 단위 테스트 모듈.
PyTorch 레퍼런스와 Vulkan 구현 결과를 비교하여 정확성 검증.

### Test Components

```
test/
├── add_identity/   # AddIdentityNode (TNet identity matrix)
├── batchnorm/      # BatchNorm1DNode
├── encoder/        # PointNetEncoder (full encoder)
├── fc/             # FullyConnectedNode
├── fcbn/           # FCBNNode (FC + BN + ReLU)
├── fcbn_seq/       # FCBNSequence<N>
├── fcseq/          # FCSequence<N>
├── matmul/         # MatMulNode
├── maxpool/        # MaxPooling1DNode
├── mlp/            # PointWiseMLPNode
├── mlp_maxpool_fc/ # MLP → MaxPool → FC 체인
├── mlpseq/         # MLPSequence<N>
├── segment/        # PointNetSegment (full network)
├── tnet/           # TNetBlock
└── validation/     # End-to-end accuracy test
```

## Implementation Patterns

### Test File Structure

```
test/<component>/
├── generate_reference.py   # PyTorch 레퍼런스 생성 스크립트
├── reference.safetensors   # 생성된 레퍼런스 데이터
└── test_<component>.cpp    # C++ 테스트 코드
```

### Python Reference Generator (Template)

```python
#!/usr/bin/env python3
"""Generate reference for <ComponentName> test."""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from safetensors.torch import save_file

# Fixed seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
N, C_in, C_out = 8, 3, 64

# Create PyTorch model
class MyComponent(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # ... layers ...
    
    def forward(self, x):
        # ... forward pass ...
        return x

model = MyComponent(C_in, C_out)
model.eval()

# Generate input
input_data = torch.randn(1, C_in, N)

# Forward pass
with torch.no_grad():
    output_data = model(input_data)

# Transpose to [N, C] for C++ (row-major)
input_nc = input_data.squeeze(0).transpose(0, 1).numpy()
output_nc = output_data.squeeze(0).transpose(0, 1).numpy()

# Extract weights (transpose for GEMM)
weight = model.layer.weight.detach().squeeze(-1).transpose(0, 1).numpy()
bias = model.layer.bias.detach().numpy()

# Save SafeTensors
tensors = {
    "input": torch.from_numpy(input_nc).contiguous(),
    "expected": torch.from_numpy(output_nc).contiguous(),
    "weight": torch.from_numpy(weight).contiguous(),
    "bias": torch.from_numpy(bias).contiguous(),
    "shape": torch.tensor([N, C_in, C_out], dtype=torch.float32)
}

output_dir = Path(__file__).parent
save_file(tensors, str(output_dir / "reference.safetensors"))
print(f"Saved: {output_dir / 'reference.safetensors'}")
```

### C++ Test Code (Template)

```cpp
/**
 * <ComponentName> Vulkan Test
 * Tests <description> against PyTorch reference
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include "neuralNet.h"
#include "neuralNodes.h"
#include "vulkanApp.h"
#include "safeTensorsParser.h"

using namespace vk;

class TestNet : public NeuralNet {
    MyNode node;
    
public:
    TestNet(Device& device, uint32_t in_dim, uint32_t out_dim)
    : NeuralNet(device, 1, 1)
    , node(in_dim, out_dim)
    {
        input(0) - node - output(0);
    }
    
    Tensor& operator[](const std::string& name) {
        return node[name];
    }
};

void test() {
    loadShaders();
    
    SafeTensorsParser data(PROJECT_CURRENT_DIR"/test/<component>/reference.safetensors");
    
    // Extract shape
    auto shape = data["shape"].parseNDArray();
    uint32_t N = static_cast<uint32_t>(shape[0]);
    uint32_t C_in = static_cast<uint32_t>(shape[1]);
    uint32_t C_out = static_cast<uint32_t>(shape[2]);
    
    // Create network
    TestNet net(netGlobalDevice, C_in, C_out);
    net["weight"] = Tensor(data["weight"]);
    net["bias"] = Tensor(data["bias"]);
    net.prepare();
    
    // Run inference
    auto input = data["input"].parseNDArray();
    Tensor inputTensor = Tensor(N, C_in).set(input);
    auto result = net(inputTensor);
    
    // Compare with expected
    auto expected = data["expected"].parseNDArray();
    // ... download result and compare ...
    
    float maxError = 0.0f;
    const float epsilon = 1e-4f;
    
    for (size_t i = 0; i < expected.size(); i++) {
        float err = std::abs(result_cpu[i] - expected[i]);
        maxError = std::max(maxError, err);
    }
    
    if (maxError < epsilon) {
        std::cout << "PASS (max error: " << maxError << ")\n";
    } else {
        std::cout << "FAIL (max error: " << maxError << ")\n";
        exit(1);
    }
}

int main() {
    test();
    return 0;
}
```

## Testing Strategy

### Test Execution

```bash
# Build all tests
./build.sh --test

# Run all tests via CTest
cd ../build && ctest --output-on-failure

# Run specific test with verbose output
../bin/debug/test_mlp
../bin/debug/test_encoder

# Regenerate reference after PyTorch model change
cd test/encoder && python generate_reference.py
```

### Adding New Test

```bash
# 1. Create test directory
mkdir test/mynode

# 2. Create Python reference generator
# (Copy template from above, modify for your component)

# 3. Generate reference
cd test/mynode && python generate_reference.py

# 4. Create C++ test
# (Copy template from above, modify for your component)

# 5. Add to CMakeLists.txt
# add_executable(test_mynode test/mynode/test_mynode.cpp)
# target_link_libraries(test_mynode PRIVATE VAI_LIBRARY)
# add_test(NAME MyNode COMMAND $<TARGET_FILE:test_mynode>)
```

## Local Golden Rules

### Do's

- 모든 PyTorch 레퍼런스는 `torch.manual_seed(42)` 사용
- SafeTensors 저장 전 `.contiguous()` 호출
- C++에서 결과 비교 시 epsilon = 1e-4 사용
- 테스트 실패 시 `exit(1)` 반환

### Don'ts

- reference.safetensors를 수동 편집하지 말 것
- 테스트 코드에서 PROJECT_CURRENT_DIR 외 하드코딩 경로 사용 금지
- PyTorch와 C++ 간 데이터 변환 시 transpose 누락 주의
  - PyTorch: [B, C, N] → C++: [N, C]

### Weight Transpose Rules

```python
# Conv1d weight: [C_out, C_in, 1] → [C_in, C_out]
weight = layer.weight.squeeze(-1).transpose(0, 1).numpy()

# Linear weight: [out, in] → [in, out]
weight = layer.weight.transpose(0, 1).numpy()

# BatchNorm: 그대로 사용
running_mean = layer.running_mean.numpy()
running_var = layer.running_var.numpy()
gamma = layer.weight.numpy()
beta = layer.bias.numpy()
```

## Test Results Summary

`TEST_RESULTS.md` 파일에 테스트 결과 기록:

```markdown
| Component | Status | Max Error | Notes |
|-----------|--------|-----------|-------|
| BatchNorm | PASS   | 1.2e-6    |       |
| MLP       | PASS   | 3.4e-5    |       |
| Encoder   | PASS   | 8.7e-5    |       |
```


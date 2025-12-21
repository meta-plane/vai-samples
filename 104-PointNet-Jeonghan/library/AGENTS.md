# AGENTS.md - Vulkan Core Library

## Module Context

Vulkan Compute Shader 기반 Neural Network 연산 노드 라이브러리.
모든 GPU 연산은 이 모듈의 Node 클래스를 통해 수행됨.

### Dependencies

- Vulkan SDK 1.3+ (compute shader, buffer binding)
- glslang (GLSL 런타임 컴파일)
- SPIRV-Reflect (shader reflection)

### File Structure

```
library/
├── neuralNet.h          # Node, NodeGroup, NeuralNet 기본 클래스
├── neuralNodes.h/cpp    # GPU 연산 노드 구현
├── vulkanApp.h/cpp      # Vulkan 디바이스, 파이프라인 래퍼
├── safeTensorsParser.h/cpp  # SafeTensors 파일 파싱
├── jsonParser.h/cpp     # JSON 파일 파싱 (레거시)
├── tensor.h             # Tensor 클래스 (shape + GPU buffer)
├── spirvHelpers.cpp     # SPIRV 유틸리티
└── templateHelper.h     # GLSL 템플릿 치환 헬퍼
```

## Tech Stack & Constraints

- Shader Language: GLSL 450
- Compute Shader Workgroup: 기본 256 threads
- Buffer Memory: VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT (성능), STAGING (전송)
- 모든 Tensor는 float32 (VK_FORMAT_R32_SFLOAT)

## Implementation Patterns

### Node 클래스 생성 패턴

```cpp
// neuralNodes.h
class MyOperationNode : public Node
{
    uint32_t param1, param2;
    ComputePipeline pipeline;
    DescriptorSet descSet;

public:
    MyOperationNode(uint32_t p1, uint32_t p2);
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};
```

```cpp
// neuralNodes.cpp
MyOperationNode::MyOperationNode(uint32_t p1, uint32_t p2)
: Node()
, param1(p1), param2(p2)
{
    // Define tensor slots
    tensors["in0"] = Tensor();   // Input
    tensors["out0"] = Tensor();  // Output
    tensors["weight"] = Tensor(p1, p2);  // Learnable parameter
    
    // Define slot connections
    defineSlot("in0", tensors["in0"]);
    defineSlot("out0", tensors["out0"]);
}

void MyOperationNode::prepare()
{
    // Infer output shape from input
    auto& in = tensors["in0"];
    tensors["out0"] = Tensor(in.shape[0], param2);
    
    // Create compute pipeline
    std::string shaderCode = R"(
        #version 450
        layout(local_size_x = 256) in;
        layout(set = 0, binding = 0) buffer In { float data[]; } inBuf;
        layout(set = 0, binding = 1) buffer Out { float data[]; } outBuf;
        void main() {
            uint idx = gl_GlobalInvocationID.x;
            // ... operation ...
        }
    )";
    pipeline = netGlobalDevice.createComputePipeline(shaderCode);
    descSet = netGlobalDevice.createDescriptorSet(pipeline);
}

void MyOperationNode::run(CommandBuffer cmdBuff)
{
    descSet.update({
        {0, tensors["in0"].buffer},
        {1, tensors["out0"].buffer}
    });
    cmdBuff.bindPipeline(pipeline);
    cmdBuff.bindDescriptorSet(descSet);
    cmdBuff.dispatch(/* workgroups */);
}
```

### Tensor Slot Naming

```
in0, in1      - 입력 텐서
out0, out1    - 출력 텐서
weight        - 가중치 (학습 파라미터)
bias          - 편향
mean, var     - BatchNorm 통계값
gamma, beta   - BatchNorm 스케일/시프트
bn_*          - BatchNorm 관련 (PointWiseMLP 복합 노드)
```

### GLSL Shader 템플릿

```glsl
#version 450
layout(local_size_x = ${WORKGROUP_SIZE}) in;

layout(push_constant) uniform PushConstants {
    uint N;  // batch/points
    uint C;  // channels
} pc;

layout(set = 0, binding = 0) readonly buffer Input { float data[]; } inBuf;
layout(set = 0, binding = 1) writeonly buffer Output { float data[]; } outBuf;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= pc.N * pc.C) return;
    // ...
}
```

## Testing Strategy

```bash
# 새 노드 테스트 추가 시
mkdir test/myop
cd test/myop

# 1. PyTorch 레퍼런스 생성
cat > generate_reference.py << 'EOF'
#!/usr/bin/env python3
import torch
from safetensors.torch import save_file
# ... generate reference ...
EOF
python generate_reference.py

# 2. C++ 테스트 작성
cat > test_myop.cpp << 'EOF'
#include "neuralNodes.h"
#include "safeTensorsParser.h"
// ... test implementation ...
EOF

# 3. CMakeLists.txt에 테스트 타겟 추가
```

## Local Golden Rules

### Do's

- Shader 작성 시 `gl_GlobalInvocationID` 범위 체크 필수
- 새 Node 추가 시 `operator[]`로 파라미터 접근 구현
- Buffer 크기 계산 시 sizeof(float) 명시

### Don'ts

- Shader에서 동적 배열 크기 사용 금지 (push_constant로 전달)
- 단일 dispatch에서 65535 workgroups 초과 금지
- `tensors[]` 직접 접근 대신 `slot()` 또는 `operator[]` 사용
- Vulkan 동기화 없이 버퍼 읽기 금지 (vkQueueWaitIdle 필요)

## Key APIs

```cpp
// Device (vulkanApp.h)
Buffer createBuffer(BufferCreateInfo);
ComputePipeline createComputePipeline(std::string glslCode);
DescriptorSet createDescriptorSet(ComputePipeline);
CommandBuffer beginCompute();
void submitCompute(CommandBuffer);

// Tensor (tensor.h)
Tensor(uint32_t... dims);           // Create with shape
Tensor& set(std::vector<float>);    // Upload data
std::vector<uint32_t> shape;        // Dimension info
Buffer buffer;                       // GPU buffer handle

// Node (neuralNet.h)
Tensor& slot(std::string name);     // Get tensor by slot name
void defineSlot(name, Tensor&);     // Register slot
Node& operator-(Node& next);        // Connect nodes
```


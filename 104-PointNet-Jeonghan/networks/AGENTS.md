# AGENTS.md - PointNet Networks

## Module Context

PointNet Semantic Segmentation 네트워크 아키텍처 구현.
library/ 모듈의 Node 클래스를 조합하여 고수준 네트워크 블록 정의.

### Dependencies

- `library/neuralNet.h` - Node, NodeGroup, NeuralNet 기본 클래스
- `library/neuralNodes.h` - GPU 연산 노드 (MLP, FC, BatchNorm, MatMul 등)

### File Structure

```
networks/
├── include/
│   ├── pointnet.hpp     # PointNet 아키텍처 정의
│   ├── inference.h      # 추론 인터페이스
│   └── weights.h        # 가중치 로딩 유틸
└── src/
    ├── pointnet.cpp     # 구현 (대부분 헤더에 template)
    ├── inference.cpp    # 추론 로직
    └── weights.cpp      # SafeTensors 가중치 로딩
```

## Architecture Classes

### Sequence Blocks (NodeGroup)

```
FCSequence<N>      : N개의 FullyConnectedNode 체인
FCBNSequence<N>    : (N-1)개 FC+BN+ReLU + 1개 FC (마지막 블록 no activation)
MLPSequence<N>     : N개의 PointWiseMLP (Conv1x1+BN+ReLU) 체인
```

### Network Blocks

```
FCBNNode           : FC → Reshape → BN → Reshape → ReLU
TNetBlock          : MLPSeq(3) → MaxPool → FCBNSeq(3) → Reshape → AddIdentity
PointNetEncoder    : STN → MatMul → Conv1 → FSTN → MatMul → Conv2-3
PointNetSegment    : Encoder → MaxPool → Broadcast → Concat → SegHead
```

## Implementation Patterns

### NodeGroup 패턴 (복합 블록)

```cpp
class MyBlock : public NodeGroup
{
    NodeA nodeA;
    NodeB nodeB;

public:
    MyBlock(uint32_t dim)
    : NodeGroup()
    , nodeA(dim)
    , nodeB(dim)
    {
        // 노드 연결
        nodeA - nodeB;
        
        // 외부 슬롯 정의 (필수)
        defineSlot("in0", nodeA.slot("in0"));
        defineSlot("out0", nodeB.slot("out0"));
    }

    // 파라미터 접근자 (가중치 로딩용)
    Tensor& operator[](const std::string& name) {
        if (name.compare(0, 7, "nodeA.") == 0)
            return nodeA[name.substr(7)];
        if (name.compare(0, 7, "nodeB.") == 0)
            return nodeB[name.substr(7)];
        throw std::runtime_error("Unknown: " + name);
    }
};
```

### 다중 경로 연결 (Split/Dual-path)

```cpp
// IdentityNode로 신호 분기
IdentityNode split;

// 경로 A: split.out0 → nodeA → matmul.in0
split / "out0" - "in0" / matmul;

// 경로 B: split.out1 → tnet → matmul.in1
split / "out1" - tnet - "in1" / matmul;
```

### Weight Key Mapping (yanx27 format)

```cpp
Tensor& operator[](const std::string& name) {
    // yanx27: feat.stn.mlp.mlp0.weight
    // 내부:   stn["mlp.mlp0.weight"]
    if (name.compare(0, 4, "stn.") == 0)
        return stn[name.substr(4)];
    
    // yanx27: feat.conv.mlp0.weight → conv1["mlp0.weight"]
    if (name.compare(0, 10, "conv.mlp0.") == 0)
        return conv1[name.substr(5)];  // keep "mlp0.*"
    
    throw std::runtime_error("Unknown: " + name);
}
```

## Weight Loading

```cpp
// weights.cpp 패턴
void loadWeights(PointNetSegment& net, const std::string& path) {
    SafeTensorsParser parser(path);
    
    // yanx27 key 순회
    for (auto& [key, tensor] : parser) {
        // feat.* → encoder, conv* → segHead
        net[mapKey(key)] = Tensor(tensor);
    }
}
```

## Testing Strategy

```bash
# 블록별 테스트
./build.sh --test
../bin/debug/test_tnet      # TNetBlock
../bin/debug/test_encoder   # PointNetEncoder
../bin/debug/test_segment   # PointNetSegment

# PyTorch 레퍼런스 갱신
cd test/encoder && python generate_reference.py
```

## Local Golden Rules

### Do's

- NodeGroup 생성자에서 반드시 `defineSlot("in0", ...)`, `defineSlot("out0", ...)` 호출
- Template 클래스는 헤더에 구현 (링킹 에러 방지)
- 새 블록 추가 시 `operator[]`로 모든 하위 파라미터 접근 가능하게 구현
- 다중 출력 시 `defineSlot("out0", ...)`, `defineSlot("out1", ...)` 명시

### Don'ts

- NodeGroup 내부 노드를 public으로 노출하지 말 것 (캡슐화)
- 노드 연결 순서 변경 시 슬롯 정의 누락 주의
- NeuralNet 상속 클래스에서 `input()`/`output()` 연결 누락 금지
- Sequence 템플릿 파라미터 N과 배열 크기 불일치 금지

## Key Patterns

### PointNet Encoder Dual-Path

```
Input [N, 3]
    │
Split1 ──┬──→ STN → [3,3] matrix ──→ MatMul1.in1
         │                                │
         └──────────────────────────→ MatMul1.in0
                                          │
                              MatMul1 output [N,3]
                                          │
                                       Conv1 [N,64]
                                          │
Split2 ──┬──→ FSTN → [64,64] matrix ──→ MatMul2.in1
         │                                  │
         └───────────────────────────→ MatMul2.in0
                                            │
                                MatMul2 output [N,64]
                                            │
                                     Conv2-3 [N,1024]
```

### Segmentation Concat

```
Encoder
   ├─ out0: pointfeat [N, 64]  ───→ Concat.in0
   │
   └─ out1: full [N, 1024] ──→ MaxPool → Reshape → Broadcast ──→ Concat.in1
                                                                      │
                                                              [N, 1088]
                                                                      │
                                                              SegHead [N, 13]
```


# FCBNNode Design Document

## 개요
FullyConnected + BatchNorm + ReLU를 조합한 NodeGroup

## 목적
- TNet의 FC layer에 BatchNorm + ReLU 추가
- 원본 PointNet 논문 구현과 일치
- 기존 코드 재사용 (FullyConnectedNode, BatchNorm1DNode, ReluNode)

## 설계

### 1. NodeGroup 구조
```cpp
class FCBNNode : public NodeGroup
{
    FullyConnectedNode fc;
    BatchNorm1DNode bn;
    ReluNode relu;

public:
    FCBNNode(uint32_t inDim, uint32_t outDim);
};
```

### 2. 연결 구조
```
Input [I] 
  ↓
FullyConnectedNode [I] → [O]
  ↓
BatchNorm1DNode [O] → [O]
  ↓
ReluNode [O] → [O]
  ↓
Output [O]
```

### 3. 슬롯 정의
```cpp
// Input slots
defineSlot("in0", fc.slot("in0"));

// Weight slots (expose for weight loading)
defineSlot("weight", fc.slot("weight"));
defineSlot("bias", fc.slot("bias"));
defineSlot("bn_mean", bn.slot("mean"));
defineSlot("bn_var", bn.slot("var"));
defineSlot("bn_gamma", bn.slot("gamma"));
defineSlot("bn_beta", bn.slot("beta"));

// Output slot
defineSlot("out0", relu.slot("out0"));
```

### 4. FCBNSequence
```cpp
template<uint32_t nBlocks>
class FCBNSequence : public NodeGroup
{
    std::unique_ptr<FCBNNode> blocks[nBlocks];
    
public:
    FCBNSequence(const uint32_t(&channels)[nBlocks + 1]);
    
    Tensor& operator[](const std::string& name);
    // "fc0.weight", "fc0.bias", "fc0.bn_mean", etc.
};
```

## 구현 위치

### 파일 구조
```
networks/include/pointnet.hpp
  - class FCBNNode (새로 추가)
  - class FCBNSequence (새로 추가)
  - class TNetBlock (수정: FCSequence → FCBNSequence)

test/fc_bn/
  - generate_reference.py (PyTorch reference)
  - test_fc_bn.cpp (Vulkan test)
  - reference.json
```

## 구현 단계

### Phase 1: FCBNNode 구현
1. `pointnet.hpp`에 FCBNNode 클래스 추가
2. 생성자에서 fc - bn - relu 연결
3. 슬롯 정의

### Phase 2: FCBNSequence 구현
1. FCBNNode 배열 생성
2. operator[] 구현 (weight 접근용)
3. 슬롯 노출

### Phase 3: 테스트 작성
1. `test/fc_bn/generate_reference.py`
   - nn.Linear + nn.BatchNorm1d + nn.ReLU
2. `test/fc_bn/test_fc_bn.cpp`
   - Single FC+BN+ReLU 테스트
3. 검증

### Phase 4: TNet 통합
1. TNetBlock에서 FCSequence → FCBNSequence 교체
2. generate_reference.py 업데이트 (BatchNorm 파라미터 포함)
3. test_tnet 검증

## PyTorch Reference 구조

```python
# Single FC+BN+ReLU
class FCBNBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# Weight extraction
weight = model.fc.weight.t()  # [O, I] → [I, O]
bias = model.fc.bias
bn_mean = model.bn.running_mean
bn_var = model.bn.running_var
bn_gamma = model.bn.weight
bn_beta = model.bn.bias
```

## Weight 구조

### TNet FC Sequence (3 layers)
```
fc.fc0.weight: [1024, 512]
fc.fc0.bias: [512]
fc.fc0.bn_mean: [512]
fc.fc0.bn_var: [512]
fc.fc0.bn_gamma: [512]
fc.fc0.bn_beta: [512]

fc.fc1.weight: [512, 256]
fc.fc1.bias: [256]
fc.fc1.bn_mean: [256]
fc.fc1.bn_var: [256]
fc.fc1.bn_gamma: [256]
fc.fc1.bn_beta: [256]

fc.fc2.weight: [256, 9]
fc.fc2.bias: [9]
# fc2는 마지막이라 BN+ReLU 없음
```

## 주의사항

### 1. 마지막 Layer
- FCBNSequence의 마지막 block은 **ReLU 없음**
- 옵션 A: 마지막만 FullyConnectedNode 사용
- 옵션 B: FCBNNode에 `withRelu` 플래그 추가
- **권장**: 옵션 A (단순함)

### 2. BatchNorm 기본값
- 테스트: Identity (mean=0, var=1, gamma=1, beta=0)
- 실사용: 학습된 값 로드

### 3. 기존 코드 영향
- FCSequence는 유지 (다른 곳에서 사용 중일 수 있음)
- FCBNSequence는 새로 추가
- 호환성 유지

## 예상 결과

### 전
```cpp
FCSequence<3> fc({1024, 512, 256, 9});
// Linear only
```

### 후
```cpp
FCBNSequence<2> fc_bn({1024, 512, 256});
FullyConnectedNode fc_final(256, 9);
// fc_bn: Linear+BN+ReLU × 2
// fc_final: Linear only
```

또는:

```cpp
FCBNSequence<3> fc({1024, 512, 256, 9}, {true, true, false});
// withRelu flags for each layer
```

## 검증 기준

### 테스트 통과 조건
1. test_fc_bn: Single FC+BN+ReLU 정확도 < 1e-4
2. test_tnet: 논문 구조로 정확도 < 1e-4
3. 빌드 경고 없음
4. 기존 테스트 영향 없음

## 타임라인

| 단계 | 예상 시간 | 파일 |
|------|----------|------|
| FCBNNode 구현 | 30분 | pointnet.hpp |
| FCBNSequence 구현 | 30분 | pointnet.hpp |
| test_fc_bn 생성 | 1시간 | test/fc_bn/* |
| TNet 통합 | 30분 | test/tnet/* |
| **총계** | **2.5시간** | |

## 다음 액션

1. ✅ 설계 문서 작성 (현재)
2. ⏳ BatchNorm1DNode 존재 확인
3. ⏳ ReluNode 존재 확인
4. ⏳ FCBNNode 구현
5. ⏳ test_fc_bn 작성
6. ⏳ TNet 통합


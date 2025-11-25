# PointNet Tests - PyTorch Comparison

이 디렉토리는 Vulkan PointNet 구현이 PyTorch 참조 구현과 동일한 결과를 내는지 검증하는 테스트를 포함합니다.

## 🎯 테스트 전략

**작은 것부터 시작해서 점진적으로 확장:**

1. ✅ **Single MLP Layer**: 가장 기본적인 레이어
2. ✅ **TNet Input Transform**: 3x3 공간 변환
3. ✅ **TNet Feature Transform**: 64x64 특성 변환
4. ✅ **PointNet Encoder**: 전체 인코더
5. ✅ **Full Segmentation**: 전체 네트워크

## 📋 사전 준비

### 1. PyTorch 설치 (Python)

```bash
# Conda 환경 생성
conda create -n pointnet python=3.7
conda activate pointnet

# PyTorch 설치
conda install pytorch==1.6.0 cudatoolkit=10.1 -c pytorch
# 또는 CPU 버전
# conda install pytorch==1.6.0 cpuonly -c pytorch
```

### 2. 참조 데이터 생성

```bash
cd 104-PointNet-Jeonghan

# PyTorch로 참조 출력 생성
python test/generate_reference.py
```

이 스크립트는 다음을 생성합니다:
- 고정된 입력 데이터 (재현 가능하도록 seed 고정)
- PyTorch 모델의 출력
- 모델 weights (`.pth` 형식)

생성된 파일:
```
test/references/
├── mlp_layer/
│   └── reference.pth
├── tnet_k3/
│   └── reference.pth
├── tnet_k64/
│   └── reference.pth
├── encoder/
│   └── reference.pth
└── full_network/
    ├── reference.pth
    ├── input.txt
    └── output.txt
```

### 3. Vulkan 구현 빌드

```bash
# 전체 빌드
cd ../build
cmake ..
make -j$(nproc)

# 또는 테스트만
make pointnet-tests
```

### 4. 테스트 실행

```bash
# 실행 파일 위치로 이동
cd /home/jeonghan/workspace/vai-samples/build/bin/debug

# 테스트 실행
./pointnet-tests
```

## 📊 예상 출력

```
╔════════════════════════════════════════════════════════╗
║      PointNet Unit Tests - PyTorch Comparison         ║
╚════════════════════════════════════════════════════════╝

>>> Test 1: Single MLP Layer
============================================================
Test: MLP Layer (3 -> 64)
------------------------------------------------------------
Status: ✓ PASSED
Elements: 1024
Max diff: 1.234e-05
Mean diff: 3.456e-06
Message: All elements within tolerance
============================================================

>>> Test 2: TNet Input Transform (3x3)
...

>>> Test 5: Full PointNet Segmentation
...

============================================================
Test Summary
============================================================
Overall: ✓ ALL TESTS PASSED
============================================================
```

## 🔧 허용 오차 (Tolerance)

- **Relative tolerance (rtol)**: 1e-3 (0.1%)
- **Absolute tolerance (atol)**: 1e-5

이 값들은 다음 차이를 허용합니다:
- 부동소수점 연산 오차
- GPU vs CPU 계산 차이
- Vulkan compute shader 구현 차이

## 🐛 문제 해결

### "Reference file not found"
```bash
# 참조 데이터를 먼저 생성하세요
python test/generate_reference.py
```

### "PyTorch not installed"
```bash
# Conda 환경 활성화
conda activate pointnet

# PyTorch 설치
conda install pytorch==1.6.0 cpuonly -c pytorch
```

### 테스트 실패시
1. **작은 차이 (< 1e-3)**: GPU 부동소수점 오차 - 정상
2. **큰 차이 (> 1e-2)**: 구현 오류 가능성
   - Weights 로딩 확인
   - 레이어 순서 확인
   - 활성화 함수 확인
3. **Shape mismatch**: 텐서 차원 오류
   - 입력/출력 shape 확인
   - Transpose/Reshape 연산 확인

### 디버깅 팁

```cpp
// test_pointnet.cpp에서 중간 결과 출력
std::cout << "Intermediate output: " << output[0] << "\n";

// 특정 레이어만 테스트
// main()에서 다른 테스트 주석 처리
```

## 📚 추가 자료

- **PyTorch 참조**: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
- **원 논문**: https://arxiv.org/abs/1612.00593
- **Vulkan Compute 가이드**: https://www.khronos.org/blog/vulkan-compute-shaders

## 🎓 테스트 작성 가이드

새로운 테스트를 추가하려면:

1. **PyTorch 참조 구현** (`generate_reference.py`):
```python
def test_new_component():
    model = NewComponent()
    model.eval()
    
    x = generate_test_inputs()
    output = model(x)
    
    torch.save({'input': x, 'output': output}, 
               'test/references/new_component/reference.pth')
```

2. **C++ 테스트** (`test_pointnet.cpp`):
```cpp
bool test_new_component() {
    // Load reference
    auto ref = test_utils::loadReferenceOutput("test/references/new_component/output.txt");
    
    // Run Vulkan implementation
    // ... your code ...
    
    // Compare
    auto result = test_utils::compareTensors(output, ref);
    test_utils::printTestResult("New Component", result);
    
    return result.passed;
}
```

3. **main()에 추가**:
```cpp
all_passed &= test_new_component();
```

---

**Last Updated**: 2025-01-23  
**목표**: Vulkan 구현이 PyTorch와 수치적으로 동일함을 검증


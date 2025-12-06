# 테스트 데이터 생성기

GPT-2 레이어 테스트를 위한 데이터 생성 스크립트 모음

## 📁 폴더 구조

```
test_data_generators/
├── json_exporter.py              # JSON 내보내기 유틸리티 (공통)
├── numpy/                        # NumPy 기반 생성기 (레거시)
│   └── generate_*_test.py
└── torch/                        # PyTorch 기반 생성기 (현재 사용) ⭐
    ├── torch_layers.py           # LLM-from-Scratch PyTorch 구현
    ├── generate_*_test_cpu.py    # CPU 버전 생성기
    ├── generate_*_test_gpu.py    # GPU 버전 생성기
    ├── generate_all_tests_cpu.py # 전체 생성 (CPU)
    └── generate_all_tests_gpu.py # 전체 생성 (GPU)
```

## 🚀 사용법

### 1. PyTorch GPU로 테스트 데이터 생성 (권장)

**모든 레이어 테스트 데이터 생성:**

```bash
cd test/test_data_generators/torch
C:\Users\USER\.conda\envs\torch\python.exe generate_all_tests_gpu.py
```

**개별 레이어 생성:**

```bash
python generate_gelu_test_gpu.py
python generate_linear_test_gpu.py
python generate_layernorm_test_gpu.py
python generate_add_test_gpu.py
python generate_attention_test_gpu.py
python generate_feedforward_test_gpu.py
python generate_transformer_test_gpu.py
```

### 2. PyTorch CPU로 테스트 데이터 생성

```bash
cd test/test_data_generators/torch
C:\Users\USER\.conda\envs\torch\python.exe generate_all_tests_cpu.py
```

## 📊 생성되는 테스트 파일

모든 생성기는 `../../assets/test_data/` 위치에 JSON 파일을 생성합니다:

| 레이어 | 파일명 | 설명 |
|--------|--------|------|
| GELU | `gelu_test.json` | GELU 활성화 함수 |
| Linear | `linear_test.json` | 선형 변환 레이어 |
| LayerNorm | `layernorm_test.json` | 레이어 정규화 |
| Add | `add_test.json` | 잔차 연결 (Residual) |
| Attention | `attention_test.json` | 멀티헤드 셀프 어텐션 |
| FeedForward | `feedforward_test.json` | MLP (Linear → GELU → Linear) |
| Transformer | `transformer_test.json` | 전체 트랜스포머 블록 |

## 🔧 json_exporter.py

모든 생성기에서 사용하는 공통 유틸리티

### 주요 함수

```python
# 테스트 데이터 내보내기
export_test_data(
    output_path="../../assets/test_data/gelu_test.json",
    input_data=input_tensor,
    output_data=output_tensor,
    parameters={"weight": weight, "bias": bias}  # 선택사항
)

# 데이터 변환
to_list(data)              # NumPy/PyTorch → Python list
to_nested_list(tensor)     # 별칭

# 재현성을 위한 시드 설정
set_seed(42)
```

**지원 형식:**
- NumPy array
- PyTorch tensor (CPU/GPU 모두)
- 자동으로 Python list로 변환하여 JSON 저장

## 📝 네이밍 규칙

- **CPU 버전**: `generate_<layer>_test_cpu.py`
- **GPU 버전**: `generate_<layer>_test_gpu.py`

## 🎯 PyTorch GPU vs CPU 차이점

### GPU 버전 (권장)
- PyTorch GPU에서 실행 후 결과 저장
- Vulkan 구현의 기준(reference)으로 사용
- 더 정확한 비교 가능 (같은 GPU 환경)

### CPU 버전
- PyTorch CPU에서 실행
- GPU 없는 환경에서 사용
- 개발/디버깅용

## 🧪 검증 방법

테스트 데이터 생성 후 C++ 테스트 실행:

```bash
cd ../../
../bin/debug/gpt2-unit-tests.exe
```

Vulkan vs PyTorch GPU 비교:

```bash
cd ../../utils
python final_comparison.py
```

## 📚 레이어별 상세 정보

### GELU (Gaussian Error Linear Unit)
- 입력 shape: `[2, 3, 8]`
- 활성화 함수
- 파라미터 없음

### Linear (선형 변환)
- 입력 shape: `[2, 4, 768]`
- 출력 shape: `[2, 4, 768]`
- 파라미터: `weight`, `bias`

### LayerNorm (레이어 정규화)
- 입력 shape: `[2, 4, 768]`
- 파라미터: `scale`, `shift`

### Add (잔차 연결)
- 입력 shape: `[2, 4, 768]`
- 두 텐서의 element-wise 덧셈
- 파라미터: `in1` (두 번째 입력)

### MultiHeadAttention (멀티헤드 셀프 어텐션)
- 입력 shape: `[1, 4, 768]`
- 12개 헤드
- 파라미터: `W_query`, `B_query`, `W_key`, `B_key`, `W_value`, `B_value`, `W_out`, `B_out`

### FeedForward (MLP)
- 입력 shape: `[2, 4, 768]`
- 구조: Linear(768→3072) → GELU → Linear(3072→768)
- 파라미터: `weight1`, `bias1`, `weight2`, `bias2`

### TransformerBlock (트랜스포머 블록)
- 입력 shape: `[1, 4, 768]`
- Pre-LayerNorm 구조
- 총 16개 파라미터 (norm1 + attention + norm2 + feedforward)

## ✅ 현재 상태

- ✅ PyTorch GPU 기반 생성기 완성
- ✅ json_exporter 통합 완료
- ✅ CPU/GPU 파일 구분 완료
- ✅ 모든 레이어 테스트 데이터 생성 가능
- ✅ Vulkan vs PyTorch GPU 비교 검증 완료
- ✅ LLM-from-Scratch 구현 기반

## 🔗 관련 링크

- 비교 스크립트: `../../utils/final_comparison.py`
- C++ 테스트: `../runTests.cpp`
- LLM-from-Scratch: https://github.com/rickiepark/llm-from-scratch

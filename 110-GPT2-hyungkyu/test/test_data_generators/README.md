# 테스트 데이터 생성기

딥러닝 레이어 테스트를 위한 공통 데이터 생성 프레임워크

## 📁 폴더 구조

```
test_data_generators/
├── json_exporter.py              # JSON 내보내기 유틸리티 (공통)
├── <model_name>/                 # 모델별 생성기 폴더
│   ├── layers.py                 # 레이어 구현
│   ├── generate_*_test_cpu.py    # CPU 버전 생성기
│   ├── generate_*_test_gpu.py    # GPU 버전 생성기
│   └── generate_all_tests_*.py   # 전체 생성 스크립트
└── ...
```

## 🎯 설계 철학

이 프레임워크는 **모델 독립적**으로 설계되었습니다:
- 새로운 모델의 테스트 데이터 생성기를 쉽게 추가 가능
- 공통 유틸리티(`json_exporter.py`) 재사용
- 일관된 JSON 형식으로 C++ 테스트와 연동

## 🚀 새로운 테스트 생성기 추가 방법

### 1단계: 모델 폴더 생성

```bash
mkdir test_data_generators/<model_name>
```

### 2단계: 레이어 구현 작성

`<model_name>/layers.py` 파일을 작성합니다:

```python
"""
<Model Name> 레이어 구현
참조 구현 또는 공식 라이브러리 기반
"""
import torch
import torch.nn as nn

class YourLayer(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # 레이어 초기화

    def forward(self, x):
        # Forward pass 구현
        return output
```

### 3단계: 생성기 스크립트 작성

`generate_<layer>_test_gpu.py` 템플릿:

```python
"""
<Layer> 테스트 데이터 생성 (GPU 버전)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from layers import YourLayer  # 레이어 import
from json_exporter import export_test_data, set_seed

# CUDA 확인
if not torch.cuda.is_available():
    print("ERROR: CUDA not available")
    sys.exit(1)

# 재현성을 위한 시드 설정
set_seed(42)

# 테스트 입력 생성
input_data = torch.randn(batch, seq_len, d_model, dtype=torch.float32)

# 레이어 생성 및 초기화
layer = YourLayer(...)

# GPU로 이동
layer = layer.cuda()
input_gpu = input_data.cuda()

# Forward pass
layer.eval()
with torch.no_grad():
    output_gpu = layer(input_gpu)

# CPU로 이동하여 export
output_data = output_gpu.cpu()

# JSON으로 저장
export_test_data(
    output_path="../../assets/test_data/<layer>_test.json",
    input_data=input_data,
    output_data=output_data,
    parameters={
        "weight": layer.weight.cpu(),
        "bias": layer.bias.cpu()
    }  # 파라미터가 있는 경우
)

print(f"\n<Layer> test data generated with PyTorch GPU!")
```

### 4단계: CPU 버전 작성

GPU 버전과 동일하되, `.cuda()` 호출만 제거:

```python
# GPU 대신 CPU에서 실행
output_cpu = layer(input_data)

export_test_data(
    output_path="../../assets/test_data/<layer>_test.json",
    input_data=input_data,
    output_data=output_cpu,
    parameters={...}
)
```

### 5단계: 전체 생성 스크립트

`generate_all_tests_gpu.py`:

```python
"""
모든 테스트 데이터 일괄 생성 (GPU)
"""
import subprocess
import sys
import os

generators = [
    "generate_layer1_test_gpu.py",
    "generate_layer2_test_gpu.py",
    # ... 추가
]

for i, generator in enumerate(generators, 1):
    print(f"[{i}/{len(generators)}] Running {generator}...")
    result = subprocess.run([sys.executable, generator], ...)
    # 에러 처리
```

## 🔧 json_exporter.py 사용법

### 기본 사용

```python
from json_exporter import export_test_data, set_seed

# 1. 시드 설정 (재현성)
set_seed(42)

# 2. 테스트 데이터 생성
input_data = torch.randn(2, 4, 768)
output_data = layer(input_data)

# 3. JSON으로 내보내기
export_test_data(
    output_path="../../assets/test_data/my_layer_test.json",
    input_data=input_data,
    output_data=output_data,
    parameters=None  # 파라미터가 없는 레이어
)
```

### 파라미터가 있는 레이어

```python
export_test_data(
    output_path="../../assets/test_data/linear_test.json",
    input_data=input_data,
    output_data=output_data,
    parameters={
        "weight": layer.weight.cpu(),
        "bias": layer.bias.cpu()
    }
)
```

### 지원 형식

- ✅ PyTorch tensor (CPU/GPU)
- ✅ NumPy array
- ✅ 자동 Python list 변환

## 📝 파일 네이밍 규칙

| 용도 | 파일명 | 설명 |
|------|--------|------|
| 레이어 구현 | `layers.py` | 모델의 레이어 구현 모음 |
| GPU 생성기 | `generate_<layer>_test_gpu.py` | GPU 기반 데이터 생성 |
| CPU 생성기 | `generate_<layer>_test_cpu.py` | CPU 기반 데이터 생성 |
| 전체 생성 (GPU) | `generate_all_tests_gpu.py` | 모든 GPU 테스트 생성 |
| 전체 생성 (CPU) | `generate_all_tests_cpu.py` | 모든 CPU 테스트 생성 |

## 📊 생성되는 JSON 형식

```json
{
  "input": [[[1.0, 2.0, ...]]],
  "output": [[[3.0, 4.0, ...]]],
  "parameters": {
    "weight": [[...]],
    "bias": [...]
  }
}
```

**필수 필드:**
- `input`: 입력 텐서 (중첩 리스트)
- `output`: 기대 출력 텐서 (중첩 리스트)

**선택 필드:**
- `parameters`: 레이어 파라미터 딕셔너리

## 🎯 GPU vs CPU 버전

### GPU 버전 (권장)
- **용도**: 프로덕션 테스트, Vulkan 비교
- **장점**: GPU 구현과 직접 비교 가능
- **요구사항**: CUDA 지원 GPU

### CPU 버전
- **용도**: 개발, 디버깅, CI/CD
- **장점**: GPU 없이도 실행 가능
- **단점**: GPU 결과와 약간의 수치 차이

---

## 📚 예제: GPT-2 모델

현재 구현된 GPT-2 예제를 참조하세요:

### 폴더 구조

```
test_data_generators/
├── json_exporter.py
└── torch/                        # GPT-2 예제
    ├── torch_layers.py           # GPT-2 레이어 (LLM-from-Scratch 기반)
    ├── generate_gelu_test_gpu.py
    ├── generate_linear_test_gpu.py
    ├── generate_layernorm_test_gpu.py
    ├── generate_add_test_gpu.py
    ├── generate_attention_test_gpu.py
    ├── generate_feedforward_test_gpu.py
    ├── generate_transformer_test_gpu.py
    ├── generate_all_tests_gpu.py
    └── ... (CPU 버전들)
```

### 사용 예시

```bash
# GPT-2 테스트 데이터 생성
cd test_data_generators/torch
C:\Users\USER\.conda\envs\torch\python.exe generate_all_tests_gpu.py
```

### 구현된 레이어

| 레이어 | 파일명 | 설명 |
|--------|--------|------|
| GELU | `gelu_test.json` | GELU 활성화 함수 |
| Linear | `linear_test.json` | 선형 변환 |
| LayerNorm | `layernorm_test.json` | 레이어 정규화 |
| Add | `add_test.json` | 잔차 연결 |
| MultiHeadAttention | `attention_test.json` | 멀티헤드 셀프 어텐션 |
| FeedForward | `feedforward_test.json` | MLP |
| TransformerBlock | `transformer_test.json` | 전체 트랜스포머 블록 |

### 레이어 구현 예시

`torch/torch_layers.py`에서 발췌:

```python
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
```

### 생성기 예시

`torch/generate_gelu_test_gpu.py`에서 발췌:

```python
import torch
from torch_layers import GELU
from json_exporter import export_test_data, set_seed

set_seed(42)

input_data = torch.randn(2, 3, 8, dtype=torch.float32)
gelu = GELU().cuda()
output_gpu = gelu(input_data.cuda())

export_test_data(
    output_path="../../../assets/test_data/gelu_test.json",
    input_data=input_data,
    output_data=output_gpu.cpu()
)
```

---

## 🔗 관련 문서

- C++ 테스트 프레임워크: `../README.md`
- 비교 스크립트: `../../utils/final_comparison.py`
- GPT-2 레이어 구현: `torch/torch_layers.py`

## ✅ 체크리스트

새로운 모델 추가 시:

- [ ] 모델 폴더 생성
- [ ] `layers.py` 작성
- [ ] 각 레이어별 GPU 생성기 작성
- [ ] 각 레이어별 CPU 생성기 작성
- [ ] `generate_all_tests_gpu.py` 작성
- [ ] `generate_all_tests_cpu.py` 작성
- [ ] JSON 파일 생성 확인
- [ ] C++ 테스트 케이스 작성
- [ ] 검증 및 비교 스크립트 작성

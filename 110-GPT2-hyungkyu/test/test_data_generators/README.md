# Test Data Generators

테스트 데이터 생성 스크립트 모음

## 📁 폴더 구조

```
test_data_generators/
├── json_exporter.py          # JSON 내보내기 유틸리티 (공통)
├── numpy/                    # NumPy 기반 생성기 (레거시)
│   ├── generate_*_test.py
│   └── ...
└── torch/                    # PyTorch 기반 생성기 (현재 사용)
    ├── torch_layers.py       # LLM-from-Scratch 구현
    ├── generate_*_test_cpu.py    # CPU 버전 생성기
    ├── generate_*_test_gpu.py    # GPU 버전 생성기
    ├── generate_all_tests_cpu.py # 모든 CPU 테스트 생성
    ├── generate_all_tests_gpu.py # 모든 GPU 테스트 생성
    └── *_comparison.py       # 비교 스크립트
```

## 🚀 사용법

### PyTorch GPU로 테스트 데이터 생성 (권장)

```bash
cd test/test_data_generators/torch
C:\Users\USER\.conda\envs\torch\python.exe generate_all_tests_gpu.py
```

### PyTorch CPU로 테스트 데이터 생성

```bash
cd test/test_data_generators/torch
C:\Users\USER\.conda\envs\torch\python.exe generate_all_tests_cpu.py
```

### 개별 레이어 테스트 데이터 생성

```bash
# GPU 버전
python generate_gelu_test_gpu.py
python generate_linear_test_gpu.py
python generate_layernorm_test_gpu.py
python generate_add_test_gpu.py
python generate_attention_test_gpu.py
python generate_feedforward_test_gpu.py
python generate_transformer_test_gpu.py

# CPU 버전
python generate_gelu_test_cpu.py
python generate_linear_test_cpu.py
...
```

## 📊 비교 스크립트

### PyTorch GPU vs Vulkan 최종 비교

```bash
python final_comparison.py
```

출력: Vulkan 오차 (PyTorch GPU 기준)

### 상세 비교 (CPU/GPU/Vulkan)

```bash
python detailed_comparison.py
```

출력: 모든 플랫폼 간 오차 비교

## 🔧 json_exporter.py

모든 생성기에서 사용하는 공통 유틸리티

### 주요 함수

- `export_test_data(output_path, input_data, output_data, parameters=None)`
  - 테스트 데이터를 JSON 형식으로 내보내기
  - NumPy array와 PyTorch tensor 모두 지원

- `to_list(data)` / `to_nested_list(tensor)`
  - NumPy/PyTorch를 Python list로 변환

- `set_seed(seed=42)`
  - 재현 가능성을 위한 랜덤 시드 설정

## 📝 네이밍 규칙

- **CPU 버전**: `generate_<layer>_test_cpu.py`
- **GPU 버전**: `generate_<layer>_test_gpu.py`

## ⚙️ 생성되는 파일

모든 생성기는 다음 위치에 JSON 파일을 생성합니다:

```
../../assets/test_data/<layer>_test.json
```

예:
- `gelu_test.json`
- `linear_test.json`
- `layernorm_test.json`
- `add_test.json`
- `attention_test.json`
- `feedforward_test.json`
- `transformer_test.json`

## 🎯 현재 상태

- ✅ PyTorch GPU 기반 생성기 완성
- ✅ json_exporter 통합 완료
- ✅ CPU/GPU 파일 구분 완료
- ✅ 모든 레이어 테스트 데이터 생성 가능
- ✅ Vulkan vs PyTorch 비교 검증 완료

# Utils - 유틸리티 스크립트

GPT-2 프로젝트 관련 유틸리티 스크립트 모음

## 📁 파일 구조

```
utils/
├── final_comparison.py          # Vulkan vs PyTorch GPU 최종 비교 ⭐
├── download_gpt2_weights.py     # GPT-2 가중치 다운로드
├── convert_openai_weights.py    # OpenAI 가중치 변환
├── setup_weights.py             # 가중치 설정
└── archive/                     # 레거시 스크립트 보관
    ├── compare_*.py             # 과거 비교 스크립트들
    └── README.md
```

## 🎯 주요 스크립트

### 1. final_comparison.py ⭐

**Vulkan vs PyTorch GPU 최종 비교 결과**

```bash
cd utils
python final_comparison.py
```

**출력:**
- Layer별 Mean Error / Max Error
- PyTorch GPU를 기준(reference)으로 Vulkan 오차 측정
- Float32 정밀도 검증

**결과 예시:**
```
Layer                     Mean Error             Max Error
======================================================================
GELU                      1.55e-09               5.96e-08
Linear                    4.17e-09               3.73e-08
LayerNorm                 1.49e-07               1.43e-06
AddNode                   0.00e+00               0.00e+00
MultiHeadAttention        3.17e-09               2.24e-08
FeedForward               6.67e-09               6.15e-08
TransformerBlock          4.23e-07               2.86e-06
```

### 2. download_gpt2_weights.py

GPT-2 가중치를 다운로드합니다.

```bash
python download_gpt2_weights.py
```

### 3. convert_openai_weights.py

OpenAI 형식의 가중치를 프로젝트 형식으로 변환합니다.

```bash
python convert_openai_weights.py
```

### 4. setup_weights.py

가중치 설정을 자동화합니다.

```bash
python setup_weights.py
```

## 📦 archive/

레거시 비교 스크립트들이 보관되어 있습니다.
- 개발 과정에서 사용된 다양한 비교 방법론
- 필요시 참고용
- 자세한 내용은 `archive/README.md` 참조

## 🔗 관련 폴더

테스트 데이터 생성은 `test/test_data_generators/` 참조:
- PyTorch GPU/CPU 기반 테스트 데이터 생성
- JSON 내보내기 유틸리티
- LLM-from-Scratch 구현

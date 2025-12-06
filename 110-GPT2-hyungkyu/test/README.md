# 테스트 프레임워크

딥러닝 레이어를 위한 JSON 기반 공통 테스트 프레임워크

## 🎯 설계 철학

이 테스트 프레임워크는 **모델 독립적**으로 설계되었습니다:
- 다양한 딥러닝 모델의 레이어를 쉽게 테스트
- JSON 기반 데이터로 유연한 확장
- Variadic Template을 활용한 일반화된 구조
- 한 줄의 코드로 테스트 추가 가능

## 📁 폴더 구조

```
test/
├── README.md                      # 이 문서
├── graphTest.h                    # 테스트 프레임워크 선언
├── graphTest.inl                  # 템플릿 구현 (header-only)
├── runTests.cpp                   # 테스트 실행기
├── jsonParser.h/cpp               # JSON 파싱 유틸리티
└── test_data_generators/          # 테스트 데이터 생성기
    ├── json_exporter.py           # 공통 유틸리티
    └── <model_name>/              # 모델별 생성기
```

**파일 설명:**
- `graphTest.h`: GraphTest 템플릿 클래스 선언, 마지막에 `graphTest.inl` include
- `graphTest.inl`: 모든 템플릿 메서드 구현 (inline implementation)
- Header-only 방식이므로 별도의 `.cpp` 파일 불필요
- 새로운 노드 타입 추가 시 템플릿 인스턴스화 코드 작성 불필요

## 🚀 새로운 테스트 추가 방법

### 1단계: 테스트 데이터 생성 (Python)

```python
from json_exporter import export_test_data
import torch

# 테스트 입력/출력 생성
input_data = torch.randn(2, 4, 768)
output_data = your_layer(input_data)

# JSON으로 저장 (데이터만, 설정은 제외)
export_test_data(
    output_path="../../assets/test_data/your_test.json",
    input_data=input_data,
    output_data=output_data,
    parameters={"weight": weight, "bias": bias}  # 선택사항
)
```

**중요**: 노드 생성자 인자는 JSON이 아닌 C++에서 전달합니다.

### 2단계: 테스트 등록 (C++)

`runTests.cpp`에 한 줄만 추가:

```cpp
void registerTests() {
    // 생성자 인자가 없는 노드
    addTest<GELUNode>(
        "GELU - Standard (2x3x8)",
        PROJECT_CURRENT_DIR "/assets/test_data/gelu_test.json");

    // 생성자 인자가 있는 노드
    addTest<LinearNode>(
        "Linear - Forward Pass",
        PROJECT_CURRENT_DIR "/assets/test_data/linear_test.json",
        768, 3072);  // in_features, out_features
}
```

**중요**: Header-only 템플릿 방식이므로 `graphTest.cpp`에 템플릿 인스턴스화를 추가할 필요가 없습니다!

### 3단계: 빌드 및 실행

```bash
cmake --build ../build --target gpt2-unit-tests
../bin/debug/gpt2-unit-tests.exe
```

## 📊 JSON 형식

```json
{
  "input": [[[...]]],        // 필수: 입력 텐서
  "output": [[[...]]],       // 필수: 기대 출력
  "parameters": {             // 선택: 레이어 파라미터
    "weight": [[...]],
    "bias": [...]
  }
}
```

**필수 필드:**
- `input`: 입력 텐서 (중첩 리스트)
- `output`: 기대 출력 텐서 (중첩 리스트)

**선택 필드:**
- `parameters`: 파라미터 딕셔너리

**중요**: 노드 생성자 인자(예: `d_model`, `num_heads`)는 JSON이 아닌 `addTest()` 호출에서 전달합니다.

## 🔧 GraphTest 템플릿

### 핵심 기능

```cpp
// graphTest.h - 템플릿 선언
template<typename NodeType>
class GraphTest : public ITest {
public:
    // Variadic 템플릿: 어떤 생성자 시그니처도 지원
    template<typename... Args>
    GraphTest(const std::string& name,
              const std::string& jsonPath,
              Args&&... args);

    bool execute() override;
};

// graphTest.inl - 템플릿 구현 (header-only)
#include "graphTest.inl"
```

**Header-only 템플릿 방식:**
- 템플릿 선언은 `graphTest.h`에 위치
- 템플릿 구현은 `graphTest.inl`에 위치
- `graphTest.h` 끝에서 `graphTest.inl`을 include
- 컴파일러가 사용 시점에 자동으로 템플릿 인스턴스화
- **별도의 `.cpp` 파일 불필요, 명시적 템플릿 인스턴스화 불필요**

**자동으로 처리:**
- ✅ JSON에서 입력/출력/파라미터 로딩
- ✅ 노드 인스턴스 생성 (가변 인자 전달)
- ✅ CPU 데이터 → GPU 텐서 변환
- ✅ 파라미터 슬롯 이름 매핑
- ✅ Forward pass 실행 및 검증
- ✅ 오차 계산 및 리포트

### 동적 파라미터 로딩

프레임워크는 JSON의 모든 파라미터를 자동으로 순회:

```cpp
void GraphTest<T>::loadParametersFromJSON(JsonParser& json) {
    auto paramKeys = json["parameters"].getKeys();  // 모든 키 가져오기
    for (const auto& key : paramKeys) {
        // 각 파라미터를 노드의 operator[]로 매핑
        node[key] = loadTensorFromJSON(json["parameters"][key]);
    }
}
```

### 파라미터 슬롯 매핑

레이어는 `operator[]`로 파라미터 접근을 제공해야 합니다:

```cpp
Tensor& YourLayer::operator[](const std::string& name) {
    if (name == "weight") return this->weight;
    if (name == "bias") return this->bias;
    throw std::runtime_error("Unknown parameter: " + name);
}
```

## 📝 테스트 실행 흐름

```
1. JSON 파싱
   ↓
2. 입력/출력/파라미터 로딩
   ↓
3. 노드 생성 (Variadic 템플릿)
   ↓
4. GPU 메모리 할당 및 전송
   ↓
5. Forward Pass 실행
   ↓
6. 결과 비교 및 오차 계산
   ↓
7. PASS/FAIL 판정
```

## 🎓 고급 기능

### NodeGroup 테스트

여러 노드를 포함하는 복합 레이어도 테스트 가능:

```cpp
class TransformerBlock : public NodeGroup {
    LayerNormNode norm1;
    MultiHeadAttentionNode attention;
    AddNode add1;
    // ...
};

// 테스트 등록
addTest<TransformerBlock>(
    "TransformerBlock - Full Block",
    PROJECT_CURRENT_DIR "/assets/test_data/transformer_test.json",
    768, 12  // d_model, num_heads
);
```

### 중첩 파라미터 매핑

NodeGroup의 내부 노드 파라미터도 접근 가능:

```cpp
Tensor& TransformerBlock::operator[](const std::string& name) {
    // 중첩된 노드의 파라미터 접근
    if (name == "norm1_scale") return norm1["scale"];
    if (name == "attn_wq") return attention["W_query"];
    if (name == "ff_w1") return feedforward["weight1"];
    // ...
}
```

### 허용 오차 조정

```cpp
void registerTests() {
    auto test = std::make_unique<GraphTest<YourNode>>(
        "High Precision Test",
        PROJECT_CURRENT_DIR "/assets/test_data/test.json",
        args...);
    test->setTolerance(1e-6f);  // 기본값: 1e-4
    tests.push_back(std::move(test));
}
```

---

## 📚 예제: GPT-2 모델

현재 구현된 GPT-2 테스트 케이스들을 참조하세요.

### 구현된 레이어

| 레이어 | 생성자 인자 | JSON 파일 |
|--------|-------------|-----------|
| GELUNode | 없음 | `gelu_test.json` |
| LinearNode | `(in_features, out_features)` | `linear_test.json` |
| LayerNormNode | `(normalized_shape)` | `layernorm_test.json` |
| AddNode | 없음 | `add_test.json` |
| MultiHeadAttentionNode | `(d_in, d_out, num_heads)` | `attention_test.json` |
| FeedForwardNode | `(d_model)` | `feedforward_test.json` |
| TransformerBlock | `(d_model, num_heads)` | `transformer_test.json` |

### 테스트 등록 예시

`runTests.cpp`:

```cpp
void registerTests() {
    // 파라미터 없는 노드
    addTest<GELUNode>(
        "GELU - Standard (2x3x8)",
        PROJECT_CURRENT_DIR "/assets/test_data/gelu_test.json");

    // 단일 파라미터
    addTest<LayerNormNode>(
        "LayerNorm - Standard (2x4x768)",
        PROJECT_CURRENT_DIR "/assets/test_data/layernorm_test.json",
        768);

    // 다중 파라미터
    addTest<LinearNode>(
        "Linear - Forward Pass (2x4x768 -> 2x4x768)",
        PROJECT_CURRENT_DIR "/assets/test_data/linear_test.json",
        768, 768);

    // NodeGroup (복합 레이어)
    addTest<TransformerBlock>(
        "TransformerBlock - Full Block (1x4x768, 12 heads)",
        PROJECT_CURRENT_DIR "/assets/test_data/transformer_test.json",
        768, 12);
}
```

### 실행 결과

```
╔════════════════════════════════════════════════════════╗
║               Unit Tests - Layer Testing               ║
╚════════════════════════════════════════════════════════╝

GELU - Standard (2x3x8)
  Input:  [2, 3, 8]
  Output: [2, 3, 8]
  Tolerance: 0.0001
  Max Error:  5.96e-08
  Mean Error: 1.55e-09
  Time: 72.124 ms
  Result: PASS

TransformerBlock - Full Block (1x4x768, 12 heads)
  Input:  [1, 4, 768]
  Output: [1, 4, 768]
  Max Error:  2.86e-06
  Mean Error: 4.23e-07
  Time: 89.892 ms
  Result: PASS

============================================================
Total tests run: 7
Tests passed: 7
Tests failed: 0

✓ ALL TESTS PASSED!
```

---

## 🛠️ 문제 해결

### "Data size mismatch" 오류

JSON의 입력/출력 shape이 C++ 레이어가 기대하는 것과 일치하는지 확인

### "invalid map<K, T> key" 오류

파라미터 슬롯 이름 불일치. `operator[]`에서 올바른 슬롯 이름을 반환하는지 확인

### 파라미터가 로딩되지 않음

JSON의 파라미터 이름이 노드의 슬롯 이름과 일치하는지 확인

### 높은 오차 값

1. Python 참조 구현이 GPU 셰이더 로직과 일치하는지 확인
2. 텐서 shape 확인 (특히 Linear의 weight transpose)
3. float32 정밀도 차이 고려

## ✨ 프레임워크의 장점

1. **보일러플레이트 없음**: 파생 테스트 클래스 불필요
2. **Variadic 템플릿**: 생성자 인자를 직접 전달
3. **Python 유연성**: NumPy/PyTorch로 참조 구현
4. **깔끔한 분리**:
   - 테스트 데이터 (JSON) → `assets/`
   - 생성 스크립트 (Python) → `test_data_generators/`
   - 테스트 로직 (C++) → `test/`
5. **타입 안전성**: 컴파일 타임 타입 체크
6. **쉬운 유지보수**: 한 줄로 테스트 추가
7. **자동 매핑**: 파라미터 슬롯 이름 자동 처리

## 🎯 설계 원칙

1. **JSON은 데이터만 포함** (설정은 C++ 코드에)
2. **한 줄로 테스트 등록** (`addTest<NodeType>(...)`)
3. **Factory 패턴 불필요** (Variadic 템플릿이 모든 것을 처리)
4. **모델 독립적** (어떤 딥러닝 모델도 지원)
5. **자동 파라미터 로딩** (JSON 키 기반)

## 🔗 관련 문서

- 테스트 데이터 생성: [`test_data_generators/README.md`](test_data_generators/README.md)
- 비교 스크립트: `../utils/final_comparison.py`

## ✅ 체크리스트

새로운 테스트 추가 시:

- [ ] Python으로 테스트 데이터 생성
- [ ] C++ 레이어 `operator[]` 구현 (파라미터가 있는 경우)
- [ ] `runTests.cpp`에 `addTest` 추가
- [ ] 빌드 및 실행
- [ ] PASS 확인

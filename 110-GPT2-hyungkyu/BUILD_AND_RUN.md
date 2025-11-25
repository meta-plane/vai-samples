# Build and Run Guide

## Quick Start

### 1. CMakeLists.txt 교체
```bash
# 백업 (선택사항)
cp CMakeLists.txt CMakeLists_old.txt

# 새 버전으로 교체
cp CMakeLists_new.txt CMakeLists.txt
```

### 2. 빌드
```bash
# Windows (현재 디렉토리에서)
cmake --build . --config debug

# 또는 전체 재빌드
cmake --build . --config debug --clean-first
```

### 3. 실행

이제 **3개의 실행 파일**이 생성됩니다:

---

## 실행 파일 설명

### 📦 1. `110-GPT2-hyungkyu.exe` (기존 main - 하위 호환성)
**목적**: 기존 코드와의 호환성 유지

**실행:**
```bash
./bin/debug/110-GPT2-hyungkyu.exe
./bin/debug/110-GPT2-hyungkyu.exe "Hello, world" 30
./bin/debug/110-GPT2-hyungkyu.exe --test-basic
```

**특징:**
- 기존 main.cpp 사용
- 기존 테스트 함수들 포함
- 추후 deprecated 예정

---

### 🧪 2. `runAllTests.exe` (유닛 테스트 러너) ⭐ 새로 추가

**목적**: 빠른 유닛 테스트 실행 (개발 중 자주 실행)

**실행:**
```bash
./bin/debug/runAllTests.exe
```

**출력 예시:**
```
╔════════════════════════════════════════╗
║  GPT-2 Unit Test Suite                ║
╚════════════════════════════════════════╝

================================================================================
Running Layer Tests...
================================================================================

╔════════════════════════════════════════╗
║  Test Suite: Basic Transformer Layers  ║
╚════════════════════════════════════════╝

========================================
Test: LayerNorm - Basic Functionality
========================================
  ✓ Output shape verified: [2, 4, 768]
  ✓ Normalization verified: mean ~0, std ~1
✓ Test completed successfully

[... 더 많은 테스트 ...]

========================================
Test Suite Summary: Basic Transformer Layers
========================================
✓ PASS: LayerNorm - Basic Functionality (12.45 ms)
✓ PASS: LayerNorm - PyTorch Reference (23.67 ms)
✓ PASS: GELU - Basic Functionality (8.32 ms)
[...]

Total: 13 tests
Passed: 13
Failed: 0
Total time: 156.42 ms

✓ All tests passed!
```

**테스트 목록:**
- LayerNorm (기본 + PyTorch 검증)
- GELU (기본 + PyTorch 검증)
- AddNode, IdentityNode
- FeedForward (기본 + PyTorch 검증)
- LinearNode, SoftmaxNode
- MultiHeadAttention (기본 + PyTorch 검증)
- KV Cache 통합

**장점:**
- 🚀 빠름 (~200ms, 가중치 로드 불필요)
- ✅ 자동화된 검증
- 📊 자동 타이밍 측정
- 🐛 개발 중 버그 조기 발견

**언제 사용:**
- 코드 변경 후
- 커밋 전
- 리팩토링 중
- 새 기능 추가 후

---

### 🚀 3. `gpt2-inference.exe` (새 추론 CLI) ⭐ 새로 추가

**목적**: 사전학습된 가중치로 텍스트 생성

**기본 사용:**
```bash
# 기본 생성 (KV cache 활성화)
./bin/debug/gpt2-inference.exe

# 커스텀 프롬프트
./bin/debug/gpt2-inference.exe generate "Once upon a time"

# 토큰 수 지정
./bin/debug/gpt2-inference.exe generate "Hello" 50
```

**고급 옵션:**
```bash
# KV cache 비활성화 (느림)
./bin/debug/gpt2-inference.exe --no-cache generate "Hello" 30

# Temperature 조정 (창의성)
./bin/debug/gpt2-inference.exe --temperature 1.0 generate "Hello" 30

# Top-k 샘플링
./bin/debug/gpt2-inference.exe --top-k 50 generate "Hello" 30

# 재현 가능한 결과 (시드 고정)
./bin/debug/gpt2-inference.exe --seed 42 generate "Hello" 30
```

**3가지 모드:**

#### Mode 1: Generate (기본 생성)
```bash
./bin/debug/gpt2-inference.exe generate "The future of AI is" 50
```

**출력:**
```
========================================
Text Generation
========================================
Prompt: "The future of AI is"
Max tokens: 50
Mode: KV Cache Enabled
Temperature: 0.8
Top-k: 40
========================================

Prompt encoded to 6 tokens

Generating...

--- Generated Text ---
The future of AI is bright and full of possibilities. We are on the cusp of
a new era where machines will assist humans in ways we never imagined...
--- End ---

Statistics:
  Generated tokens: 44
  Total tokens: 50
  Generation time: 2134 ms (2.13 sec)
  Generation speed: 20.66 tokens/sec
```

#### Mode 2: Compare (성능 비교)
```bash
./bin/debug/gpt2-inference.exe compare "Hello, I'm a language model," 50
```

**출력:**
```
╔════════════════════════════════════════╗
║  KV Cache Performance Comparison       ║
╚════════════════════════════════════════╝

[Standard 생성 결과...]
[Cached 생성 결과...]

Performance Comparison Summary
================================================================================
Standard Generation:
  Time: 6.82 sec
  Speed: 7.33 tokens/sec

Cached Generation:
  Time: 2.18 sec
  Speed: 22.94 tokens/sec

Speedup: 3.13x faster with cache
Output verification: ✓ MATCH
```

#### Mode 3: Interactive (대화형)
```bash
./bin/debug/gpt2-inference.exe interactive
```

**사용:**
```
╔════════════════════════════════════════╗
║  GPT-2 Interactive Text Generation     ║
╚════════════════════════════════════════╝

Loading...
✓ Model loaded
✓ Tokenizer loaded

Ready for text generation!
Type your prompt (or 'quit' to exit)

> Once upon a time
[생성 결과 출력...]

> In a galaxy far, far away
[생성 결과 출력...]

> quit
Goodbye!
```

**도움말:**
```bash
./bin/debug/gpt2-inference.exe --help
```

---

## 개발 워크플로우

### 일반적인 작업 순서:

1. **코드 수정**
   ```bash
   # 예: model/transformerBlock/transformer.cpp 수정
   ```

2. **빌드**
   ```bash
   cmake --build . --config debug
   ```

3. **유닛 테스트 실행** (빠름 ~200ms)
   ```bash
   ./bin/debug/runAllTests.exe
   ```

4. **수정 사항이 테스트 통과하면, 추론 테스트** (느림 ~10초)
   ```bash
   ./bin/debug/gpt2-inference.exe generate "Test prompt" 30
   ```

5. **커밋**
   ```bash
   git add .
   git commit -m "Fix: ..."
   ```

---

## 문제 해결

### 빌드 에러

**에러: "Cannot open source file"**
```
Solution: CMakeLists.txt가 올바르게 업데이트되었는지 확인
cp CMakeLists_new.txt CMakeLists.txt
```

**에러: "Unresolved external symbol"**
```
Solution: 전체 재빌드
cmake --build . --config debug --clean-first
```

### 실행 에러

**에러: "Pretrained weights not found"**
```
Solution: 가중치 다운로드
cd utils
python download_gpt2_weights.py
```

**에러: "Vulkan initialization failed"**
```
Solution: Vulkan 드라이버 확인
- GPU 드라이버 업데이트
- Vulkan SDK 설치 확인
```

---

## 성능 비교

| 작업 | 실행 파일 | 시간 | 용도 |
|------|----------|------|------|
| 유닛 테스트 | runAllTests.exe | ~200ms | 개발 중 자주 실행 |
| 기본 생성 (30 tokens) | gpt2-inference.exe | ~1-2초 | 빠른 검증 |
| 생성 (100 tokens, cache) | gpt2-inference.exe | ~4-5초 | 일반 사용 |
| 생성 (100 tokens, no cache) | gpt2-inference.exe | ~12-15초 | 캐시 미사용 |
| 성능 비교 | gpt2-inference.exe compare | ~15-20초 | 벤치마크 |

---

## 추천 설정

### 개발 중:
```bash
# 자주 실행 (빠름)
./bin/debug/runAllTests.exe
```

### 커밋 전:
```bash
# 1. 유닛 테스트
./bin/debug/runAllTests.exe

# 2. 짧은 추론 테스트
./bin/debug/gpt2-inference.exe generate "Test" 20
```

### 데모/테스트:
```bash
# 대화형 모드로 여러 프롬프트 테스트
./bin/debug/gpt2-inference.exe interactive
```

### 성능 검증:
```bash
# 캐시 효과 측정
./bin/debug/gpt2-inference.exe compare "Hello" 50
```

---

## 다음 단계

1. ✅ CMakeLists.txt 교체
2. ✅ 빌드
3. ✅ runAllTests.exe 실행 (유닛 테스트)
4. ✅ gpt2-inference.exe 실행 (추론)
5. 🔜 기존 테스트 코드 마이그레이션
6. 🔜 CMakeLists.txt 최종 확정

**참고:**
- `main_new.cpp` → 추후 `main.cpp`로 교체 예정
- `CMakeLists_new.txt` → 추후 `CMakeLists.txt`로 교체 예정
- 기존 `110-GPT2-hyungkyu.exe`는 하위 호환성 유지 후 제거 예정

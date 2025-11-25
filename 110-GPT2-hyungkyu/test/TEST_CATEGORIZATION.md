# Test Categorization and Migration Plan

## Test Categories

### 1. Unit Tests (→ test/ directory)
Tests that verify individual component functionality without pretrained weights.
- Fast execution
- No external dependencies (weights files)
- Verify correctness of implementation

### 2. Inference Tests (→ model/ directory or separate inference/)
Tests that load pretrained weights and run generation.
- Require weight files
- Longer execution time
- Verify end-to-end generation quality

## Current Test Files Analysis

### model/gpt2Test.cpp
**Unit Tests (migrate to test/):**
- `testGPT2()` - Basic network creation test
- `testGPT2Generation()` - Generation with random weights (integration test)
- Helper functions for weight initialization

**Inference Tests (keep in model/ or move to inference/):**
- `testGPT2Pretrained()` - Text generation with pretrained weights
- `testGPT2WithCache()` - KV cache performance comparison
- Requires: pretrained weights, tokenizer

### model/embedding/test.cpp
**Unit Tests (migrate to test/):**
- `testTokenEmbedding()` - Token embedding functionality
- `testPositionalEmbedding()` - Position embedding functionality
- `testGPTEmbedding()` - Combined embedding test
All are pure unit tests, no weight loading needed.

### tokenizer/test.cpp
**Unit Tests (migrate to test/):**
- Tokenizer encoding/decoding tests
- No external dependencies

### dataloader/test.cpp
**Unit Tests (migrate to test/):**
- Data loading and batching tests
- No external dependencies

## Migration Plan

### Phase 1: Create Test Structure
✅ Create test/ directory
✅ Move testFramework files to test/
✅ Fix include paths

### Phase 2: Migrate Layer Unit Tests
- [ ] Create test/layerTests.cpp
  - LayerNorm tests
  - GELU tests
  - FeedForward tests
  - Add tests
  - Identity tests
- [ ] Create test/attentionTests.cpp
  - MultiHeadAttention basic tests
  - KV cache unit tests (without full model)
- [ ] Create test/transformerTests.cpp
  - TransformerBlock tests
- [ ] Create test/embeddingTests.cpp
  - Migrate from model/embedding/test.cpp
  - TokenEmbedding, PositionalEmbedding, GPTEmbedding
- [ ] Create test/tokenizerTests.cpp
  - Migrate from tokenizer/test.cpp
- [ ] Create test/dataloaderTests.cpp
  - Migrate from dataloader/test.cpp

### Phase 3: Create Inference Module
- [ ] Create model/inference.h
  - Functions for loading pretrained weights
  - Text generation interface
- [ ] Create model/inference.cpp
  - `GPT2Inference` class
  - `generate()` method
  - `generate_with_cache()` method
- [ ] Update main.cpp to use inference module

### Phase 4: Test Runner
- [ ] Create test/runAllTests.cpp
  - Main test runner that executes all test suites
  - Reports summary
- [ ] Update CMakeLists.txt
  - Add test target
  - Link test files

## File Structure (Target)

```
110-GPT2-hyungkyu/
├── test/
│   ├── README.md                    (framework documentation)
│   ├── testFramework.h              (framework implementation)
│   ├── testHelpers.h                (JSON helpers)
│   ├── testFramework_example.cpp    (usage examples)
│   ├── runAllTests.cpp              (test runner)
│   ├── layerTests.cpp               (LayerNorm, GELU, FeedForward, etc.)
│   ├── attentionTests.cpp           (Attention unit tests)
│   ├── transformerTests.cpp         (TransformerBlock tests)
│   ├── embeddingTests.cpp           (Embedding tests)
│   ├── tokenizerTests.cpp           (Tokenizer tests)
│   └── dataloaderTests.cpp          (Dataloader tests)
├── model/
│   ├── inference.h                  (inference interface)
│   ├── inference.cpp                (inference implementation)
│   ├── gpt2Net.h
│   ├── gpt2Net.cpp
│   └── ... (other model files)
└── main.cpp                         (uses inference module)
```

## Benefits

### Separation of Concerns
- **test/**: Fast, focused unit tests for development
- **model/inference**: End-to-end inference for validation

### Improved Development Workflow
1. Run unit tests frequently during development (fast)
2. Run inference tests before commits (slower, comprehensive)

### Better Organization
- All test infrastructure in one place
- Clear distinction between testing and inference
- Easier to maintain and extend

# Test Framework Documentation

## Overview

This test framework provides a structured, maintainable approach to testing GPT-2 model components. It standardizes the common pattern of:
1. Setup (create layer/model)
2. Run (execute forward pass)
3. Verify (check results)

## Core Components

### 1. BaseTest Class
Base class for all tests providing:
- **Lifecycle methods**: `setup()`, `run()`, `teardown()`
- **Automatic timing**: Measures execution time
- **Error handling**: Catches and reports exceptions
- **Command buffer management**: Simplified Vulkan command submission

```cpp
class MyTest : public BaseTest {
public:
    MyTest() : BaseTest("Test Name") {}

    void setup() override {
        // Create layers, allocate resources
    }

    void run() override {
        // Execute test logic
    }

    void teardown() override {
        // Cleanup (optional)
    }
};
```

### 2. TestAssert Class
Common assertion utilities:
- `assertEqual(actual, expected, tolerance)` - Compare tensors
- `assertTrue(condition, msg)` - Assert boolean condition
- `assertClose(actual, expected, tolerance)` - Compare floats
- `assertShape(tensor, expected_shape)` - Verify tensor shape

```cpp
void run() override {
    TestAssert::assertShape(output, {2, 4, 768}, "Output shape");
    TestAssert::assertEqual(output, expected, 1e-4f, "Output values");
}
```

### 3. LayerTest<T> Template
Template for testing individual layers:
- Provides structured layer testing interface
- Automatically calls lifecycle methods

```cpp
class MyLayerTest : public LayerTest<LayerNormNode> {
    void createLayer() override { /* ... */ }
    void prepareInputs() override { /* ... */ }
    void runLayer() override { /* ... */ }
    void verifyOutputs() override { /* ... */ }
};
```

### 4. TestSuite Class
Groups multiple tests together:
- Executes tests in sequence
- Collects and reports results
- Provides summary statistics

```cpp
TestSuite suite("Component Tests");
suite.addTest([]() { MyTest test; return test.execute(); });
suite.addTest([]() { AnotherTest test; return test.execute(); });
suite.runAll();
```

### 5. TestResult Struct
Stores test execution results:
- Test name
- Pass/fail status
- Error message (if failed)
- Execution time

## Usage Patterns

### Pattern 1: Simple Test

```cpp
class LayerNormBasicTest : public BaseTest {
    std::unique_ptr<LayerNormNode> layer;
    Tensor input, output;

public:
    LayerNormBasicTest() : BaseTest("LayerNorm Basic") {}

    void setup() override {
        layer = std::make_unique<LayerNormNode>(768);
        input = Tensor(2, 4, 768);
        // ... initialize input data
    }

    void run() override {
        (*layer)["in0"] = input;
        layer->prepare();

        cmd_buffer.begin();
        layer->run(cmd_buffer);
        submitAndWait();

        output = (*layer)["out0"];
        TestAssert::assertShape(output, {2, 4, 768});
    }
};
```

### Pattern 2: JSON Reference Test

```cpp
class AttentionJSONTest : public BaseTest {
    json test_data;
    std::unique_ptr<MultiHeadAttentionNode> layer;

public:
    AttentionJSONTest() : BaseTest("Attention vs PyTorch") {}

    void setup() override {
        test_data = loadTestData("../assets/test_data/mha_test_data.json");
        uint32_t d_in = test_data["config"]["d_in"];
        uint32_t d_out = test_data["config"]["d_out"];
        uint32_t num_heads = test_data["config"]["num_heads"];

        layer = std::make_unique<MultiHeadAttentionNode>(d_in, d_out, num_heads);
    }

    void run() override {
        // Load input and weights from JSON
        Tensor input = createTensorFromJSON(test_data["input"]);
        // ... load weights, run layer

        // Verify against reference
        Tensor expected = createTensorFromJSON(test_data["output"]);
        TestAssert::assertEqual(output, expected, 1e-3f);
    }
};
```

### Pattern 3: Parameterized Tests

```cpp
class ParameterizedTest : public BaseTest {
    uint32_t B, S, D;

public:
    ParameterizedTest(uint32_t batch, uint32_t seq, uint32_t dim)
        : BaseTest("Test [" + std::to_string(batch) + "," +
                   std::to_string(seq) + "," + std::to_string(dim) + "]"),
          B(batch), S(seq), D(dim) {}

    void run() override {
        // Test with parameters B, S, D
    }
};

// Usage:
TestSuite suite("Parameterized Tests");
for (auto [B, S, D] : configs) {
    suite.addTest([=]() {
        ParameterizedTest test(B, S, D);
        return test.execute();
    });
}
suite.runAll();
```

### Pattern 4: Multi-Step Tests (e.g., Cache Tests)

```cpp
class CacheTest : public BaseTest {
public:
    void run() override {
        // Step 1: Initial forward pass
        std::cout << "  Step 1: Initial pass" << std::endl;
        // ... test logic

        // Step 2: Enable cache
        std::cout << "  Step 2: Cached pass" << std::endl;
        // ... test logic

        // Step 3: Verify cache state
        std::cout << "  Step 3: Verification" << std::endl;
        TestAssert::assertTrue(cache.current_len == expected_len);
    }
};
```

## Benefits

### Before (Ad-hoc Tests)
```cpp
void testLayerNorm() {
    std::cout << "=== Test LayerNorm ===" << std::endl;
    LayerNormNode layer(768);
    // ... lots of boilerplate code
    std::cout << "✓ Test passed" << std::endl;
}
```

**Problems:**
- Duplicated boilerplate
- No timing information
- Manual error handling
- Inconsistent output format
- Hard to maintain

### After (Framework-based Tests)
```cpp
class LayerNormTest : public BaseTest {
public:
    LayerNormTest() : BaseTest("LayerNorm") {}
    void run() override {
        // Just the test logic, no boilerplate
    }
};
```

**Benefits:**
- **Standardized**: Consistent structure across all tests
- **Less boilerplate**: Framework handles setup/teardown/timing
- **Better error reporting**: Automatic exception handling and reporting
- **Timing**: Automatic execution time measurement
- **Grouping**: Easy to organize tests into suites
- **Maintainable**: Changes to test infrastructure in one place
- **Reusable**: Common assertions and utilities

## Migration Guide

### Migrating Existing Tests

**Old style:**
```cpp
void testGPT2() {
    std::cout << "========== GPT-2 Test ==========" << std::endl;

    GPT2Config config = GPT2TinyConfig();
    GPT2Net gpt2Net(netGlobalDevice, config);

    std::cout << "✓ Network created" << std::endl;
    std::cout << "✓ Test passed" << std::endl;
}
```

**New style:**
```cpp
class GPT2BasicTest : public BaseTest {
    GPT2Config config;
    std::unique_ptr<GPT2Net> gpt2Net;

public:
    GPT2BasicTest() : BaseTest("GPT-2 Network Creation") {}

    void setup() override {
        config = GPT2TinyConfig();
    }

    void run() override {
        gpt2Net = std::make_unique<GPT2Net>(netGlobalDevice, config);
        TestAssert::assertTrue(gpt2Net != nullptr, "Network created");
        std::cout << "  ✓ Network created successfully" << std::endl;
    }
};

// Usage:
GPT2BasicTest test;
auto result = test.execute();
```

## Example Output

```
╔════════════════════════════════════════╗
║  Test Suite: Transformer Components   ║
╚════════════════════════════════════════╝

========================================
Test: LayerNorm Basic Functionality
========================================
  ✓ Output shape verified: [2, 4, 768]
  ✓ Normalization verified (mean ~0)
✓ Test completed successfully

========================================
Test: FeedForward vs PyTorch Reference
========================================
  ✓ Output matches PyTorch reference (within 1e-3 tolerance)
✓ Test completed successfully

========================================
Test Suite Summary: Transformer Components
========================================
✓ PASS: LayerNorm Basic Functionality (12.45 ms)
✓ PASS: FeedForward vs PyTorch Reference (23.67 ms)

Total: 2 tests
Passed: 2
Failed: 0
Total time: 36.12 ms

✓ All tests passed!
========================================
```

## Best Practices

1. **One test per class**: Each test class should test one specific functionality
2. **Descriptive names**: Use clear, descriptive test names
3. **Small, focused tests**: Keep tests small and focused on one aspect
4. **Use assertions**: Always use TestAssert methods instead of manual checks
5. **Clean separation**: Use setup/run/teardown for clear separation of concerns
6. **Group related tests**: Use TestSuite to group related tests
7. **Document test intent**: Add comments explaining what the test verifies

## Future Enhancements

Potential improvements:
- Benchmark mode (multiple iterations, statistical analysis)
- Test filtering (run specific tests by name/tag)
- Parallel test execution
- Test fixtures (shared setup/teardown across multiple tests)
- Mock objects for dependency injection
- Test coverage reporting

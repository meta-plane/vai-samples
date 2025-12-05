# GPT-2 Graph Unit Test Framework

A JSON-driven test framework for validating computational graph nodes and node groups on Vulkan.

## Overview

This test framework uses a template-based architecture where:
- **Python scripts** generate test data (inputs, expected outputs, parameters)
- **JSON files** store the test data in a standardized format
- **C++ GraphTest** loads and executes tests using variadic templates

**Key principle**: Test writers only need to add one line in `runTests.cpp` and generate JSON data. Node constructor arguments are passed directly through `addTest()`.

## Quick Start

### 1. Generate Test Data (Python)

```python
from json_exporter import export_test_data
import numpy as np

# Create your test data
input_data = np.random.randn(2, 3, 8).astype(np.float32)
output_data = your_layer_function(input_data)

# Export to JSON (only data, no config)
export_test_data(
    output_path="../../assets/test_data/your_test.json",
    input_data=input_data,
    output_data=output_data,
    parameters={"weight": weight, "bias": bias}  # Optional
)
```

**Important**: Python scripts are located in `utils/test_data_generators/`, not in `assets/`.

### 2. Register Test (C++)

```cpp
void registerTests() {
    // Node with no constructor arguments
    addTest<GELUNode>(
        "GELU - Standard (2x3x8)",
        PROJECT_CURRENT_DIR "/assets/test_data/gelu_test.json");

    // Node with constructor arguments
    addTest<LinearNode>(
        "Linear - Forward Pass",
        PROJECT_CURRENT_DIR "/assets/test_data/linear_test.json",
        768, 3072);  // in_features, out_features
}
```

### 3. Build and Run

```bash
cmake --build ../build --config Debug --target gpt2-unit-tests
../bin/debug/gpt2-unit-tests.exe
```

## Directory Structure

```
110-GPT2-hyungkyu/
├── test/
│   ├── graphTest.h          # GraphTest template class
│   ├── graphTest.cpp        # Implementation
│   ├── runTests.cpp         # Test registration and runner
│   └── README.md            # This file
├── utils/test_data_generators/
│   ├── json_exporter.py           # Standard JSON export utility
│   ├── generate_gelu_test.py      # GELU test data generator
│   ├── generate_linear_test.py    # Linear test data generator
│   └── generate_layernorm_test.py # LayerNorm test data generator
└── assets/test_data/
    ├── gelu_test.json        # Generated test data
    ├── linear_test.json      # Generated test data
    └── layernorm_test.json   # Generated test data
```

**Separation principle**:
- `utils/` = Python scripts (generation code)
- `assets/` = JSON data (generated results)

## JSON Format

JSON files contain **only data** (no node configuration):

```json
{
  "input": [[[...]]],
  "output": [[[...]]],
  "parameters": {
    "weight": [[[...]]],
    "bias": [...]
  }
}
```

### Fields

- **input** (required): N-dimensional array representing input tensor
- **output** (required): Expected output tensor for validation
- **parameters** (optional): Dictionary of parameter tensors (weights, biases, etc.)

**Note**: Node constructor arguments are specified in C++ `addTest()` call, not in JSON.

## Python Test Data Generation

### Using json_exporter.py

The `json_exporter.py` utility handles conversion from NumPy/PyTorch to JSON:

```python
from json_exporter import export_test_data
import numpy as np

# Works with NumPy arrays
input_np = np.random.randn(2, 3, 768).astype(np.float32)
output_np = layer_function(input_np)

export_test_data(
    output_path="../../assets/test_data/test.json",
    input_data=input_np,
    output_data=output_np,
    parameters={"weight": weight, "bias": bias}
)
```

### Example: Linear Layer Test

```python
import numpy as np
from json_exporter import export_test_data

def main():
    np.random.seed(42)

    # Configuration
    batch_size = 1
    seq_len = 4
    in_features = 768
    out_features = 3072

    # Generate data
    input_data = np.random.randn(batch_size, seq_len, in_features).astype(np.float32)

    # Weight shape: [out_features, in_features] for GPU
    weight = np.random.randn(out_features, in_features).astype(np.float32) * 0.02
    bias = np.zeros(out_features, dtype=np.float32)

    # Compute output: Y = X @ W^T + b
    output_data = input_data @ weight.T + bias

    # Export
    export_test_data(
        output_path="../../assets/test_data/linear_test.json",
        input_data=input_data,
        output_data=output_data,
        parameters={"weight": weight, "bias": bias}
    )

if __name__ == "__main__":
    main()
```

## C++ Test Implementation

### GraphTest Template

The `GraphTest<T>` template handles all test execution:

```cpp
template<typename T>
class GraphTest : public ITest {
public:
    // Variadic template constructor
    template<typename... Args>
    GraphTest(const std::string& name, const std::string& jsonPath, Args&&... args);

    bool execute() override;
    void setTolerance(float tol);
    // ... getters
};
```

**Automatic behavior:**
- Loads input/output/parameters from JSON
- Creates node instance with forwarded constructor arguments
- Converts CPU data to GPU tensors
- Maps parameter slot names (e.g., "weight"/"bias" → "scale"/"shift" for LayerNorm)
- Runs inference and validates results
- Reports timing and error metrics

### Test Registration

Tests are registered in `runTests.cpp` using the `addTest<NodeType>()` helper:

```cpp
void registerTests() {
    // GELU (no constructor args)
    addTest<GELUNode>(
        "GELU - Standard (2x3x8)",
        PROJECT_CURRENT_DIR "/assets/test_data/gelu_test.json");

    // Linear (constructor args: in_features, out_features)
    addTest<LinearNode>(
        "Linear - Forward Pass (1x4x768 -> 1x4x3072)",
        PROJECT_CURRENT_DIR "/assets/test_data/linear_test.json",
        768, 3072);

    // LayerNorm (constructor arg: normalized_shape)
    addTest<LayerNormNode>(
        "LayerNorm - Standard (2x4x768)",
        PROJECT_CURRENT_DIR "/assets/test_data/layernorm_test.json",
        768);
}
```

### Supported Node Types

Currently instantiated template types (see `graphTest.cpp` bottom):

- `GraphTest<LinearNode>`
- `GraphTest<LayerNormNode>`
- `GraphTest<GELUNode>`

To add support for new node types, add template instantiation:

```cpp
template class GraphTest<YourNewNode>;
```

### Parameter Slot Name Mapping

The framework automatically maps JSON parameter names to node slot names:

- **LinearNode**: `"weight"` → `"weight"`, `"bias"` → `"bias"`
- **LayerNormNode**: `"weight"` → `"scale"`, `"bias"` → `"shift"`
- **GELUNode**: No parameters

This mapping is handled in `loadTestDataFromJSON()` using `if constexpr`.

## Test Output

```
╔════════════════════════════════════════════════════════╗
║  GPT-2 Unit Tests - Layer Testing                    ║
╚════════════════════════════════════════════════════════╝

GELU - Standard (2x3x8)
  Input:  [2, 3, 8]
  Output: [2, 3, 8]
  Tolerance: 0.0001
  Expected: [ 0.83408, -0.108144, 1.16872, 3.04292, -0.14977 ... ]
  Actual:   [ 0.83408, -0.108144, 1.16872, 3.04292, -0.14977 ... ]
  Max Error:  5.96046e-08
  Mean Error: 5.27749e-09
  Time: 17.472 ms
  Result: PASS

Linear - Forward Pass (1x4x768 -> 1x4x3072)
  Input:  [1, 4, 768]
  Output: [1, 4, 3072]
  Tolerance: 0.0001
  Expected: [ 0.755197, 0.47362, -0.270116, -0.0604766, 0.100855 ... ]
  Actual:   [ 0.755198, 0.47362, -0.270116, -0.0604767, 0.100855 ... ]
  Max Error:  2.02656e-06
  Mean Error: 1.9643e-07
  Time: 18.682 ms
  Result: PASS

LayerNorm - Standard (2x4x768)
  Input:  [2, 4, 768]
  Output: [2, 4, 768]
  Tolerance: 0.0001
  Expected: [ 0.514371, -0.127667, 0.667024, 1.5521, -0.224622 ... ]
  Actual:   [ 0.514371, -0.127667, 0.667024, 1.5521, -0.224622 ... ]
  Max Error:  1.66893e-06
  Mean Error: 1.3325e-07
  Time: 17.293 ms
  Result: PASS

============================================================
OVERALL TEST SUMMARY
============================================================
Total tests run: 3
Tests passed: 3
Tests failed: 0

✓ ALL TESTS PASSED!
```

## Adding New Tests

### Step 1: Create Python Generator in `utils/test_data_generators/`

```python
from json_exporter import export_test_data
import numpy as np

def your_layer_reference(input_data, weight, bias):
    # Reference implementation
    return output_data

def main():
    np.random.seed(42)

    # Generate test data
    input_data = np.random.randn(batch, seq, dim).astype(np.float32)
    weight = np.random.randn(out_dim, in_dim).astype(np.float32)
    bias = np.random.randn(out_dim).astype(np.float32)

    output_data = your_layer_reference(input_data, weight, bias)

    # Export (no node_config needed)
    export_test_data(
        output_path="../../assets/test_data/yourtest.json",
        input_data=input_data,
        output_data=output_data,
        parameters={"weight": weight, "bias": bias}
    )

if __name__ == "__main__":
    main()
```

### Step 2: Run Generator

```bash
cd utils/test_data_generators
python generate_yourtest.py
```

### Step 3: Register in C++

Add to `runTests.cpp`:

```cpp
void registerTests() {
    // ... existing tests

    addTest<YourNodeType>(
        "YourTest - Description (shape)",
        PROJECT_CURRENT_DIR "/assets/test_data/yourtest.json",
        constructor_arg1, constructor_arg2);  // Pass constructor args here
}
```

### Step 4: Add Template Instantiation (if new node type)

Add to bottom of `graphTest.cpp`:

```cpp
template class GraphTest<YourNodeType>;
```

### Step 5: Add Parameter Mapping (if needed)

If your node uses different slot names, add mapping in `graphTest.cpp::loadTestDataFromJSON()`:

```cpp
// Map to appropriate slot name based on node type
if constexpr (std::is_same_v<T, YourNodeType>) {
    weightParam.slotName = "your_custom_slot_name";
} else if constexpr (std::is_same_v<T, LayerNormNode>) {
    weightParam.slotName = "scale";
} else {
    weightParam.slotName = "weight";
}
```

### Step 6: Build and Test

```bash
cmake --build ../build --config Debug --target gpt2-unit-tests
../bin/debug/gpt2-unit-tests.exe
```

## Tolerance Configuration

Default tolerance is `0.0001f`. To customize per test:

```cpp
void registerTests() {
    auto test = std::make_unique<GraphTest<LinearNode>>(
        "Linear - High Precision Test",
        PROJECT_CURRENT_DIR "/assets/test_data/linear_test.json",
        768, 3072);
    test->setTolerance(1e-6f);  // Stricter tolerance
    tests.push_back(std::move(test));
}
```

## Troubleshooting

### Test Fails with "Data size mismatch"

Check that your JSON input/output shapes match what the C++ layer expects.

### "invalid map<K, T> key" error

Parameter slot name mismatch. Check that your node's slot names match the JSON parameter names, or add custom mapping in `loadTestDataFromJSON()`.

### Parameters not loading

Verify that parameter names in JSON (`"weight"`, `"bias"`) match or are mapped to the correct slot names for your node type.

### High error values / Value mismatch

1. Verify your Python reference implementation matches the GPU shader logic
2. Check tensor shapes (especially weight matrix transpose for Linear layers)
3. Consider numerical precision differences between CPU (float64) and GPU (float32)

### Weight shape issues for Linear layers

Linear layers expect weight shape `[out_features, in_features]` and compute `Y = X @ W^T + b`.
Ensure your Python generator creates weights with correct shape:

```python
weight = np.random.randn(out_features, in_features).astype(np.float32)
output = input_data @ weight.T + bias  # Transpose during computation
```

## Architecture Benefits

1. **No boilerplate**: No derived test classes needed
2. **Variadic templates**: Constructor arguments passed directly through `addTest()`
3. **Python flexibility**: Use NumPy/PyTorch for reference implementations
4. **Clean separation**:
   - Test data (JSON) in `assets/`
   - Generation scripts (Python) in `utils/`
   - Test logic (C++) in `test/`
5. **Type safety**: Compile-time checking with templates
6. **Easy maintenance**: Add tests with one line + JSON file
7. **Automatic parameter mapping**: Framework handles slot name differences

## Design Principles

1. **Test data generation scripts belong in `utils/`, not `assets/`**
2. **JSON contains only data, not configuration** (constructor args in C++)
3. **One line to register a test** (just `addTest<NodeType>(name, path, args...)`)
4. **No factory patterns needed** (variadic templates handle everything)
5. **Slot name mapping is automatic** (handled by GraphTest internally)

#ifndef TEST_BASE_H
#define TEST_BASE_H

#include "../core/neuralNet.h"
#include "../core/vulkanApp.h"
#include "../core/globalContext.h"
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <cmath>

using namespace vk;

/**
 * Test Interface - For polymorphic test management
 */
class ITest {
public:
    virtual ~ITest() = default;
    virtual bool execute() = 0;
    virtual bool passed() const = 0;
    virtual std::string getName() const = 0;
    virtual std::string getError() const = 0;
};

/**
 * Test Base Class (Unified)
 *
 * Template-based test class for testing layers, NodeGroups, or entire neural networks.
 * Uses smart pointers for automatic memory management.
 *
 * Design Philosophy:
 * - Template parameter T can be any Node-derived type (Node, NodeGroup, etc.)
 * - Test classes hold smart pointers to layers/subgraphs being tested
 * - Test classes hold input/output tensors as member variables
 * - Separate initialization functions for clear separation of concerns
 *
 * Test Lifecycle:
 * 1. createGraph() - Initialize layer/subgraph and create computation graph
 * 2. setupInputs() - Initialize test input data (CPU tensors)
 * 3. setupParameters() - Initialize parameter data like weight, bias, etc. (CPU tensors)
 * 4. setupExpectedOutputs() - Initialize expected output data (CPU tensors)
 * 5. run() - Convert CPU to GPU tensors, execute computation
 * 6. verifyResults() - Compare actual outputs with expected outputs
 */
template<typename T>
class TestBase : public ITest {
protected:
    std::string testName;

    // Layer/NodeGroup being tested (smart pointer for automatic cleanup)
    std::unique_ptr<T> targetGraph;

    // NeuralNet wrapper (automatically created to safely run the layer)
    std::unique_ptr<NeuralNet> network;

    // CPU-side test data (set by derived classes)
    struct CPUTensorData {
        std::vector<uint32_t> shape;
        std::vector<float> data;
        std::string slotName;  // For parameters: "weight", "bias", "scale", "shift", etc.
    };

    CPUTensorData cpuInput;
    CPUTensorData cpuExpectedOutput;
    std::vector<CPUTensorData> cpuParameters;  // For weight, bias, scale, shift, etc.

    // GPU-side test data (actual output from network)
    std::vector<Tensor> actualOutputTensors;

    // Test configuration
    float tolerance = 1e-4f;  // Default tolerance for float comparison

    // Test status
    bool testPassed = false;
    std::string errorMessage;

    // ========================================================================
    // Helper Functions (declarations only)
    // ========================================================================

    // Connect targetGraph to NeuralNet wrapper
    void connectToNetwork();

    // CPU to GPU conversion helpers
    void convertInputToGPU();
    void convertParametersToGPU();
    void validateExpectedOutput();

    // Tensor helpers
    std::vector<float> readTensorData(const Tensor& tensor);
    Tensor createInputTensor(const std::vector<uint32_t>& shape,
                            const std::vector<float>& data);
    Tensor createExpectedTensor(const std::vector<uint32_t>& shape,
                                const std::vector<float>& data);

    // Verification helpers
    struct ErrorMetrics {
        float maxError;
        float meanError;
    };

    ErrorMetrics assertClose(const std::vector<float>& actual,
                             const std::vector<float>& expected,
                             float tol,
                             const std::string& msg = "");
    void verifyTensorOutput(const Tensor& actual,
                           const Tensor& expected,
                           const std::string& tensorName = "output");
    void verifyResults();

    // Output helpers
    void printTestHeader();
    void printTestConfig();
    void printTestResult(double duration_ms);
    void printComparisonValues(const std::vector<float>& actual, const std::vector<float>& expected);
    void printErrorMetrics(const ErrorMetrics& metrics);
    std::string shapeToString(const std::vector<uint32_t>& shape);

public:
    TestBase(const std::string& name);
    virtual ~TestBase() = default;

    // ========================================================================
    // Test Lifecycle Methods (to be overridden by derived classes)
    // ========================================================================

    virtual void createGraph() = 0;
    virtual void setupInputs() = 0;
    virtual void setupParameters() = 0;
    virtual void setupExpectedOutputs() = 0;

    // run() is now provided by TestBase (no need to override)
    void run();

    // ========================================================================
    // Test Execution
    // ========================================================================

    bool execute();

    // ========================================================================
    // Configuration & Getters
    // ========================================================================

    void setTolerance(float tol);
    bool passed() const;
    std::string getName() const;
    std::string getError() const;
    T* getLayer();
    const T* getLayer() const;
};

#endif // TEST_BASE_H

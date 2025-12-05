#ifndef GRAPH_TEST_H
#define GRAPH_TEST_H

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
 * Graph Test Class (JSON-based)
 *
 * Template-based test class for testing computational graph nodes/groups.
 * Loads all configuration from JSON. No derived classes needed.
 */
template<typename T>
class GraphTest : public ITest {
protected:
    std::string testName;
    std::string jsonPath;

    // Layer/NodeGroup being tested
    std::unique_ptr<T> targetGraph;

    // NeuralNet wrapper
    std::unique_ptr<NeuralNet> network;

    // CPU-side test data
    struct CPUTensorData {
        std::vector<uint32_t> shape;
        std::vector<float> data;
        std::string slotName;
    };

    CPUTensorData cpuInput;
    CPUTensorData cpuExpectedOutput;
    std::vector<CPUTensorData> cpuParameters;

    // GPU-side test data
    std::vector<Tensor> actualOutputTensors;

    // Test configuration
    float tolerance = 1e-4f;

    // Test status
    bool testPassed = false;
    std::string errorMessage;

    // ========================================================================
    // Helper Functions
    // ========================================================================

    void connectToNetwork();
    void convertInputToGPU();
    void convertParametersToGPU();
    void validateExpectedOutput();

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
    void verifyResults();

    // Output helpers
    void printTestHeader();
    void printTestConfig();
    void printTestResult(double duration_ms);
    void printComparisonValues(const std::vector<float>& actual, const std::vector<float>& expected);
    void printErrorMetrics(const ErrorMetrics& metrics);
    std::string shapeToString(const std::vector<uint32_t>& shape);

    // JSON loading
    void loadTestDataFromJSON();

public:
    // Constructor: takes test name, JSON path, and node constructor arguments
    template<typename... Args>
    GraphTest(const std::string& name, const std::string& jsonPath, Args&&... args)
        : testName(name), jsonPath(jsonPath) {
        loadTestDataFromJSON();
        targetGraph = std::make_unique<T>(std::forward<Args>(args)...);
    }

    virtual ~GraphTest() = default;

    // run() is provided by GraphTest
    void run();

    // Test Execution
    bool execute();

    // Configuration & Getters
    void setTolerance(float tol);
    bool passed() const;
    std::string getName() const;
    std::string getError() const;
    T* getLayer();
    const T* getLayer() const;
};

#endif // GRAPH_TEST_H

#include "testBase.h"
#include "../model/attention/attentionNode.h"
#include "../model/transformerBlock/transformer.h"
#include <chrono>

using namespace vk;

// ============================================================================
// TestBase Implementation
// ============================================================================

template<typename T>
TestBase<T>::TestBase(const std::string& name) : testName(name) {}

// ============================================================================
// Helper Functions Implementation
// ============================================================================

template<typename T>
void TestBase<T>::connectToNetwork() {
    if (!targetGraph) {
        throw std::runtime_error("targetGraph is null - call createGraph() first");
    }

    network = std::make_unique<NeuralNet>(netGlobalDevice, 1, 1);
    network->input(0) - *targetGraph - network->output(0);
}

template<typename T>
std::vector<float> TestBase<T>::readTensorData(const Tensor& tensor) {
    if (!tensor.validShape()) {
        throw std::runtime_error("Invalid tensor shape");
    }

    uint32_t numElements = tensor.numElements();

    Buffer outBuffer = netGlobalDevice.createBuffer({
        .size = numElements * sizeof(float),
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });

    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuffer, tensor.buffer())
        .end()
        .submit()
        .wait();

    float* data = (float*)outBuffer.map();
    std::vector<float> result(data, data + numElements);
    outBuffer.unmap();
    return result;
}

template<typename T>
Tensor TestBase<T>::createInputTensor(const std::vector<uint32_t>& shape,
                                      const std::vector<float>& data) {
    Tensor tensor(shape);
    if (data.size() != tensor.numElements()) {
        throw std::runtime_error(
            "Data size mismatch: expected " + std::to_string(tensor.numElements()) +
            ", got " + std::to_string(data.size())
        );
    }
    tensor.set(data);
    return tensor;
}

template<typename T>
Tensor TestBase<T>::createExpectedTensor(const std::vector<uint32_t>& shape,
                                         const std::vector<float>& data) {
    return createInputTensor(shape, data);
}

template<typename T>
typename TestBase<T>::ErrorMetrics TestBase<T>::assertClose(const std::vector<float>& actual,
                                                             const std::vector<float>& expected,
                                                             float tol,
                                                             const std::string& msg) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("Size mismatch: " + msg);
    }

    size_t numMismatches = 0;
    float maxDiff = 0.0f;
    float sumDiff = 0.0f;
    size_t maxDiffIdx = 0;

    for (size_t i = 0; i < actual.size(); ++i) {
        float diff = std::abs(actual[i] - expected[i]);
        sumDiff += diff;
        if (diff > maxDiff) {
            maxDiff = diff;
            maxDiffIdx = i;
        }
        if (diff > tol) {
            numMismatches++;
        }
    }

    if (numMismatches > 0) {
        throw std::runtime_error(
            "Value mismatch: " + std::to_string(numMismatches) +
            " values exceed tolerance " + std::to_string(tol) + " " + msg
        );
    }

    float meanDiff = actual.size() > 0 ? sumDiff / actual.size() : 0.0f;
    return {maxDiff, meanDiff};
}

template<typename T>
void TestBase<T>::verifyTensorOutput(const Tensor& actual,
                                     const Tensor& expected,
                                     const std::string& tensorName) {
    // Check shape
    if (actual.shape() != expected.shape()) {
        throw std::runtime_error(
            tensorName + " shape mismatch: expected " +
            shapeToString(expected.shape()) + ", got " +
            shapeToString(actual.shape())
        );
    }

    std::vector<float> actualData = readTensorData(actual);

    std::vector<float> expectedData;
    if (expected.hasHostData()) {
        float* hostPtr = const_cast<Tensor&>(expected).hostData();
        uint32_t numElements = expected.numElements();
        expectedData.assign(hostPtr, hostPtr + numElements);
    } else {
        expectedData = readTensorData(expected);
    }

    printComparisonValues(actualData, expectedData);

    ErrorMetrics metrics = assertClose(actualData, expectedData, tolerance, tensorName);
    printErrorMetrics(metrics);
}

template<typename T>
void TestBase<T>::verifyResults() {
    if (actualOutputTensors.size() != 1) {
        throw std::runtime_error("Expected exactly 1 output tensor");
    }

    if (actualOutputTensors[0].shape() != cpuExpectedOutput.shape) {
        throw std::runtime_error(
            "Output shape mismatch: expected " +
            shapeToString(cpuExpectedOutput.shape) + ", got " +
            shapeToString(actualOutputTensors[0].shape())
        );
    }

    std::vector<float> actualData = readTensorData(actualOutputTensors[0]);

    printComparisonValues(actualData, cpuExpectedOutput.data);

    ErrorMetrics metrics = assertClose(actualData, cpuExpectedOutput.data, tolerance, "output");
    printErrorMetrics(metrics);
}

template<typename T>
std::string TestBase<T>::shapeToString(const std::vector<uint32_t>& shape) {
    std::string result = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        result += std::to_string(shape[i]);
        if (i < shape.size() - 1) result += ", ";
    }
    result += "]";
    return result;
}

// ============================================================================
// CPU to GPU Conversion Helpers
// ============================================================================

template<typename T>
void TestBase<T>::convertInputToGPU() {
    if (cpuInput.data.empty()) {
        throw std::runtime_error("CPU input data not set");
    }
    Tensor inputTensor = createInputTensor(cpuInput.shape, cpuInput.data);
    network->input(0)["in0"] = inputTensor;
}

template<typename T>
void TestBase<T>::convertParametersToGPU() {
    for (const auto& param : cpuParameters) {
        if (param.slotName.empty()) {
            throw std::runtime_error("Parameter slot name not set");
        }
        Tensor paramTensor = createInputTensor(param.shape, param.data);
        (*targetGraph)[param.slotName] = paramTensor;
    }
}

template<typename T>
void TestBase<T>::validateExpectedOutput() {
    if (cpuExpectedOutput.data.empty()) {
        throw std::runtime_error("CPU expected output data not set");
    }
}

// ============================================================================
// Output Helpers
// ============================================================================

template<typename T>
void TestBase<T>::printTestHeader() {
    std::cout << "\n" << testName << std::endl;
}

template<typename T>
void TestBase<T>::printTestConfig() {
    std::cout << "  Input Shape:  " << shapeToString(cpuInput.shape) << std::endl;
    std::cout << "  Output Shape: " << shapeToString(cpuExpectedOutput.shape) << std::endl;
    std::cout << "  Tolerance: " << tolerance << std::endl;
}

template<typename T>
void TestBase<T>::printTestResult(double duration_ms) {
    std::cout << "  Time: " << duration_ms << " ms" << std::endl;
    std::cout << "  Result: PASS" << std::endl;
}

template<typename T>
void TestBase<T>::printComparisonValues(const std::vector<float>& actual, const std::vector<float>& expected) {
    size_t numToPrint = std::min(size_t(5), actual.size());

    std::cout << "  Expected: [ ";
    for (size_t i = 0; i < numToPrint; ++i) {
        std::cout << expected[i];
        if (i < numToPrint - 1) std::cout << ", ";
    }
    std::cout << " ... ]" << std::endl;

    std::cout << "  Actual:   [ ";
    for (size_t i = 0; i < numToPrint; ++i) {
        std::cout << actual[i];
        if (i < numToPrint - 1) std::cout << ", ";
    }
    std::cout << " ... ]" << std::endl;
}

template<typename T>
void TestBase<T>::printErrorMetrics(const ErrorMetrics& metrics) {
    std::cout << "  Max Error:  " << metrics.maxError << std::endl;
    std::cout << "  Mean Error: " << metrics.meanError << std::endl;
}

// ============================================================================
// Test Execution Implementation
// ============================================================================

template<typename T>
void TestBase<T>::run() {
    if (!network) {
        throw std::runtime_error("Network not initialized - call connectToNetwork() first");
    }

    Tensor& inputTensor = network->input(0)["in0"];

    if (!inputTensor.validShape()) {
        throw std::runtime_error("Input tensor not set - call setupInputs() first");
    }

    actualOutputTensors = (*network)(inputTensor);
}

template<typename T>
bool TestBase<T>::execute() {
    printTestHeader();

    try {
        createGraph();
        connectToNetwork();
        setupInputs();
        setupParameters();
        setupExpectedOutputs();

        convertInputToGPU();
        convertParametersToGPU();
        validateExpectedOutput();

        printTestConfig();

        auto start = std::chrono::high_resolution_clock::now();
        run();
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

        verifyResults();

        printTestResult(duration);

        testPassed = true;
        return true;

    } catch (const std::exception& e) {
        testPassed = false;
        errorMessage = e.what();
        std::cout << "  Result: FAIL - " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// Configuration & Getters Implementation
// ============================================================================

template<typename T>
void TestBase<T>::setTolerance(float tol) {
    tolerance = tol;
}

template<typename T>
bool TestBase<T>::passed() const {
    return testPassed;
}

template<typename T>
std::string TestBase<T>::getName() const {
    return testName;
}

template<typename T>
std::string TestBase<T>::getError() const {
    return errorMessage;
}

template<typename T>
T* TestBase<T>::getLayer() {
    return targetGraph.get();
}

template<typename T>
const T* TestBase<T>::getLayer() const {
    return targetGraph.get();
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

// Instantiate for all node types we use in tests
template class TestBase<LinearNode>;
template class TestBase<LayerNormNode>;

// Add more instantiations as needed:
// template class TestBase<GELUNode>;
// template class TestBase<AddNode>;
// template class TestBase<SoftmaxNode>;
// template class TestBase<FeedForwardNode>;
// template class TestBase<MultiHeadAttentionNode>;
// template class TestBase<TransformerBlock>;

#include "graphTest.h"
#include "../core/jsonParser.h"
#include "../model/attention/attentionNode.h"
#include "../model/transformerBlock/transformer.h"
#include <chrono>

using namespace vk;

// ============================================================================
// JSON Loading
// ============================================================================

template<typename T>
void GraphTest<T>::loadTestDataFromJSON() {
    JsonParser json(jsonPath.c_str());

    // Load input
    std::vector<uint32_t> inputShape;
    cpuInput.data = json["input"].parseNDArray(inputShape);
    cpuInput.shape = inputShape;

    // Load parameters (if exist)
    // Try to load common parameter names: weight, bias
    try {
        std::vector<uint32_t> weightShape;
        auto weightData = json["parameters"]["weight"].parseNDArray(weightShape);
        CPUTensorData weightParam;
        weightParam.shape = weightShape;
        weightParam.data = weightData;

        // Map to appropriate slot name based on node type
        if constexpr (std::is_same_v<T, LayerNormNode>) {
            weightParam.slotName = "scale";
        } else {
            weightParam.slotName = "weight";
        }
        cpuParameters.push_back(weightParam);
    } catch (...) {
        // No weight parameter
    }

    try {
        std::vector<uint32_t> biasShape;
        auto biasData = json["parameters"]["bias"].parseNDArray(biasShape);
        CPUTensorData biasParam;
        biasParam.shape = biasShape;
        biasParam.data = biasData;

        // Map to appropriate slot name based on node type
        if constexpr (std::is_same_v<T, LayerNormNode>) {
            biasParam.slotName = "shift";
        } else {
            biasParam.slotName = "bias";
        }
        cpuParameters.push_back(biasParam);
    } catch (...) {
        // No bias parameter
    }

    // Load expected output
    std::vector<uint32_t> outputShape;
    cpuExpectedOutput.data = json["output"].parseNDArray(outputShape);
    cpuExpectedOutput.shape = outputShape;
}

// ============================================================================
// Helper functions
// ============================================================================

template<typename T>
void GraphTest<T>::connectToNetwork() {
    if (!targetGraph) {
        throw std::runtime_error("targetGraph is null");
    }

    network = std::make_unique<NeuralNet>(netGlobalDevice, 1, 1);
    network->input(0) - *targetGraph - network->output(0);
}

template<typename T>
std::vector<float> GraphTest<T>::readTensorData(const Tensor& tensor) {
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
Tensor GraphTest<T>::createInputTensor(const std::vector<uint32_t>& shape,
                                      const std::vector<float>& data) {
    Tensor tensor(shape);
    if (data.size() != tensor.numElements()) {
        throw std::runtime_error("Data size mismatch");
    }
    tensor.set(data);
    return tensor;
}

template<typename T>
Tensor GraphTest<T>::createExpectedTensor(const std::vector<uint32_t>& shape,
                                         const std::vector<float>& data) {
    return createInputTensor(shape, data);
}

template<typename T>
void GraphTest<T>::convertInputToGPU() {
    if (cpuInput.data.empty()) {
        throw std::runtime_error("CPU input data not set");
    }
    Tensor inputTensor = createInputTensor(cpuInput.shape, cpuInput.data);
    network->input(0)["in0"] = inputTensor;
}

template<typename T>
void GraphTest<T>::convertParametersToGPU() {
    for (const auto& param : cpuParameters) {
        if (param.slotName.empty()) {
            throw std::runtime_error("Parameter slot name not set");
        }
        Tensor paramTensor = createInputTensor(param.shape, param.data);
        (*targetGraph)[param.slotName] = paramTensor;
    }
}

template<typename T>
void GraphTest<T>::validateExpectedOutput() {
    if (cpuExpectedOutput.data.empty()) {
        throw std::runtime_error("CPU expected output data not set");
    }
}

template<typename T>
typename GraphTest<T>::ErrorMetrics GraphTest<T>::assertClose(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float tol,
    const std::string& msg) {

    if (actual.size() != expected.size()) {
        throw std::runtime_error("Size mismatch: " + msg);
    }

    size_t numMismatches = 0;
    float maxDiff = 0.0f;
    float sumDiff = 0.0f;

    for (size_t i = 0; i < actual.size(); ++i) {
        float diff = std::abs(actual[i] - expected[i]);
        sumDiff += diff;
        if (diff > maxDiff) {
            maxDiff = diff;
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
void GraphTest<T>::verifyResults() {
    if (actualOutputTensors.size() != 1) {
        throw std::runtime_error("Expected exactly 1 output tensor");
    }

    std::vector<float> actualData = readTensorData(actualOutputTensors[0]);
    printComparisonValues(actualData, cpuExpectedOutput.data);

    ErrorMetrics metrics = assertClose(actualData, cpuExpectedOutput.data, tolerance, "output");
    printErrorMetrics(metrics);
}

template<typename T>
std::string GraphTest<T>::shapeToString(const std::vector<uint32_t>& shape) {
    std::string result = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        result += std::to_string(shape[i]);
        if (i < shape.size() - 1) result += ", ";
    }
    result += "]";
    return result;
}

template<typename T>
void GraphTest<T>::printTestHeader() {
    std::cout << "\n" << testName << std::endl;
}

template<typename T>
void GraphTest<T>::printTestConfig() {
    std::cout << "  Input:  " << shapeToString(cpuInput.shape) << std::endl;
    std::cout << "  Output: " << shapeToString(cpuExpectedOutput.shape) << std::endl;
    std::cout << "  Tolerance: " << tolerance << std::endl;
}

template<typename T>
void GraphTest<T>::printComparisonValues(const std::vector<float>& actual, const std::vector<float>& expected) {
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
void GraphTest<T>::printErrorMetrics(const ErrorMetrics& metrics) {
    std::cout << "  Max Error:  " << metrics.maxError << std::endl;
    std::cout << "  Mean Error: " << metrics.meanError << std::endl;
}

template<typename T>
void GraphTest<T>::printTestResult(double duration_ms) {
    std::cout << "  Time: " << duration_ms << " ms" << std::endl;
    std::cout << "  Result: PASS" << std::endl;
}

template<typename T>
void GraphTest<T>::run() {
    if (!network) {
        throw std::runtime_error("Network not initialized");
    }

    Tensor& inputTensor = network->input(0)["in0"];

    if (!inputTensor.validShape()) {
        throw std::runtime_error("Input tensor not set");
    }

    actualOutputTensors = (*network)(inputTensor);
}

template<typename T>
bool GraphTest<T>::execute() {
    printTestHeader();

    try {
        connectToNetwork();
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
// Getters
// ============================================================================

template<typename T>
void GraphTest<T>::setTolerance(float tol) {
    tolerance = tol;
}

template<typename T>
bool GraphTest<T>::passed() const {
    return testPassed;
}

template<typename T>
std::string GraphTest<T>::getName() const {
    return testName;
}

template<typename T>
std::string GraphTest<T>::getError() const {
    return errorMessage;
}

template<typename T>
T* GraphTest<T>::getLayer() {
    return targetGraph.get();
}

template<typename T>
const T* GraphTest<T>::getLayer() const {
    return targetGraph.get();
}

// ============================================================================
// Template Instantiations
// ============================================================================

template class GraphTest<LinearNode>;
template class GraphTest<LayerNormNode>;
template class GraphTest<GELUNode>;

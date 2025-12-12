// ============================================================================
// GraphTest Template Implementation
//
// This file contains the implementation of GraphTest template methods.
// It is included at the end of graphTest.h to enable header-only templates.
// ============================================================================

#include "jsonParser.h"
#include <chrono>

// ============================================================================
// JSON Loading
// ============================================================================

template<typename T>
void GraphTest<T>::loadInput(JsonParser& json) {
    std::vector<uint32_t> inputShape;
    cpuInput.data = json["input"].parseNDArray(inputShape);
    cpuInput.shape = inputShape;
}

template<typename T>
void GraphTest<T>::loadParameters(JsonParser& json) {
    try {
        auto paramKeys = json["parameters"].getKeys();
        for (const auto& key : paramKeys) {
            std::vector<uint32_t> paramShape;
            auto paramData = json["parameters"][key].parseNDArray(paramShape);

            CPUTensorData param;
            param.shape = paramShape;
            param.data = paramData;
            param.slotName = key;

            cpuParameters.push_back(param);
        }
    } catch (...) {
    }
}

template<typename T>
void GraphTest<T>::loadExpectedOutput(JsonParser& json) {
    std::vector<uint32_t> outputShape;
    cpuExpectedOutput.data = json["output"].parseNDArray(outputShape);
    cpuExpectedOutput.shape = outputShape;
}

template<typename T>
void GraphTest<T>::loadTestDataFromJSON() {
    JsonParser json(jsonPath.c_str());

    // Check if this is sequence mode
    try {
        auto mode = json["mode"].parseString();
        if (mode == "sequence") {
            isSequenceMode = true;
            loadConfig(json);
            loadSequenceSteps(json);
            loadParameters(json);
            return;
        }
    } catch (...) {
        // No "mode" field - standard test
    }

    // Standard single-step test
    loadInput(json);
    loadParameters(json);
    loadExpectedOutput(json);
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
        paramTensor.setConstant(true);  // Mark as constant so it persists across multiple runs
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
        convertParametersToGPU();

        if (isSequenceMode) {
            // Sequence mode execution
            auto start = std::chrono::high_resolution_clock::now();
            runSequence();
            auto end = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

            verifySequenceResults();
            printTestResult(duration);
        } else {
            // Standard single-step execution
            convertInputToGPU();
            validateExpectedOutput();
            printTestConfig();

            auto start = std::chrono::high_resolution_clock::now();
            run();
            auto end = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

            verifyResults();
            printTestResult(duration);
        }

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
// Sequence Mode Support
// ============================================================================

template<typename T>
void GraphTest<T>::loadConfig(JsonParser& json) {
    try {
        auto config = json["config"];
        maxCacheLength = config["max_cache_length"].parseInt();
    } catch (...) {
        maxCacheLength = 1024;  // Default
    }
}

template<typename T>
void GraphTest<T>::loadSequenceSteps(JsonParser& json) {
    auto stepsArray = json["steps"];
    int numSteps = stepsArray.size();

    for (int i = 0; i < numSteps; ++i) {
        auto step = stepsArray[i];

        SequenceStep seqStep;

        // Load input
        std::vector<uint32_t> inputShape;
        seqStep.input.data = step["input"].parseNDArray(inputShape);
        seqStep.input.shape = inputShape;

        // Load expected output
        std::vector<uint32_t> outputShape;
        seqStep.expectedOutput.data = step["output"].parseNDArray(outputShape);
        seqStep.expectedOutput.shape = outputShape;

        // Load cache length
        seqStep.cacheLength = step["cache_length"].parseInt();

        sequenceSteps.push_back(seqStep);
    }
}

template<typename T>
void GraphTest<T>::runSequence() {
    std::cout << "  Running sequence test with " << sequenceSteps.size() << " steps..." << std::endl;

    // Initialize KV cache for MultiHeadAttentionNode
    // This requires access to cache - we'll handle this in specialized template
    // For now, process each step and collect outputs

    actualOutputTensors.clear();

    for (size_t i = 0; i < sequenceSteps.size(); ++i) {
        auto& step = sequenceSteps[i];

        std::cout << "    Step " << (i+1) << ": cache_length=" << step.cacheLength << std::endl;

        // Create input tensor
        Tensor inputTensor = createInputTensor(step.input.shape, step.input.data);
        network->input(0)["in0"] = inputTensor;

        // Run inference
        auto outputs = (*network)(inputTensor);

        // Store output
        actualOutputTensors.push_back(outputs[0]);
    }
}

template<typename T>
void GraphTest<T>::verifySequenceResults() {
    std::cout << "\n  Verifying sequence results..." << std::endl;

    if (actualOutputTensors.size() != sequenceSteps.size()) {
        throw std::runtime_error("Output count mismatch");
    }

    float maxErrorOverall = 0.0f;
    float meanErrorOverall = 0.0f;
    int totalElements = 0;

    for (size_t i = 0; i < sequenceSteps.size(); ++i) {
        auto& step = sequenceSteps[i];
        auto actualData = readTensorData(actualOutputTensors[i]);

        try {
            auto metrics = assertClose(actualData, step.expectedOutput.data, tolerance,
                                       "Step " + std::to_string(i+1));

            maxErrorOverall = std::max(maxErrorOverall, metrics.maxError);
            meanErrorOverall += metrics.meanError * actualData.size();
            totalElements += actualData.size();

            std::cout << "    Step " << (i+1) << ": max_error=" << metrics.maxError
                      << ", mean_error=" << metrics.meanError << std::endl;
        } catch (const std::exception& e) {
            // DEBUG: Print first 16 actual and expected values for failed step
            std::cout << "    Step " << (i+1) << " FAILED - printing debug values:" << std::endl;
            size_t numToPrint = std::min(size_t(16), actualData.size());
            std::cout << "      Actual[0-15]:   ";
            for (size_t j = 0; j < numToPrint; ++j) {
                std::cout << actualData[j];
                if (j < numToPrint - 1) std::cout << ", ";
            }
            std::cout << std::endl;
            std::cout << "      Expected[0-15]: ";
            for (size_t j = 0; j < numToPrint; ++j) {
                std::cout << step.expectedOutput.data[j];
                if (j < numToPrint - 1) std::cout << ", ";
            }
            std::cout << std::endl;
            throw;
        }
    }

    meanErrorOverall /= totalElements;

    std::cout << "\n  Overall Sequence Results:" << std::endl;
    std::cout << "    Max Error:  " << maxErrorOverall << std::endl;
    std::cout << "    Mean Error: " << meanErrorOverall << std::endl;

    if (maxErrorOverall >= tolerance) {
        throw std::runtime_error("Sequence test failed: max error exceeds tolerance");
    }
}

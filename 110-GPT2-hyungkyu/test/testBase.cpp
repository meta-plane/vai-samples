#include "testBase.h"
#include "../model/attention/attentionNode.h"
#include "../model/transformerBlock/transformer.h"

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

    // Create NeuralNet with 1 input and 1 output
    network = std::make_unique<NeuralNet>(netGlobalDevice, 1, 1);

    // Connect: input(0) -> targetGraph -> output(0)
    network->input(0) - *targetGraph - network->output(0);
}

template<typename T>
std::vector<float> TestBase<T>::readTensorData(const Tensor& tensor) {
    if (!tensor.validShape()) {
        throw std::runtime_error("Invalid tensor shape");
    }

    uint32_t numElements = tensor.numElements();

    // Create host-visible buffer
    Buffer outBuffer = netGlobalDevice.createBuffer({
        .size = numElements * sizeof(float),
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });

    // Copy tensor data to host buffer
    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuffer, tensor.buffer())
        .end()
        .submit()
        .wait();

    // Map and read data
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
void TestBase<T>::assertClose(const std::vector<float>& actual,
                              const std::vector<float>& expected,
                              float tol,
                              const std::string& msg) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("Size mismatch: " + msg);
    }

    size_t numMismatches = 0;
    float maxDiff = 0.0f;
    size_t maxDiffIdx = 0;

    for (size_t i = 0; i < actual.size(); ++i) {
        float diff = std::abs(actual[i] - expected[i]);
        if (diff > maxDiff) {
            maxDiff = diff;
            maxDiffIdx = i;
        }
        if (diff > tol) {
            numMismatches++;
            // Print first few mismatches
            if (numMismatches <= 5) {
                std::cout << "      Mismatch at [" << i << "]: expected "
                          << expected[i] << ", got " << actual[i]
                          << " (diff: " << diff << ")" << std::endl;
            }
        }
    }

    if (numMismatches > 0) {
        std::cout << "      Total mismatches: " << numMismatches << " / " << actual.size() << std::endl;
        std::cout << "      Max diff: " << maxDiff << " at index " << maxDiffIdx << std::endl;
        throw std::runtime_error(
            "Value mismatch: " + std::to_string(numMismatches) +
            " values exceed tolerance " + std::to_string(tol) + " " + msg
        );
    }
}

template<typename T>
void TestBase<T>::verifyTensorOutput(const Tensor& actual,
                                     const Tensor& expected,
                                     const std::string& tensorName) {
    std::cout << "    Checking shape..." << std::endl;

    // Check shape
    if (actual.shape() != expected.shape()) {
        throw std::runtime_error(
            tensorName + " shape mismatch: expected " +
            shapeToString(expected.shape()) + ", got " +
            shapeToString(actual.shape())
        );
    }

    std::cout << "    Reading actual data from GPU..." << std::endl;
    // Read actual data from GPU
    std::vector<float> actualData = readTensorData(actual);

    std::cout << "    Getting expected data from host..." << std::endl;
    // Expected tensor has host data - use it directly
    std::vector<float> expectedData;
    if (expected.hasHostData()) {
        float* hostPtr = const_cast<Tensor&>(expected).hostData();
        uint32_t numElements = expected.numElements();
        expectedData.assign(hostPtr, hostPtr + numElements);
    } else {
        // Fallback: read from GPU if it has device data
        expectedData = readTensorData(expected);
    }

    std::cout << "    Comparing values (tolerance: " << tolerance << ")..." << std::endl;

    // Print first few values for verification
    size_t numToPrint = std::min(size_t(10), actualData.size());
    std::cout << "      First " << numToPrint << " values:" << std::endl;
    std::cout << "      Expected: [ ";
    for (size_t i = 0; i < numToPrint; ++i) {
        std::cout << expectedData[i];
        if (i < numToPrint - 1) std::cout << ", ";
    }
    std::cout << " ]" << std::endl;
    std::cout << "      Actual:   [ ";
    for (size_t i = 0; i < numToPrint; ++i) {
        std::cout << actualData[i];
        if (i < numToPrint - 1) std::cout << ", ";
    }
    std::cout << " ]" << std::endl;

    // Compare values
    assertClose(actualData, expectedData, tolerance, tensorName);
    std::cout << "    ✓ All values match within tolerance!" << std::endl;
}

template<typename T>
void TestBase<T>::verifyAllOutputs() {
    std::cout << "  Verifying " << actualOutputTensors.size() << " outputs..." << std::endl;

    if (actualOutputTensors.size() != expectedOutputTensors.size()) {
        throw std::runtime_error(
            "Output count mismatch: expected " +
            std::to_string(expectedOutputTensors.size()) +
            ", got " + std::to_string(actualOutputTensors.size())
        );
    }

    for (size_t i = 0; i < actualOutputTensors.size(); ++i) {
        std::cout << "  Verifying output " << i << "..." << std::endl;
        std::string tensorName = "output_" + std::to_string(i);
        verifyTensorOutput(actualOutputTensors[i],
                         expectedOutputTensors[i],
                         tensorName);
        std::cout << "  Output " << i << " verified" << std::endl;
    }
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
// Test Execution Implementation
// ============================================================================

template<typename T>
void TestBase<T>::run() {
    std::cout << "Running forward pass..." << std::endl;

    if (!network) {
        throw std::runtime_error("Network not initialized - call connectToNetwork() first");
    }

    if (inputTensors.empty()) {
        throw std::runtime_error("No input tensors - call setupInputs() first");
    }

    // Run inference through NeuralNet (safe execution)
    std::vector<Tensor> outputs = (*network)(inputTensors[0]);

    // Store actual outputs
    actualOutputTensors = outputs;

    std::cout << "  Forward pass complete" << std::endl;
}

template<typename T>
bool TestBase<T>::execute() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test: " << testName << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        createGraph();
        connectToNetwork();  // Automatically connect to NeuralNet
        setupInputs();
        setupExpectedOutputs();
        run();
        verifyResults();

        testPassed = true;
        std::cout << "✓ Test PASSED" << std::endl;
        return true;

    } catch (const std::exception& e) {
        testPassed = false;
        errorMessage = e.what();
        std::cout << "✗ Test FAILED: " << e.what() << std::endl;
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

// Add more instantiations as needed:
// template class TestBase<LayerNormNode>;
// template class TestBase<GELUNode>;
// template class TestBase<AddNode>;
// template class TestBase<SoftmaxNode>;
// template class TestBase<FeedForwardNode>;
// template class TestBase<MultiHeadAttentionNode>;
// template class TestBase<TransformerBlock>;

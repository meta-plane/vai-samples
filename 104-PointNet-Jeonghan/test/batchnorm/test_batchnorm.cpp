/**
 * BatchNorm1D Vulkan Test (SafeTensors-based)
 * Tests BatchNorm1D implementation against PyTorch reference
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include "neuralNet.h"
#include "neuralNodes.h"
#include "vulkanApp.h"
#include "safeTensorsParser.h"

using namespace vk;

/**
 * Simple test network with just BatchNorm1D
 */
class BatchNormTestNet : public NeuralNet {
    BatchNorm1DNode bn;
    
public:
    BatchNormTestNet(Device& device, uint32_t channels)
    : NeuralNet(device, 1, 1)
    , bn(channels)
    {
        input(0) - bn - output(0);
    }
    
    Tensor& operator[](const std::string& name) {
        return bn[name];
    }
};

/**
 * Run BatchNorm1D inference
 */
Tensor eval_batchnorm(const std::vector<float>& inputData, const SafeTensorsParser& json, uint32_t N, uint32_t C) {
    // Create network
    BatchNormTestNet net(netGlobalDevice, C);
    
    // Set parameters from JSON using JsonParserRef constructor
    net["mean"] = Tensor(json["mean"]);
    net["var"] = Tensor(json["var"]);
    net["gamma"] = Tensor(json["gamma"]);
    net["beta"] = Tensor(json["beta"]);
    
    // Prepare network
    net.prepare();
    
    // Create input tensor [N, C]
    Tensor inputTensor = Tensor(N, C).set(inputData);
    
    // Run inference
    auto result = net(inputTensor);
    
    return result[0];
}

void test() {
    void loadShaders();
    loadShaders();
    
    // Load reference data
    SafeTensorsParser json = SafeTensorsParser(PROJECT_CURRENT_DIR"/test/batchnorm/reference.safetensors");
    
    // Extract shape from array [N, C]
    std::vector<float> shape_data = json["shape"].parseNDArray();
    uint32_t N = static_cast<uint32_t>(shape_data[0]);
    uint32_t C = static_cast<uint32_t>(shape_data[1]);
    
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║   BatchNorm1D Vulkan Compute Test       ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n\n";
    
    std::cout << "Test configuration:\n";
    std::cout << "  Shape: [" << N << ", " << C << "]\n";
    std::cout << "  Total values: " << (N * C) << "\n\n";
    
    // Get input data
    std::vector<float> inputData = json["input"].parseNDArray();
    
    // Run inference
    std::cout << "Running BatchNorm1D on GPU...\n";
    Tensor result = eval_batchnorm(inputData, json, N, C);
    
    // Download result from GPU to CPU
    Buffer outBuffer = netGlobalDevice.createBuffer({
        .size = N * C * sizeof(float),
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });
    
    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuffer, result.buffer())
        .end()
        .submit()
        .wait();
    
    float* output = (float*)outBuffer.map();
    
    // Get expected values
    std::vector<float> expected = json["expected"].parseNDArray();
    
    // Compare results
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Comparing results...\n";
    std::cout << std::string(60, '=') << "\n\n";
    
    float maxDiff = 0.0f;
    float avgDiff = 0.0f;
    int mismatches = 0;
    const float tolerance = 1e-4f;
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Index | Expected    | Got         | Diff        | Status\n";
    std::cout << std::string(60, '-') << "\n";
    
    for (size_t i = 0; i < N * C; i++) {
        float diff = std::abs(output[i] - expected[i]);
        maxDiff = std::max(maxDiff, diff);
        avgDiff += diff;
        
        bool match = diff < tolerance;
        if (!match) mismatches++;
        
        // Show first 5, last 5, and all mismatches
        if (!match || i < 5 || i >= N * C - 5) {
            std::cout << std::setw(5) << i << " | "
                      << std::setw(11) << expected[i] << " | "
                      << std::setw(11) << output[i] << " | "
                      << std::setw(11) << diff << " | "
                      << (match ? "✓" : "✗") << "\n";
        } else if (i == 5) {
            std::cout << "  ... (" << (N * C - 10) << " more) ...\n";
        }
    }
    
    avgDiff /= (N * C);
    
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Results Summary:\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "Total values:   " << (N * C) << "\n";
    std::cout << "Mismatches:     " << mismatches << "\n";
    std::cout << "Max difference: " << maxDiff << "\n";
    std::cout << "Avg difference: " << avgDiff << "\n";
    std::cout << "Tolerance:      " << tolerance << "\n";
    
    outBuffer.unmap();
    
    if (mismatches == 0) {
        std::cout << "\n✓ TEST PASSED! All values match within tolerance.\n";
    } else {
        std::cout << "\n✗ TEST FAILED! " << mismatches << " values differ.\n";
        exit(1);
    }
}

int main() {
    test();
    return 0;
}

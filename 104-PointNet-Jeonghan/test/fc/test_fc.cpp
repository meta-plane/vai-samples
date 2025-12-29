/**
 * FullyConnected Vulkan Test (SafeTensors-based)
 * Tests FullyConnectedNode against PyTorch reference
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
 * Test network with FullyConnected layer
 */
class FCTestNet : public NeuralNet {
    FullyConnectedNode fc;
    
public:
    FCTestNet(Device& device, uint32_t in_dim, uint32_t out_dim)
    : NeuralNet(device, 1, 1)
    , fc(in_dim, out_dim)
    {
        // Single FC layer
        input(0) - fc - output(0);
    }
    
    Tensor& operator[](const std::string& name) {
        return fc[name];
    }
};

/**
 * Run FullyConnected inference
 */
Tensor eval_fc(const std::vector<float>& inputData, const SafeTensorsParser& json, 
               uint32_t I, uint32_t O) {
    // Create network
    FCTestNet net(netGlobalDevice, I, O);
    
    // Load weights with proper shapes
    std::vector<float> weight_data = json["weight"].parseNDArray();
    std::vector<float> bias_data = json["bias"].parseNDArray();
    
    // Set parameters with PyTorch format [O, I]
    net["weight"] = Tensor(O, I).set(weight_data);  // [O, I] - PyTorch convention
    net["bias"] = Tensor(O).set(bias_data);         // [O]
    
    // Prepare network
    net.prepare();
    
    // Create input tensor [I]
    Tensor inputTensor = Tensor(I).set(inputData);
    
    // Run inference
    auto result = net(inputTensor);
    
    return result[0];
}

void test() {
    void loadShaders();
    loadShaders();
    
    // Load reference data
    SafeTensorsParser json = SafeTensorsParser(PROJECT_CURRENT_DIR"/test/fc/reference.safetensors");
    
    // Extract shape from array [I, O]
    std::vector<float> shape_data = json["shape"].parseNDArray();
    uint32_t I = static_cast<uint32_t>(shape_data[0]);
    uint32_t O = static_cast<uint32_t>(shape_data[1]);
    
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║   FullyConnected Vulkan Compute Test   ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n\n";
    
    std::cout << "Test configuration:\n";
    std::cout << "  Input dim:  " << I << "\n";
    std::cout << "  Output dim: " << O << "\n";
    std::cout << "  Weight:     [" << O << ", " << I << "] (PyTorch format)\n\n";
    
    // Get input data
    std::vector<float> inputData = json["input"].parseNDArray();
    
    // Run inference
    std::cout << "Running FullyConnected on GPU...\n";
    Tensor result = eval_fc(inputData, json, I, O);
    
    // Download result from GPU to CPU
    Buffer outBuffer = netGlobalDevice.createBuffer({
        .size = O * sizeof(float),
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
    
    for (size_t i = 0; i < O; i++) {
        float diff = std::abs(output[i] - expected[i]);
        maxDiff = std::max(maxDiff, diff);
        avgDiff += diff;
        
        bool match = diff < tolerance;
        if (!match) mismatches++;
        
        // Show first 5, last 5, and all mismatches
        if (!match || i < 5 || i >= O - 5) {
            std::cout << std::setw(5) << i << " | "
                      << std::setw(11) << expected[i] << " | "
                      << std::setw(11) << output[i] << " | "
                      << std::setw(11) << diff << " | "
                      << (match ? "✓" : "✗") << "\n";
        } else if (i == 5) {
            std::cout << "  ... (" << (O - 10) << " more) ...\n";
        }
    }
    
    avgDiff /= O;
    
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Results Summary:\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "Total values:   " << O << "\n";
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

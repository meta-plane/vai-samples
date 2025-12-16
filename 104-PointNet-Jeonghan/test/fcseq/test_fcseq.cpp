/**
 * FCSequence Vulkan Test (SafeTensors-based)
 * Tests FCSequence (chain of FullyConnectedNodes) against PyTorch reference
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include "neuralNet.h"
#include "neuralNodes.h"
#include "vulkanApp.h"
#include "safeTensorsParser.h"
#include "pointnet.hpp"  // For FCSequence template

using namespace vk;
using namespace networks;

/**
 * Test network with FCSequence
 */
class FCSeqTestNet : public NeuralNet {
    FCSequence<3> fcseq;  // 3 FC layers: 128->256->512->64
    
public:
    FCSeqTestNet(Device& device, const uint32_t(&channels)[4])
    : NeuralNet(device, 1, 1)
    , fcseq(channels)
    {
        // Single FCSequence group
        input(0) - fcseq - output(0);
    }
    
    Tensor& operator[](const std::string& name) {
        return fcseq[name];
    }
};

/**
 * Run FCSequence inference
 */
Tensor eval_fcseq(const std::vector<float>& inputData, const SafeTensorsParser& json, 
                  const uint32_t(&channels)[4]) {
    // Create network
    FCSeqTestNet net(netGlobalDevice, channels);
    
    // Load weights for each FC layer
    for (int i = 0; i < 3; ++i) {
        std::string weight_key = "fc" + std::to_string(i) + ".weight";
        std::string bias_key = "fc" + std::to_string(i) + ".bias";
        
        std::vector<float> weight_data = json[weight_key].parseNDArray();
        std::vector<float> bias_data = json[bias_key].parseNDArray();
        
        // Set parameters with explicit shapes
        net[weight_key] = Tensor(channels[i], channels[i+1]).set(weight_data);
        net[bias_key] = Tensor(channels[i+1]).set(bias_data);
    }
    
    // Prepare network
    net.prepare();
    
    // Create input tensor
    Tensor inputTensor = Tensor(channels[0]).set(inputData);
    
    // Run inference
    auto result = net(inputTensor);
    
    return result[0];
}

void test() {
    void loadShaders();
    loadShaders();
    
    // Load reference data
    SafeTensorsParser json = SafeTensorsParser(PROJECT_CURRENT_DIR"/test/fcseq/reference.safetensors");
    
    // Extract channels [128, 256, 512, 64]
    std::vector<float> channels_data = json["channels"].parseNDArray();
    uint32_t channels[4] = {
        static_cast<uint32_t>(channels_data[0]),
        static_cast<uint32_t>(channels_data[1]),
        static_cast<uint32_t>(channels_data[2]),
        static_cast<uint32_t>(channels_data[3])
    };
    
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║    FCSequence Vulkan Compute Test      ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n\n";
    
    std::cout << "Test configuration:\n";
    std::cout << "  Architecture: " << channels[0] << " -> " << channels[1] 
              << " -> " << channels[2] << " -> " << channels[3] << "\n";
    std::cout << "  Layers: 3 FC layers\n\n";
    
    // Get input data
    std::vector<float> inputData = json["input"].parseNDArray();
    
    // Run inference
    std::cout << "Running FCSequence on GPU...\n";
    Tensor result = eval_fcseq(inputData, json, channels);
    
    // Download result from GPU to CPU
    uint32_t output_size = channels[3];
    Buffer outBuffer = netGlobalDevice.createBuffer({
        .size = output_size * sizeof(float),
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
    
    for (size_t i = 0; i < output_size; i++) {
        float diff = std::abs(output[i] - expected[i]);
        maxDiff = std::max(maxDiff, diff);
        avgDiff += diff;
        
        bool match = diff < tolerance;
        if (!match) mismatches++;
        
        // Show first 5, last 5, and all mismatches
        if (!match || i < 5 || i >= output_size - 5) {
            std::cout << std::setw(5) << i << " | "
                      << std::setw(11) << expected[i] << " | "
                      << std::setw(11) << output[i] << " | "
                      << std::setw(11) << diff << " | "
                      << (match ? "✓" : "✗") << "\n";
        } else if (i == 5) {
            std::cout << "  ... (" << (output_size - 10) << " more) ...\n";
        }
    }
    
    avgDiff /= output_size;
    
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Results Summary:\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "Total values:   " << output_size << "\n";
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

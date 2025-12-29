/**
 * PointWiseMLP Vulkan Test (SafeTensors-based)
 * Tests PointWiseMLPNode + BatchNorm1D + ReLU chain against PyTorch reference
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
 * Test network with PointWiseMLP (includes Conv + BatchNorm + ReLU internally)
 */
class MLPTestNet : public NeuralNet {
    PointWiseMLPNode mlp;
    
public:
    MLPTestNet(Device& device, uint32_t in_channels, uint32_t out_channels)
    : NeuralNet(device, 1, 1)
    , mlp(in_channels, out_channels)
    {
        // Single node now contains full Conv + BN + ReLU chain
        input(0) - mlp - output(0);
    }
    
    Tensor& operator[](const std::string& name) {
        // All weights go directly to mlp node
        return mlp[name];
    }
};

/**
 * Run PointWiseMLP inference
 */
Tensor eval_mlp(const std::vector<float>& inputData, const SafeTensorsParser& weights, 
                uint32_t N, uint32_t C_in, uint32_t C_out) {
    // Create network
    MLPTestNet net(netGlobalDevice, C_in, C_out);
    
    // Set Conv parameters (SafeTensors - direct tensor loading)
    net["weight"] = Tensor(weights["conv_weight"]);
    net["bias"] = Tensor(weights["conv_bias"]);
    
    // Set BatchNorm parameters
    net["bn_mean"] = Tensor(weights["bn_mean"]);
    net["bn_var"] = Tensor(weights["bn_var"]);
    net["bn_gamma"] = Tensor(weights["bn_gamma"]);
    net["bn_beta"] = Tensor(weights["bn_beta"]);
    
    // Prepare network
    net.prepare();
    
    // Create input tensor [C_in, N] (PyTorch convention)
    Tensor inputTensor = Tensor(C_in, N).set(inputData);
    
    // Run inference
    auto result = net(inputTensor);
    
    return result[0];
}

void test() {
    void loadShaders();
    loadShaders();
    
    // Load reference data (SafeTensors format - preferred)
    SafeTensorsParser data(PROJECT_CURRENT_DIR"/test/mlp/reference.safetensors");
    
    // Extract shape from tensor - [C_in, N, C_out] order (PyTorch convention)
    std::vector<float> shape_data = data["shape"].parseNDArray();
    uint32_t C_in = static_cast<uint32_t>(shape_data[0]);
    uint32_t N = static_cast<uint32_t>(shape_data[1]);
    uint32_t C_out = static_cast<uint32_t>(shape_data[2]);
    
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║   PointWiseMLP Vulkan Compute Test      ║\n";
    std::cout << "║   (SafeTensors format)                   ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n\n";
    
    std::cout << "Test configuration:\n";
    std::cout << "  Shape: [C_in=" << C_in << ", N=" << N << ", C_out=" << C_out << "]\n";
    std::cout << "  Input size:  " << (C_in * N) << " values\n";
    std::cout << "  Output size: " << (C_out * N) << " values\n\n";
    
    // Get input data
    std::vector<float> inputData = data["input"].parseNDArray();
    
    // Run inference
    std::cout << "Running PointWiseMLP on GPU...\n";
    Tensor result = eval_mlp(inputData, data, N, C_in, C_out);
    
    // Download result from GPU to CPU
    Buffer outBuffer = netGlobalDevice.createBuffer({
        .size = C_out * N * sizeof(float),  // [C_out, N] layout
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
    std::vector<float> expected = data["expected"].parseNDArray();
    
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
    
    for (size_t i = 0; i < C_out * N; i++) {
        float diff = std::abs(output[i] - expected[i]);
        maxDiff = std::max(maxDiff, diff);
        avgDiff += diff;
        
        bool match = diff < tolerance;
        if (!match) mismatches++;
        
        // Show first 5, last 5, and all mismatches
        if (!match || i < 5 || i >= C_out * N - 5) {
            std::cout << std::setw(5) << i << " | "
                      << std::setw(11) << expected[i] << " | "
                      << std::setw(11) << output[i] << " | "
                      << std::setw(11) << diff << " | "
                      << (match ? "✓" : "✗") << "\n";
        } else if (i == 5) {
            std::cout << "  ... (" << (C_out * N - 10) << " more) ...\n";
        }
    }
    
    avgDiff /= (C_out * N);
    
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Results Summary:\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "Total values:   " << (C_out * N) << "\n";
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

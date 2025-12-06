/**
 * TNetBlock Vulkan Test (JSON-based)
 * Tests TNetBlock (Spatial Transformer) against PyTorch reference
 * 
 * TNetBlock architecture:
 * - Input: [N, K] point cloud
 * - Path A: MLP(K→64→128→1024) → MaxPool → FC(1024→512→256→K²) → Reshape([K,K])
 * - Path B: MatMul(input @ transform) → [N, K]
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include "neuralNet.h"
#include "vulkanApp.h"
#include "jsonParser.h"
#include "../networks/include/pointnet.hpp"

using namespace vk;
using namespace networks;

/**
 * Test network wrapper for TNetBlock
 */
class TNetTestNet : public NeuralNet {
    TNetBlock tnet;
    
public:
    TNetTestNet(Device& device, uint32_t K)
    : NeuralNet(device, 1, 1)
    , tnet(K)
    {
        // Connect: input → tnet → output
        input(0) - tnet / "in0";
        tnet - output(0);
    }
    
    Tensor& operator[](const std::string& name) {
        return tnet[name];
    }
};

/**
 * Run TNetBlock inference
 */
Tensor eval_tnet(uint32_t N, uint32_t K,
                 const std::vector<float>& input_data,
                 JsonParser& json) {
    // Create network
    TNetTestNet net(netGlobalDevice, K);
    
    // Load MLP weights
    for (int i = 0; i < 3; i++) {
        std::string prefix = "mlp.mlp" + std::to_string(i) + ".conv.";
        
        std::vector<float> weight_data = json["weights"][prefix + "weight"].parseNDArray();
        std::vector<float> bias_data = json["weights"][prefix + "bias"].parseNDArray();
        
        // Determine dimensions from weight data
        uint32_t out_channels = bias_data.size();
        uint32_t in_channels = weight_data.size() / out_channels;
        
        net[prefix + "weight"] = Tensor(in_channels, out_channels).set(weight_data);
        net[prefix + "bias"] = Tensor(out_channels).set(bias_data);
    }
    
    // Load FC weights
    for (int i = 0; i < 3; i++) {
        std::string prefix = "fc.fc" + std::to_string(i) + ".";
        
        std::vector<float> weight_data = json["weights"][prefix + "weight"].parseNDArray();
        std::vector<float> bias_data = json["weights"][prefix + "bias"].parseNDArray();
        
        uint32_t out_dim = bias_data.size();
        uint32_t in_dim = weight_data.size() / out_dim;
        
        net[prefix + "weight"] = Tensor(in_dim, out_dim).set(weight_data);
        net[prefix + "bias"] = Tensor(out_dim).set(bias_data);
    }
    
    // Create input tensor [N, K]
    Tensor input_tensor = Tensor(N, K).set(input_data);
    
    // Run inference
    auto result = net(input_tensor);
    
    return result[0];
}

void test() {
    void loadShaders();
    loadShaders();
    
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║      TNetBlock Vulkan Compute Test      ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n";
    std::cout << std::endl;
    
    // Load reference data
    JsonParser json(PROJECT_CURRENT_DIR"/test/tnet/reference.json");
    
    // Parse configuration
    std::vector<float> shape = json["shape"].parseNDArray();
    uint32_t N = static_cast<uint32_t>(shape[0]);
    uint32_t K = static_cast<uint32_t>(shape[1]);
    
    std::cout << "Test configuration:\n";
    std::cout << "  Input: [" << N << ", " << K << "]\n";
    std::cout << "  Output: [" << N << ", " << K << "]\n";
    std::cout << "  Transform: [" << K << ", " << K << "]\n";
    std::cout << std::endl;
    
    // Parse input and expected output
    std::vector<float> input_data = json["input"].parseNDArray();
    std::vector<float> expected = json["output"].parseNDArray();
    
    std::cout << "Running TNetBlock on GPU...\n";
    
    // Run inference
    Tensor result = eval_tnet(N, K, input_data, json);
    
    // Download result from GPU
    Buffer outBuf = netGlobalDevice.createBuffer({
        .size = N * K * sizeof(float),
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    });
    
    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuf, result.buffer())
        .end()
        .submit()
        .wait();
    
    float* output = (float*)outBuf.map();
    
    // Compare results
    std::cout << "\n============================================================\n";
    std::cout << "Comparing results...\n";
    std::cout << "============================================================\n\n";
    
    std::cout << "Index | Expected    | Got         | Diff        | Status\n";
    std::cout << "------------------------------------------------------------\n";
    
    const float tolerance = 1e-4f;
    uint32_t mismatches = 0;
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    
    // Show first 5 and last 5 values
    for (uint32_t i = 0; i < N * K; ++i) {
        float diff = std::abs(output[i] - expected[i]);
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
        
        if (diff > tolerance)
            mismatches++;
        
        if (i < 5 || i >= N * K - 5) {
            std::cout << std::setw(5) << i << " | ";
            std::cout << std::setw(11) << std::fixed << std::setprecision(6) << expected[i] << " | ";
            std::cout << std::setw(11) << std::fixed << std::setprecision(6) << output[i] << " | ";
            std::cout << std::setw(11) << std::fixed << std::setprecision(6) << diff << " | ";
            std::cout << (diff <= tolerance ? "✓" : "✗") << "\n";
        } else if (i == 5) {
            std::cout << "  ... (" << (N * K - 10) << " more) ...\n";
        }
    }
    
    std::cout << "\n============================================================\n";
    std::cout << "Results Summary:\n";
    std::cout << "============================================================\n";
    std::cout << "Total values:   " << N * K << "\n";
    std::cout << "Mismatches:     " << mismatches << "\n";
    std::cout << "Max difference: " << std::fixed << std::setprecision(6) << max_diff << "\n";
    std::cout << "Avg difference: " << std::fixed << std::setprecision(6) << (sum_diff / (N * K)) << "\n";
    std::cout << "Tolerance:      " << std::fixed << std::setprecision(6) << tolerance << "\n\n";
    
    if (mismatches == 0) {
        std::cout << "✓ TEST PASSED! All values match within tolerance.\n";
    } else {
        std::cout << "✗ TEST FAILED! " << mismatches << " values exceed tolerance.\n";
        throw std::runtime_error("TNetBlock test failed");
    }
}

int main() {
    try {
        test();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

/**
 * FCBNNode Vulkan Test (SafeTensors-based)
 * Tests FullyConnected + BatchNorm + ReLU node against PyTorch reference
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include "neuralNet.h"
#include "vulkanApp.h"
#include "safeTensorsParser.h"
#include "../networks/include/pointnet.hpp"

using namespace vk;
using namespace networks;

/**
 * Test network wrapper for FCBNNode
 */
class FCBNTestNet : public NeuralNet {
    FCBNNode fcbn;
    
public:
    FCBNTestNet(Device& device, uint32_t inDim, uint32_t outDim)
    : NeuralNet(device, 1, 1)
    , fcbn(inDim, outDim)
    {
        input(0) - fcbn - output(0);
    }
    
    Tensor& operator[](const std::string& name) {
        return fcbn[name];
    }
};

/**
 * Run FCBNNode inference
 */
Tensor eval_fcbn(uint32_t inDim, uint32_t outDim,
                 const std::vector<float>& input_data,
                 SafeTensorsParser& json) {
    // Create network
    FCBNTestNet net(netGlobalDevice, inDim, outDim);
    
    // Load weights
    std::vector<float> weight_data = json["weight"].parseNDArray();
    std::vector<float> bias_data = json["bias"].parseNDArray();
    std::vector<float> bn_mean = json["mean"].parseNDArray();
    std::vector<float> bn_var = json["var"].parseNDArray();
    std::vector<float> bn_gamma = json["gamma"].parseNDArray();
    std::vector<float> bn_beta = json["beta"].parseNDArray();
    
    net["weight"] = Tensor(inDim, outDim).set(weight_data);
    net["bias"] = Tensor(outDim).set(bias_data);
    net["mean"] = Tensor(outDim).set(bn_mean);
    net["var"] = Tensor(outDim).set(bn_var);
    net["gamma"] = Tensor(outDim).set(bn_gamma);
    net["beta"] = Tensor(outDim).set(bn_beta);
    
    // Create input tensor [inDim] (FCBNNode handles 1D input)
    Tensor input_tensor = Tensor(inDim).set(input_data);
    
    // Run inference
    auto result = net(input_tensor);
    
    return result[0];
}

void test() {
    void loadShaders();
    loadShaders();
    
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║    FCBNNode Vulkan Compute Test         ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n";
    std::cout << std::endl;
    
    // Load reference data
    SafeTensorsParser json(PROJECT_CURRENT_DIR"/test/fcbn/reference.safetensors");
    
    // Parse configuration
    std::vector<float> shape = json["shape"].parseNDArray();
    uint32_t inDim = static_cast<uint32_t>(shape[0]);
    uint32_t outDim = static_cast<uint32_t>(shape[1]);
    
    std::cout << "Test configuration:\n";
    std::cout << "  Input dim:  " << inDim << "\n";
    std::cout << "  Output dim: " << outDim << "\n";
    std::cout << std::endl;
    
    // Parse input and expected output
    std::vector<float> input_data = json["input"].parseNDArray();
    std::vector<float> expected = json["expected"].parseNDArray();
    
    std::cout << "Running FCBNNode on GPU...\n";
    
    // Run inference
    Tensor result = eval_fcbn(inDim, outDim, input_data, json);
    
    // Download result from GPU
    Buffer outBuf = netGlobalDevice.createBuffer({
        .size = outDim * sizeof(float),
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
    for (uint32_t i = 0; i < outDim; ++i) {
        float diff = std::abs(output[i] - expected[i]);
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
        
        if (diff > tolerance)
            mismatches++;
        
        if (i < 5 || i >= outDim - 5) {
            std::cout << std::setw(5) << i << " | ";
            std::cout << std::setw(11) << std::fixed << std::setprecision(6) << expected[i] << " | ";
            std::cout << std::setw(11) << std::fixed << std::setprecision(6) << output[i] << " | ";
            std::cout << std::setw(11) << std::fixed << std::setprecision(6) << diff << " | ";
            std::cout << (diff <= tolerance ? "✓" : "✗") << "\n";
        } else if (i == 5) {
            std::cout << "  ... (" << (outDim - 10) << " more) ...\n";
        }
    }
    
    std::cout << "\n============================================================\n";
    std::cout << "Results Summary:\n";
    std::cout << "============================================================\n";
    std::cout << "Total values:   " << outDim << "\n";
    std::cout << "Mismatches:     " << mismatches << "\n";
    std::cout << "Max difference: " << std::fixed << std::setprecision(6) << max_diff << "\n";
    std::cout << "Avg difference: " << std::fixed << std::setprecision(6) << (sum_diff / outDim) << "\n";
    std::cout << "Tolerance:      " << std::fixed << std::setprecision(6) << tolerance << "\n\n";
    
    if (mismatches == 0) {
        std::cout << "✓ TEST PASSED! All values match within tolerance.\n";
    } else {
        std::cout << "✗ TEST FAILED! " << mismatches << " values exceed tolerance.\n";
        throw std::runtime_error("FCBNNode test failed");
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

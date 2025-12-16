/**
 * FCBNSequence Vulkan Test (SafeTensors-based)
 * Tests sequence of FC+BN+ReLU blocks against PyTorch reference
 * Last block is FC only (no BN+ReLU)
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
 * Test network wrapper for FCBNSequence
 */
class FCBNSeqTestNet : public NeuralNet {
    FCBNSequence<3> fcbn_seq;  // 3 blocks: [512->256->128->9]
    
public:
    FCBNSeqTestNet(Device& device, const uint32_t (&dims)[4])
    : NeuralNet(device, 1, 1)
    , fcbn_seq(dims)
    {
        input(0) - fcbn_seq - output(0);
    }
    
    Tensor& operator[](const std::string& name) {
        return fcbn_seq[name];
    }
};

/**
 * Run FCBNSequence inference
 */
Tensor eval_fcbn_seq(const uint32_t (&dims)[4],
                     const std::vector<float>& input_data,
                     SafeTensorsParser& json) {
    // Create network
    FCBNSeqTestNet net(netGlobalDevice, dims);
    
    // Load weights for each block
    // Block 0 (FC+BN+ReLU)
    std::vector<float> block0_weight = json["block0.weight"].parseNDArray();
    std::vector<float> block0_bias = json["block0.bias"].parseNDArray();
    std::vector<float> block0_mean = json["block0.mean"].parseNDArray();
    std::vector<float> block0_var = json["block0.var"].parseNDArray();
    std::vector<float> block0_gamma = json["block0.gamma"].parseNDArray();
    std::vector<float> block0_beta = json["block0.beta"].parseNDArray();
    
    net["block0.weight"] = Tensor(dims[0], dims[1]).set(block0_weight);
    net["block0.bias"] = Tensor(dims[1]).set(block0_bias);
    net["block0.mean"] = Tensor(dims[1]).set(block0_mean);
    net["block0.var"] = Tensor(dims[1]).set(block0_var);
    net["block0.gamma"] = Tensor(dims[1]).set(block0_gamma);
    net["block0.beta"] = Tensor(dims[1]).set(block0_beta);
    
    // Block 1 (FC+BN+ReLU)
    std::vector<float> block1_weight = json["block1.weight"].parseNDArray();
    std::vector<float> block1_bias = json["block1.bias"].parseNDArray();
    std::vector<float> block1_mean = json["block1.mean"].parseNDArray();
    std::vector<float> block1_var = json["block1.var"].parseNDArray();
    std::vector<float> block1_gamma = json["block1.gamma"].parseNDArray();
    std::vector<float> block1_beta = json["block1.beta"].parseNDArray();
    
    net["block1.weight"] = Tensor(dims[1], dims[2]).set(block1_weight);
    net["block1.bias"] = Tensor(dims[2]).set(block1_bias);
    net["block1.mean"] = Tensor(dims[2]).set(block1_mean);
    net["block1.var"] = Tensor(dims[2]).set(block1_var);
    net["block1.gamma"] = Tensor(dims[2]).set(block1_gamma);
    net["block1.beta"] = Tensor(dims[2]).set(block1_beta);
    
    // Block 2 (FC only)
    std::vector<float> lastBlock_weight = json["lastBlock.weight"].parseNDArray();
    std::vector<float> lastBlock_bias = json["lastBlock.bias"].parseNDArray();
    
    net["lastBlock.weight"] = Tensor(dims[2], dims[3]).set(lastBlock_weight);
    net["lastBlock.bias"] = Tensor(dims[3]).set(lastBlock_bias);
    
    // Create input tensor [dims[0]] - FCBNNode expects 1D input
    Tensor input_tensor = Tensor(dims[0]).set(input_data);
    
    // Run inference
    auto result = net(input_tensor);
    
    return result[0];
}

void test() {
    void loadShaders();
    loadShaders();
    
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║   FCBNSequence Vulkan Compute Test      ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n";
    std::cout << std::endl;
    
    // Load reference data
    SafeTensorsParser json(PROJECT_CURRENT_DIR"/test/fcbn_seq/reference.safetensors");
    
    // Parse dimensions
    std::vector<float> shape_vec = json["shape"].parseNDArray();
    uint32_t dims[4] = {
        static_cast<uint32_t>(shape_vec[0]),
        static_cast<uint32_t>(shape_vec[1]),
        static_cast<uint32_t>(shape_vec[2]),
        static_cast<uint32_t>(shape_vec[3])
    };
    
    std::cout << "Test configuration:\n";
    std::cout << "  Architecture: " << dims[0] << " -> " << dims[1] 
              << " -> " << dims[2] << " -> " << dims[3] << "\n";
    std::cout << "  Block 0: FC+BN+ReLU (" << dims[0] << " -> " << dims[1] << ")\n";
    std::cout << "  Block 1: FC+BN+ReLU (" << dims[1] << " -> " << dims[2] << ")\n";
    std::cout << "  Block 2: FC only    (" << dims[2] << " -> " << dims[3] << ")\n";
    std::cout << std::endl;
    
    // Parse input and expected output
    std::vector<float> input = json["input"].parseNDArray();
    std::vector<float> expected = json["expected"].parseNDArray();
    
    std::cout << "Running FCBNSequence on GPU...\n";
    
    // Run inference
    Tensor result = eval_fcbn_seq(dims, input, json);
    
    // Copy result to CPU
    Buffer outputBuffer = netGlobalDevice.createBuffer({
        .size = static_cast<uint32_t>(expected.size() * sizeof(float)),
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    });
    
    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outputBuffer, result.buffer())
        .end()
        .submit()
        .wait();
    
    float* output = (float*)outputBuffer.map();
    
    // Compare results
    std::cout << "\n============================================================\n";
    std::cout << "Comparing results...\n";
    std::cout << "============================================================\n\n";
    
    constexpr float tolerance = 1e-4f;
    uint32_t mismatches = 0;
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Index | Expected    | Got         | Diff        | Status\n";
    std::cout << "------------------------------------------------------------\n";
    
    // Show all values (only 9 outputs)
    for (size_t i = 0; i < expected.size(); i++) {
        float diff = std::abs(output[i] - expected[i]);
        bool match = diff < tolerance;
        
        std::cout << std::setw(5) << i << " | "
                  << std::setw(11) << expected[i] << " | "
                  << std::setw(11) << output[i] << " | "
                  << std::setw(11) << diff << " | "
                  << (match ? "✓" : "✗") << "\n";
        
        if (!match) mismatches++;
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
    }
    
    std::cout << "\n============================================================\n";
    std::cout << "Results Summary:\n";
    std::cout << "============================================================\n";
    std::cout << "Total values:   " << expected.size() << "\n";
    std::cout << "Mismatches:     " << mismatches << "\n";
    std::cout << "Max difference: " << max_diff << "\n";
    std::cout << "Avg difference: " << (sum_diff / expected.size()) << "\n";
    std::cout << "Tolerance:      " << tolerance << "\n\n";
    
    if (mismatches == 0) {
        std::cout << "✓ TEST PASSED! All values match within tolerance.\n";
    } else {
        std::cout << "✗ TEST FAILED! " << mismatches << " value(s) outside tolerance.\n";
        exit(1);
    }
    
    outputBuffer.unmap();
}

int main() {
    test();
    return 0;
}

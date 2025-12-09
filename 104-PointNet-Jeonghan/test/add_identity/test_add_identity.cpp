/**
 * AddIdentityNode test
 * Tests adding identity matrix to a square matrix
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include "neuralNet.h"
#include "neuralNodes.h"
#include "vulkanApp.h"
#include "jsonParser.h"

using namespace vk;

class AddIdentityTestNet : public NeuralNet
{
    AddIdentityNode addIdentity;

public:
    AddIdentityTestNet(Device& device)
    : NeuralNet(device, 1, 1)
    , addIdentity()
    {
        input(0) - addIdentity - output(0);
    }
};

void test() {
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║      AddIdentity Vulkan Test            ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n\n";
    
    JsonParser json(PROJECT_CURRENT_DIR"/test/add_identity/reference.json");
    
    std::vector<float> k_vec = json["K"].parseNDArray();
    uint32_t K = static_cast<uint32_t>(k_vec[0]);
    
    std::cout << "Config: Input=[" << K << ", " << K << "], Output=[" << K << ", " << K << "]\n\n";
    
    // Load data
    std::vector<float> input_data = json["input"].parseNDArray();
    std::vector<float> expected = json["output"].parseNDArray();
    
    // Run inference
    AddIdentityTestNet net(netGlobalDevice);
    Tensor input_tensor = Tensor(K, K).set(input_data);
    net.prepare();
    auto result = net(input_tensor);
    
    // Download result
    Buffer outBuf = netGlobalDevice.createBuffer({
        .size = K * K * sizeof(float),
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    });
    
    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuf, result[0].buffer())
        .end()
        .submit()
        .wait();
    
    float* output = (float*)outBuf.map();
    
    // Compare
    std::cout << "Input matrix:\n";
    for (uint32_t i = 0; i < K; i++) {
        std::cout << "  [";
        for (uint32_t j = 0; j < K; j++) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(6) << input_data[i * K + j];
            if (j < K - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    
    std::cout << "\nExpected output:\n";
    for (uint32_t i = 0; i < K; i++) {
        std::cout << "  [";
        for (uint32_t j = 0; j < K; j++) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(6) << expected[i * K + j];
            if (j < K - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    
    std::cout << "\nGot output:\n";
    for (uint32_t i = 0; i < K; i++) {
        std::cout << "  [";
        for (uint32_t j = 0; j < K; j++) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(6) << output[i * K + j];
            if (j < K - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    
    // Check differences
    const float tolerance = 1e-6f;
    uint32_t mismatches = 0;
    float max_diff = 0.0f;
    
    std::cout << "\n============================================================\n";
    std::cout << "Element-wise comparison:\n";
    std::cout << "============================================================\n";
    
    for (uint32_t i = 0; i < K; i++) {
        for (uint32_t j = 0; j < K; j++) {
            uint32_t idx = i * K + j;
            float diff = std::abs(output[idx] - expected[idx]);
            max_diff = std::max(max_diff, diff);
            if (diff > tolerance) mismatches++;
            
            std::cout << "[" << i << "," << j << "] Exp: " << std::setw(10) << expected[idx]
                     << " Got: " << std::setw(10) << output[idx]
                     << " Diff: " << std::setw(10) << diff
                     << " " << (diff <= tolerance ? "✓" : "✗") << "\n";
        }
    }
    
    std::cout << "\n============================================================\n";
    std::cout << "Summary:\n";
    std::cout << "============================================================\n";
    std::cout << "Total values:   " << (K * K) << "\n";
    std::cout << "Mismatches:     " << mismatches << "\n";
    std::cout << "Max difference: " << std::fixed << std::setprecision(9) << max_diff << "\n";
    std::cout << "Tolerance:      " << tolerance << "\n\n";
    
    if (mismatches == 0) {
        std::cout << "✓ TEST PASSED!\n";
    } else {
        std::cout << "✗ TEST FAILED!\n";
        throw std::runtime_error("AddIdentity test failed");
    }
}

int main() {
    try {
        test();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

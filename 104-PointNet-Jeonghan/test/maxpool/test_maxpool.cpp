/**
 * MaxPooling1D Vulkan Test
 * Tests MaxPooling1DNode against PyTorch reference
 * 
 * MaxPooling1D: [N, C] -> [C] (max along dimension 0)
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
 * Test network wrapper for MaxPooling1DNode
 */
class MaxPoolTestNet : public NeuralNet
{
    MaxPooling1DNode maxpool;

public:
    MaxPoolTestNet(Device& device)
    : NeuralNet(device, 1, 1)
    , maxpool()
    {
        input(0) - maxpool - output(0);
    }
};

/**
 * Run MaxPooling1D inference
 */
Tensor eval_maxpool(uint32_t N, uint32_t C,
                    const std::vector<float>& input_data) {
    // Create network
    MaxPoolTestNet net(netGlobalDevice);
    
    // Create input tensor [C, N]
    Tensor input_tensor = Tensor(C, N).set(input_data);
    
    // Prepare network
    net.prepare();
    
    // Run inference
    auto result = net(input_tensor);
    
    return result[0];
}

void test() {
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║    MaxPooling1D Vulkan Compute Test     ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n";
    std::cout << std::endl;
    
    // Load reference data
    SafeTensorsParser json(PROJECT_CURRENT_DIR"/test/maxpool/reference.safetensors");
    
    // Parse configuration [C, N]
    std::vector<float> shape = json["shape"].parseNDArray();
    uint32_t C = static_cast<uint32_t>(shape[0]);
    uint32_t N = static_cast<uint32_t>(shape[1]);
    
    std::cout << "Test configuration:\n";
    std::cout << "  Input: [C=" << C << ", N=" << N << "]\n";
    std::cout << "  Output: [" << C << "]\n";
    std::cout << std::endl;
    
    // Parse input and expected output
    std::vector<float> input_data = json["input"].parseNDArray();
    std::vector<float> expected = json["output"].parseNDArray();
    
    std::cout << "Running MaxPooling1D on GPU...\n";
    
    // Run inference
    Tensor result = eval_maxpool(N, C, input_data);
    
    // Download result from GPU
    Buffer outBuf = netGlobalDevice.createBuffer({
        .size = C * sizeof(float),
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
    
    const float tolerance = 1e-6f;  // MaxPool should be exact (just comparison)
    uint32_t mismatches = 0;
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    
    // Show first 10, middle 5, and last 5 values
    for (uint32_t i = 0; i < C; ++i) {
        float diff = std::abs(output[i] - expected[i]);
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
        
        if (diff > tolerance)
            mismatches++;
        
        // Print: first 10, middle 5, last 5
        bool should_print = (i < 10) || 
                           (i >= C/2 - 2 && i <= C/2 + 2) || 
                           (i >= C - 5);
        
        if (should_print) {
            std::cout << std::setw(5) << i << " | "
                     << std::setw(11) << std::fixed << std::setprecision(6) << expected[i] << " | "
                     << std::setw(11) << std::fixed << std::setprecision(6) << output[i] << " | "
                     << std::setw(11) << std::fixed << std::setprecision(6) << diff << " | "
                     << (diff <= tolerance ? "✓" : "✗") << "\n";
        } else if (i == 10 || i == C/2 - 3 || i == C - 6) {
            std::cout << "  ... (" << (i == 10 ? C/2 - 12 : (i == C/2 - 3 ? 5 : C - i - 6)) << " more) ...\n";
        }
    }
    
    std::cout << "\n============================================================\n";
    std::cout << "Results Summary:\n";
    std::cout << "============================================================\n";
    std::cout << "Total values:   " << C << "\n";
    std::cout << "Mismatches:     " << mismatches << "\n";
    std::cout << "Max difference: " << std::fixed << std::setprecision(6) << max_diff << "\n";
    std::cout << "Avg difference: " << std::fixed << std::setprecision(6) << (sum_diff / C) << "\n";
    std::cout << "Tolerance:      " << std::fixed << std::setprecision(6) << tolerance << "\n\n";
    
    if (mismatches == 0) {
        std::cout << "✓ TEST PASSED! All values match within tolerance.\n";
    } else {
        std::cout << "✗ TEST FAILED! " << mismatches << " values exceed tolerance.\n";
        throw std::runtime_error("MaxPooling1D test failed");
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

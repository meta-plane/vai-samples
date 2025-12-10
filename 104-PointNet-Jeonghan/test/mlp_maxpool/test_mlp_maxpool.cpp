/**
 * MLP + MaxPool test
 * Tests the first part of TNet: MLP(3→64→128→1024) → MaxPool
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include "neuralNet.h"
#include "neuralNodes.h"
#include "vulkanApp.h"
#include "jsonParser.h"
#include "../networks/include/pointnet.hpp"

using namespace vk;
using namespace networks;

class MLPMaxPoolNet : public NeuralNet
{
    MLPSequence<3> mlp;
    MaxPooling1DNode maxpool;

public:
    MLPMaxPoolNet(Device& device)
    : NeuralNet(device, 1, 1)
    , mlp({3, 64, 128, 1024})
    , maxpool()
    {
        input(0) - mlp - maxpool - output(0);
    }

    Tensor& operator[](const std::string& name) {
        return mlp[name];
    }
};

void test() {
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║      MLP + MaxPool Vulkan Test          ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n\n";
    
    JsonParser json(PROJECT_CURRENT_DIR"/test/mlp_maxpool/reference.json");
    
    std::vector<float> shape = json["shape"].parseNDArray();
    uint32_t N = static_cast<uint32_t>(shape[0]);
    uint32_t K = static_cast<uint32_t>(shape[1]);
    
    std::cout << "Config: Input=[" << N << ", " << K << "], Output=[1024]\n\n";
    
    // Load weights
    MLPMaxPoolNet net(netGlobalDevice);
    
    std::cout << "Loading MLP weights...\n";
    for (int i = 0; i < 3; i++) {
        std::string prefix = "mlp.mlp" + std::to_string(i) + ".";
        
        std::vector<float> weight_data = json[prefix + "weight"].parseNDArray();
        std::vector<float> bias_data = json[prefix + "bias"].parseNDArray();
        std::vector<float> mean_data = json[prefix + "mean"].parseNDArray();
        std::vector<float> var_data = json[prefix + "var"].parseNDArray();
        std::vector<float> gamma_data = json[prefix + "gamma"].parseNDArray();
        std::vector<float> beta_data = json[prefix + "beta"].parseNDArray();

        std::string net_key = "mlp" + std::to_string(i) + ".";
        net[net_key + "weight"] = Tensor(weight_data.size() / bias_data.size(), bias_data.size()).set(weight_data);
        net[net_key + "bias"] = Tensor(bias_data.size()).set(bias_data);
        net[net_key + "bn_mean"] = Tensor(bias_data.size()).set(mean_data);
        net[net_key + "bn_var"] = Tensor(bias_data.size()).set(var_data);
        net[net_key + "bn_gamma"] = Tensor(bias_data.size()).set(gamma_data);
        net[net_key + "bn_beta"] = Tensor(bias_data.size()).set(beta_data);
        
        std::cout << "  ✓ mlp" << i << " loaded\n";
    }
    
    // Run inference
    std::vector<float> input_data = json["input"].parseNDArray();
    std::vector<float> expected = json["output"].parseNDArray();
    
    Tensor input_tensor = Tensor(N, K).set(input_data);
    net.prepare();
    auto result = net(input_tensor);
    
    // Download result
    Buffer outBuf = netGlobalDevice.createBuffer({
        .size = 1024 * sizeof(float),
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
    std::cout << "\n============================================================\n";
    std::cout << "Comparing results...\n";
    std::cout << "============================================================\n\n";
    
    const float tolerance = 1e-4f;
    uint32_t mismatches = 0;
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    
    for (uint32_t i = 0; i < 1024; ++i) {
        float diff = std::abs(output[i] - expected[i]);
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
        if (diff > tolerance) mismatches++;
        
        if (i < 10 || (i >= 512 - 2 && i <= 512 + 2) || i >= 1024 - 5) {
            std::cout << std::setw(5) << i << " | "
                     << std::setw(11) << std::fixed << std::setprecision(6) << expected[i] << " | "
                     << std::setw(11) << output[i] << " | "
                     << std::setw(11) << diff << " | "
                     << (diff <= tolerance ? "✓" : "✗") << "\n";
        } else if (i == 10 || i == 512 - 3 || i == 1024 - 6) {
            std::cout << "  ... (" << (i == 10 ? 502 : (i == 512 - 3 ? 5 : 1)) << " more) ...\n";
        }
    }
    
    std::cout << "\n============================================================\n";
    std::cout << "Results Summary:\n";
    std::cout << "============================================================\n";
    std::cout << "Total values:   1024\n";
    std::cout << "Mismatches:     " << mismatches << "\n";
    std::cout << "Max difference: " << std::fixed << std::setprecision(6) << max_diff << "\n";
    std::cout << "Avg difference: " << (sum_diff / 1024) << "\n";
    std::cout << "Tolerance:      " << tolerance << "\n\n";
    
    if (mismatches == 0) {
        std::cout << "✓ TEST PASSED!\n";
    } else {
        std::cout << "✗ TEST FAILED! " << mismatches << " mismatches.\n";
        throw std::runtime_error("MLP+MaxPool test failed");
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

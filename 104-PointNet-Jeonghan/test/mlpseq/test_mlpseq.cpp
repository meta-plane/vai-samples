/**
 * MLPSequence Vulkan Test (JSON-based)
 * Tests MLPSequence (chain of PointWiseMLPNodes) against PyTorch reference
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include "neuralNet.h"
#include "neuralNodes.h"
#include "vulkanApp.h"
#include "jsonParser.h"
#include "pointnet.hpp"

using namespace vk;
using namespace networks;

class MLPSeqTestNet : public NeuralNet {
    MLPSequence<3> mlpseq;
    
public:
    MLPSeqTestNet(Device& device, const uint32_t(&channels)[4])
    : NeuralNet(device, 1, 1)
    , mlpseq(channels)
    {
        input(0) - mlpseq - output(0);
    }
    
    Tensor& operator[](const std::string& name) {
        return mlpseq[name];
    }
};

Tensor eval_mlpseq(uint32_t N, const std::vector<float>& inputData, 
                   const JsonParser& json, const uint32_t(&channels)[4]) {
    MLPSeqTestNet net(netGlobalDevice, channels);
    
    for (int i = 0; i < 3; ++i) {
        std::string prefix = "mlp" + std::to_string(i);
        
        // Load Conv weights - map to PointWiseMLPNode slots (weight, bias, not conv.weight)
        std::vector<float> conv_weight = json[prefix + ".conv.weight"].parseNDArray();
        std::vector<float> conv_bias = json[prefix + ".conv.bias"].parseNDArray();
        
        net[prefix + ".weight"] = Tensor(channels[i], channels[i+1]).set(conv_weight);
        net[prefix + ".bias"] = Tensor(channels[i+1]).set(conv_bias);
        
        // Load BatchNorm parameters
        std::vector<float> bn_mean = json[prefix + ".bn.mean"].parseNDArray();
        std::vector<float> bn_var = json[prefix + ".bn.var"].parseNDArray();
        std::vector<float> bn_gamma = json[prefix + ".bn.gamma"].parseNDArray();
        std::vector<float> bn_beta = json[prefix + ".bn.beta"].parseNDArray();
        
        net[prefix + ".bn_mean"] = Tensor(channels[i+1]).set(bn_mean);
        net[prefix + ".bn_var"] = Tensor(channels[i+1]).set(bn_var);
        net[prefix + ".bn_gamma"] = Tensor(channels[i+1]).set(bn_gamma);
        net[prefix + ".bn_beta"] = Tensor(channels[i+1]).set(bn_beta);
    }
    
    net.prepare();
    Tensor inputTensor = Tensor(N, channels[0]).set(inputData);
    auto result = net(inputTensor);
    return result[0];
}

void test() {
    void loadShaders();
    loadShaders();
    
    JsonParser json = JsonParser(PROJECT_CURRENT_DIR"/test/mlpseq/reference.json");
    
    std::vector<float> shape = json["shape"].parseNDArray();
    uint32_t N = static_cast<uint32_t>(shape[0]);
    
    uint32_t channels[4] = {3, 64, 128, 256};
    
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║   MLPSequence Vulkan Compute Test       ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n\n";
    
    std::cout << "Test configuration:\n";
    std::cout << "  Architecture: 3 -> 64 -> 128 -> 256\n";
    std::cout << "  Layers: 3 MLP blocks (Conv + BatchNorm + ReLU)\n";
    std::cout << "  Points: " << N << "\n\n";
    
    std::vector<float> inputData = json["input"].parseNDArray();
    
    std::cout << "Running MLPSequence on GPU...\n";
    Tensor result = eval_mlpseq(N, inputData, json, channels);
    
    uint32_t C_out = channels[3];
    uint32_t totalElements = N * C_out;
    
    Buffer outBuffer = netGlobalDevice.createBuffer({
        .size = totalElements * sizeof(float),
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
    std::vector<float> expected = json["expected"].parseNDArray();
    
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
    
    for (uint32_t i = 0; i < totalElements; i++) {
        float diff = std::abs(output[i] - expected[i]);
        maxDiff = std::max(maxDiff, diff);
        avgDiff += diff;
        
        bool match = diff < tolerance;
        if (!match) mismatches++;
        
        if (!match || i < 5 || i >= totalElements - 5) {
            std::cout << std::setw(5) << i << " | "
                      << std::setw(11) << expected[i] << " | "
                      << std::setw(11) << output[i] << " | "
                      << std::setw(11) << diff << " | "
                      << (match ? "✓" : "✗") << "\n";
        } else if (i == 5) {
            std::cout << "  ... (" << (totalElements - 10) << " more) ...\n";
        }
    }
    
    avgDiff /= totalElements;
    
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Results Summary:\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "Total values:   " << totalElements << "\n";
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

/**
 * T-Net Transform Path Test (Two-stage)
 * Stage 1: MLP -> MaxPool (compare pooled output)
 * Stage 2: FC -> Reshape -> AddIdentity (compare transform matrix)
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include "neuralNet.h"
#include "vulkanApp.h"
#include "jsonParser.h"
#include "../networks/include/pointnet.hpp"

using namespace vk;
using namespace networks;

/**
 * Stage 1 network: MLP -> MaxPool only
 * Outputs: [1024] (the global feature)
 */
class MLPMaxPoolNet : public NeuralNet {
    MLPSequence<3> mlp;          // K -> 64 -> 128 -> 1024
    MaxPooling1DNode maxpool;    // [N, 1024] -> [1024]
    
public:
    MLPMaxPoolNet(Device& device, uint32_t K)
    : NeuralNet(device, 1, 1)
    , mlp({K, 64, 128, 1024})
    , maxpool()
    {
        NeuralNet::input(0) - mlp - maxpool - NeuralNet::output(0);
    }
    
    Tensor& operator[](const std::string& name) {
        if (name.find("mlp.") == 0)
            return mlp[name.substr(4)];
        throw std::runtime_error("Unknown parameter: " + name);
    }
};

/**
 * Stage 2 network: FC -> Reshape -> AddIdentity only
 * Input: [1024], Output: [K, K]
 */
class FCTransformNet : public NeuralNet {
    FCBNSequence<3> fc;          // 1024 -> 512 -> 256 -> K*K
    ReShapeNode reshape;         // [K*K] -> [K, K]
    AddIdentityNode addIdentity; // [K, K] + I -> [K, K]
    
public:
    FCTransformNet(Device& device, uint32_t K)
    : NeuralNet(device, 1, 1)
    , fc({1024, 512, 256, K*K})
    , reshape({K, K})
    , addIdentity()
    {
        NeuralNet::input(0) - fc - reshape - addIdentity - NeuralNet::output(0);
    }
    
    Tensor& operator[](const std::string& name) {
        if (name.find("fc.") == 0)
            return fc[name.substr(3)];
        throw std::runtime_error("Unknown parameter: " + name);
    }
};

// Download GPU buffer to host
std::vector<float> downloadBuffer(Tensor& tensor, size_t size) {
    Buffer outBuf = netGlobalDevice.createBuffer({
        .size = size * sizeof(float),
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    });
    
    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuf, tensor.buffer())
        .end()
        .submit()
        .wait();
    
    float* data = (float*)outBuf.map();
    return std::vector<float>(data, data + size);
}

// Compare arrays and return (max_diff, avg_diff, mismatches)
std::tuple<float, float, int> compareArrays(const std::vector<float>& got, 
                                            const std::vector<float>& expected,
                                            float tolerance) {
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    int mismatches = 0;
    
    for (size_t i = 0; i < got.size(); ++i) {
        float diff = std::abs(got[i] - expected[i]);
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
        if (diff > tolerance) mismatches++;
    }
    
    return {max_diff, sum_diff / got.size(), mismatches};
}

void test() {
    void loadShaders();
    loadShaders();
    
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║  T-Net Transform Path (Two-Stage) Test  ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n\n";
    
    // Load reference data
    JsonParser json(PROJECT_CURRENT_DIR"/test/tnet_transform_path/reference.json");
    
    std::vector<float> shape = json["shape"].parseNDArray();
    uint32_t N = static_cast<uint32_t>(shape[0]);
    uint32_t K = static_cast<uint32_t>(shape[1]);
    
    std::cout << "Configuration: N=" << N << ", K=" << K << "\n\n";
    
    std::vector<float> input_data = json["input"].parseNDArray();
    const float tolerance = 0.001f;
    bool all_passed = true;
    
    //==========================================================================
    // STAGE 1: MLP -> MaxPool
    //==========================================================================
    std::cout << "═══════════════════════════════════════════\n";
    std::cout << "  STAGE 1: MLP -> MaxPool\n";
    std::cout << "═══════════════════════════════════════════\n\n";
    
    {
        MLPMaxPoolNet net(netGlobalDevice, K);
        
        std::cout << "Loading MLP weights...\n";
        
        // MLP layer 0: K -> 64
        {
            std::vector<float> weight = json["mlp.mlp0.weight"].parseNDArray();
            std::vector<float> bias = json["mlp.mlp0.bias"].parseNDArray();
            std::vector<float> mean = json["mlp.mlp0.mean"].parseNDArray();
            std::vector<float> var = json["mlp.mlp0.var"].parseNDArray();
            std::vector<float> gamma = json["mlp.mlp0.gamma"].parseNDArray();
            std::vector<float> beta = json["mlp.mlp0.beta"].parseNDArray();
            
            net["mlp.mlp0.weight"] = Tensor(K, 64).set(weight);
            net["mlp.mlp0.bias"] = Tensor(64).set(bias);
            net["mlp.mlp0.bn_mean"] = Tensor(64).set(mean);
            net["mlp.mlp0.bn_var"] = Tensor(64).set(var);
            net["mlp.mlp0.bn_gamma"] = Tensor(64).set(gamma);
            net["mlp.mlp0.bn_beta"] = Tensor(64).set(beta);
            std::cout << "  ✓ mlp0\n";
        }
        
        // MLP layer 1: 64 -> 128
        {
            std::vector<float> weight = json["mlp.mlp1.weight"].parseNDArray();
            std::vector<float> bias = json["mlp.mlp1.bias"].parseNDArray();
            std::vector<float> mean = json["mlp.mlp1.mean"].parseNDArray();
            std::vector<float> var = json["mlp.mlp1.var"].parseNDArray();
            std::vector<float> gamma = json["mlp.mlp1.gamma"].parseNDArray();
            std::vector<float> beta = json["mlp.mlp1.beta"].parseNDArray();
            
            net["mlp.mlp1.weight"] = Tensor(64, 128).set(weight);
            net["mlp.mlp1.bias"] = Tensor(128).set(bias);
            net["mlp.mlp1.bn_mean"] = Tensor(128).set(mean);
            net["mlp.mlp1.bn_var"] = Tensor(128).set(var);
            net["mlp.mlp1.bn_gamma"] = Tensor(128).set(gamma);
            net["mlp.mlp1.bn_beta"] = Tensor(128).set(beta);
            std::cout << "  ✓ mlp1\n";
        }
        
        // MLP layer 2: 128 -> 1024
        {
            std::vector<float> weight = json["mlp.mlp2.weight"].parseNDArray();
            std::vector<float> bias = json["mlp.mlp2.bias"].parseNDArray();
            std::vector<float> mean = json["mlp.mlp2.mean"].parseNDArray();
            std::vector<float> var = json["mlp.mlp2.var"].parseNDArray();
            std::vector<float> gamma = json["mlp.mlp2.gamma"].parseNDArray();
            std::vector<float> beta = json["mlp.mlp2.beta"].parseNDArray();
            
            net["mlp.mlp2.weight"] = Tensor(128, 1024).set(weight);
            net["mlp.mlp2.bias"] = Tensor(1024).set(bias);
            net["mlp.mlp2.bn_mean"] = Tensor(1024).set(mean);
            net["mlp.mlp2.bn_var"] = Tensor(1024).set(var);
            net["mlp.mlp2.bn_gamma"] = Tensor(1024).set(gamma);
            net["mlp.mlp2.bn_beta"] = Tensor(1024).set(beta);
            std::cout << "  ✓ mlp2\n";
        }
        
        // Run
        Tensor input_tensor = Tensor(N, K).set(input_data);
        net.prepare();
        auto outputs = net(input_tensor);
        
        // Compare
        std::vector<float> pooled_got = downloadBuffer(outputs[0], 1024);
        std::vector<float> pooled_expected = json["expected_pooled"].parseNDArray();
        
        auto [max_diff, avg_diff, mismatches] = compareArrays(pooled_got, pooled_expected, tolerance);
        
        std::cout << "\nPooled output comparison:\n";
        std::cout << "  Size: 1024\n";
        std::cout << "  Max diff: " << max_diff << "\n";
        std::cout << "  Avg diff: " << avg_diff << "\n";
        std::cout << "  Mismatches: " << mismatches << " / 1024\n";
        
        // Show first few values
        std::cout << "\n  First 10 values:\n";
        std::cout << "    Expected: ";
        for (int i = 0; i < 10; ++i) std::cout << std::fixed << std::setprecision(4) << pooled_expected[i] << " ";
        std::cout << "\n    Got:      ";
        for (int i = 0; i < 10; ++i) std::cout << std::fixed << std::setprecision(4) << pooled_got[i] << " ";
        std::cout << "\n";
        
        if (mismatches == 0) {
            std::cout << "\n✓ STAGE 1 PASSED!\n";
        } else {
            std::cout << "\n✗ STAGE 1 FAILED!\n";
            all_passed = false;
        }
    }
    
    //==========================================================================
    // STAGE 2a: FC only (without Reshape/AddIdentity)
    //==========================================================================
    std::cout << "\n═══════════════════════════════════════════\n";
    std::cout << "  STAGE 2a: FC ONLY (no Reshape/AddIdentity)\n";
    std::cout << "  (Using EXPECTED pooled output as input)\n";
    std::cout << "═══════════════════════════════════════════\n\n";
    
    std::vector<float> fc_only_output;
    {
        // FC-only network (no reshape, no addIdentity)
        class FCOnlyNet : public NeuralNet {
            FCBNSequence<3> fc;
        public:
            FCOnlyNet(Device& device, uint32_t K)
            : NeuralNet(device, 1, 1)
            , fc({1024, 512, 256, K*K})
            {
                NeuralNet::input(0) - fc - NeuralNet::output(0);
            }
            Tensor& operator[](const std::string& name) {
                return fc[name];
            }
        };
        
        FCOnlyNet net(netGlobalDevice, K);
        
        std::cout << "Loading FC weights...\n";
        
        // FC block 0
        {
            std::vector<float> weight = json["fc.block0.weight"].parseNDArray();
            std::vector<float> bias = json["fc.block0.bias"].parseNDArray();
            std::vector<float> mean = json["fc.block0.mean"].parseNDArray();
            std::vector<float> var = json["fc.block0.var"].parseNDArray();
            std::vector<float> gamma = json["fc.block0.gamma"].parseNDArray();
            std::vector<float> beta = json["fc.block0.beta"].parseNDArray();
            
            net["block0.weight"] = Tensor(1024, 512).set(weight);
            net["block0.bias"] = Tensor(512).set(bias);
            net["block0.mean"] = Tensor(512).set(mean);
            net["block0.var"] = Tensor(512).set(var);
            net["block0.gamma"] = Tensor(512).set(gamma);
            net["block0.beta"] = Tensor(512).set(beta);
            std::cout << "  ✓ block0\n";
        }
        
        // FC block 1
        {
            std::vector<float> weight = json["fc.block1.weight"].parseNDArray();
            std::vector<float> bias = json["fc.block1.bias"].parseNDArray();
            std::vector<float> mean = json["fc.block1.mean"].parseNDArray();
            std::vector<float> var = json["fc.block1.var"].parseNDArray();
            std::vector<float> gamma = json["fc.block1.gamma"].parseNDArray();
            std::vector<float> beta = json["fc.block1.beta"].parseNDArray();
            
            net["block1.weight"] = Tensor(512, 256).set(weight);
            net["block1.bias"] = Tensor(256).set(bias);
            net["block1.mean"] = Tensor(256).set(mean);
            net["block1.var"] = Tensor(256).set(var);
            net["block1.gamma"] = Tensor(256).set(gamma);
            net["block1.beta"] = Tensor(256).set(beta);
            std::cout << "  ✓ block1\n";
        }
        
        // FC lastBlock
        {
            std::vector<float> weight = json["fc.lastBlock.weight"].parseNDArray();
            std::vector<float> bias = json["fc.lastBlock.bias"].parseNDArray();
            
            net["lastBlock.weight"] = Tensor(256, K*K).set(weight);
            net["lastBlock.bias"] = Tensor(K*K).set(bias);
            std::cout << "  ✓ lastBlock\n";
        }
        
        // Use EXPECTED pooled output as input
        std::vector<float> pooled_input = json["expected_pooled"].parseNDArray();
        Tensor input_tensor = Tensor(1024).set(pooled_input);
        
        net.prepare();
        auto outputs = net(input_tensor);
        
        fc_only_output = downloadBuffer(outputs[0], K*K);
        std::vector<float> fc_expected = json["expected_fc2_out"].parseNDArray();
        
        auto [max_diff, avg_diff, mismatches] = compareArrays(fc_only_output, fc_expected, tolerance);
        
        std::cout << "\nFC output comparison (before Reshape/AddIdentity):\n";
        std::cout << "  Expected (first 9): ";
        for (int i = 0; i < (int)(K*K); ++i) std::cout << std::fixed << std::setprecision(4) << fc_expected[i] << " ";
        std::cout << "\n  Got:                ";
        for (int i = 0; i < (int)(K*K); ++i) std::cout << std::fixed << std::setprecision(4) << fc_only_output[i] << " ";
        std::cout << "\n";
        
        std::cout << "\n  Max diff: " << max_diff << "\n";
        std::cout << "  Avg diff: " << avg_diff << "\n";
        std::cout << "  Mismatches: " << mismatches << " / " << K*K << "\n";
        
        if (mismatches == 0) {
            std::cout << "\n✓ STAGE 2a PASSED!\n";
        } else {
            std::cout << "\n✗ STAGE 2a FAILED!\n";
            all_passed = false;
        }
    }

    //==========================================================================
    // STAGE 2: FC -> Reshape -> AddIdentity (using expected pooled as input)
    //==========================================================================
    std::cout << "\n═══════════════════════════════════════════\n";
    std::cout << "  STAGE 2: FC -> Reshape -> AddIdentity\n";
    std::cout << "  (Using EXPECTED pooled output as input)\n";
    std::cout << "═══════════════════════════════════════════\n\n";
    
    {
        FCTransformNet net(netGlobalDevice, K);
        
        std::cout << "Loading FC weights...\n";
        
        // FC block 0: 1024 -> 512 (with BN)
        {
            std::vector<float> weight = json["fc.block0.weight"].parseNDArray();
            std::vector<float> bias = json["fc.block0.bias"].parseNDArray();
            std::vector<float> mean = json["fc.block0.mean"].parseNDArray();
            std::vector<float> var = json["fc.block0.var"].parseNDArray();
            std::vector<float> gamma = json["fc.block0.gamma"].parseNDArray();
            std::vector<float> beta = json["fc.block0.beta"].parseNDArray();
            
            net["fc.block0.weight"] = Tensor(1024, 512).set(weight);
            net["fc.block0.bias"] = Tensor(512).set(bias);
            net["fc.block0.mean"] = Tensor(512).set(mean);
            net["fc.block0.var"] = Tensor(512).set(var);
            net["fc.block0.gamma"] = Tensor(512).set(gamma);
            net["fc.block0.beta"] = Tensor(512).set(beta);
            std::cout << "  ✓ fc.block0\n";
        }
        
        // FC block 1: 512 -> 256 (with BN)
        {
            std::vector<float> weight = json["fc.block1.weight"].parseNDArray();
            std::vector<float> bias = json["fc.block1.bias"].parseNDArray();
            std::vector<float> mean = json["fc.block1.mean"].parseNDArray();
            std::vector<float> var = json["fc.block1.var"].parseNDArray();
            std::vector<float> gamma = json["fc.block1.gamma"].parseNDArray();
            std::vector<float> beta = json["fc.block1.beta"].parseNDArray();
            
            net["fc.block1.weight"] = Tensor(512, 256).set(weight);
            net["fc.block1.bias"] = Tensor(256).set(bias);
            net["fc.block1.mean"] = Tensor(256).set(mean);
            net["fc.block1.var"] = Tensor(256).set(var);
            net["fc.block1.gamma"] = Tensor(256).set(gamma);
            net["fc.block1.beta"] = Tensor(256).set(beta);
            std::cout << "  ✓ fc.block1\n";
        }
        
        // FC last block: 256 -> K*K (no BN)
        {
            std::vector<float> weight = json["fc.lastBlock.weight"].parseNDArray();
            std::vector<float> bias = json["fc.lastBlock.bias"].parseNDArray();
            
            net["fc.lastBlock.weight"] = Tensor(256, K*K).set(weight);
            net["fc.lastBlock.bias"] = Tensor(K*K).set(bias);
            std::cout << "  ✓ fc.lastBlock\n";
        }
        
        // Use EXPECTED pooled output as input
        std::vector<float> pooled_input = json["expected_pooled"].parseNDArray();
        Tensor input_tensor = Tensor(1024).set(pooled_input);
        
        net.prepare();
        auto outputs = net(input_tensor);
        
        // Compare
        std::vector<float> transform_got = downloadBuffer(outputs[0], K*K);
        std::vector<float> transform_expected = json["expected_transform"].parseNDArray();
        
        auto [max_diff, avg_diff, mismatches] = compareArrays(transform_got, transform_expected, tolerance);
        
        std::cout << "\nTransform matrix comparison:\n";
        std::cout << "Expected:\n";
        for (uint32_t i = 0; i < K; ++i) {
            for (uint32_t j = 0; j < K; ++j)
                std::cout << std::setw(12) << std::fixed << std::setprecision(6) << transform_expected[i*K + j];
            std::cout << "\n";
        }
        
        std::cout << "\nGot (GPU):\n";
        for (uint32_t i = 0; i < K; ++i) {
            for (uint32_t j = 0; j < K; ++j)
                std::cout << std::setw(12) << std::fixed << std::setprecision(6) << transform_got[i*K + j];
            std::cout << "\n";
        }
        
        std::cout << "\nDifference:\n";
        for (uint32_t i = 0; i < K; ++i) {
            for (uint32_t j = 0; j < K; ++j) {
                float diff = std::abs(transform_got[i*K + j] - transform_expected[i*K + j]);
                std::cout << std::setw(12) << std::fixed << std::setprecision(6) << diff;
            }
            std::cout << "\n";
        }
        
        std::cout << "\n  Max diff: " << max_diff << "\n";
        std::cout << "  Avg diff: " << avg_diff << "\n";
        std::cout << "  Mismatches: " << mismatches << " / " << K*K << "\n";
        
        if (mismatches == 0) {
            std::cout << "\n✓ STAGE 2 PASSED!\n";
        } else {
            std::cout << "\n✗ STAGE 2 FAILED!\n";
            all_passed = false;
        }
    }
    
    //==========================================================================
    // SUMMARY
    //==========================================================================
    std::cout << "\n═══════════════════════════════════════════\n";
    std::cout << "  SUMMARY\n";
    std::cout << "═══════════════════════════════════════════\n";
    
    if (all_passed) {
        std::cout << "✓ ALL STAGES PASSED!\n";
    } else {
        std::cout << "✗ SOME STAGES FAILED!\n";
        throw std::runtime_error("Transform path test failed");
    }
}

int main() {
    try {
        test();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

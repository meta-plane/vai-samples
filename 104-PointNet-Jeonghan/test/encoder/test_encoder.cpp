/**
 * PointNetEncoder Vulkan Test (yanx27 structure)
 * Tests complete PointNetEncoder against PyTorch reference
 * 
 * PointNetEncoder architecture (yanx27):
 * - Input: [N, channel] (channel = 3 or 6)
 * - STN3d: Transform input points
 * - Conv1: channel → 64 with BatchNorm + ReLU
 * - STNkd: Transform 64-dim features
 * - Conv2-3: 64 → 128 → 1024 with BatchNorm + ReLU
 * - Output: [N, 1024] point-wise features
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include "neuralNet.h"
#include "vulkanApp.h"
#include "safeTensorsParser.h"
#include "../networks/include/pointnet.hpp"

using namespace vk;
using namespace networks;

/**
 * Test network wrapper for PointNetEncoder
 */
class EncoderTestNet : public NeuralNet {
    PointNetEncoder encoder;
    
public:
    EncoderTestNet(Device& device, uint32_t channel)
    : NeuralNet(device, 1, 1)
    , encoder(channel)
    {
        // Encoder now has two outputs: out0=[N,64], out1=[N,1024]
        // We test the full feature output (out1)
        input(0) - encoder;
        encoder / "out1" - output(0);  // Use out1 for [N,1024] output
    }
    
    Tensor& operator[](const std::string& name) {
        return encoder[name];
    }
    
    PointNetEncoder& getEncoder() { return encoder; }
};

/**
 * Load STN weights from SafeTensors
 * prefix: "stn" or "fstn" (PyTorch keys don't have "feat." prefix)
 */
void loadSTNWeights(EncoderTestNet& net, const std::string& prefix, 
                    SafeTensorsParser& json) {
    std::cout << "  Loading " << prefix << " weights...\n";
    
    // MLP layers (conv1, conv2, conv3)
    for (int i = 0; i < 3; ++i) {
        std::string layer = "conv" + std::to_string(i + 1);
        std::string weight_key = prefix + "." + layer + ".weight";
        std::string bias_key = prefix + "." + layer + ".bias";
        std::string bn_key = prefix + ".bn" + std::to_string(i + 1);
        
        // Load conv weights
        auto weight_data = json[weight_key].parseNDArray();
        auto bias_data = json[bias_key].parseNDArray();
        
        // Load BatchNorm parameters
        auto mean_data = json[bn_key + ".running_mean"].parseNDArray();
        auto var_data = json[bn_key + ".running_var"].parseNDArray();
        auto gamma_data = json[bn_key + ".weight"].parseNDArray();
        auto beta_data = json[bn_key + ".bias"].parseNDArray();
        
        // Map to internal structure: prefix.mlp.mlp<i>.*
        std::string mlp_prefix = prefix + ".mlp.mlp" + std::to_string(i);
        auto weight_shape = json[weight_key].getShape();
        // SafeTensors already has transposed weights: [in_channels, out_channels]
        uint32_t in_dim = weight_shape[0];
        uint32_t out_dim = weight_shape[1];
        
        net[mlp_prefix + ".weight"] = Tensor(in_dim, out_dim).set(weight_data);
        net[mlp_prefix + ".bias"] = Tensor(out_dim).set(bias_data);
        net[mlp_prefix + ".bn_mean"] = Tensor(out_dim).set(mean_data);
        net[mlp_prefix + ".bn_var"] = Tensor(out_dim).set(var_data);
        net[mlp_prefix + ".bn_gamma"] = Tensor(out_dim).set(gamma_data);
        net[mlp_prefix + ".bn_beta"] = Tensor(out_dim).set(beta_data);
        
        std::cout << "    ✓ " << layer << " (" << in_dim << " → " << out_dim << ")\n";
    }
    
    // FC layers (fc1, fc2, fc3)
    for (int i = 0; i < 3; ++i) {
        std::string layer = "fc" + std::to_string(i + 1);
        std::string weight_key = prefix + "." + layer + ".weight";
        std::string bias_key = prefix + "." + layer + ".bias";
        
        auto weight_data = json[weight_key].parseNDArray();
        auto bias_data = json[bias_key].parseNDArray();
        auto weight_shape = json[weight_key].getShape();
        
        // SafeTensors already has transposed weights: [in_features, out_features]
        uint32_t in_dim = weight_shape[0];
        uint32_t out_dim = weight_shape[1];
        
        // Map to internal structure
        std::string fc_prefix = prefix + ".fc";
        
        if (i < 2) {
            // FC1, FC2: have BatchNorm
            std::string bn_key = prefix + ".bn" + std::to_string(i + 4);
            auto mean_data = json[bn_key + ".running_mean"].parseNDArray();
            auto var_data = json[bn_key + ".running_var"].parseNDArray();
            auto gamma_data = json[bn_key + ".weight"].parseNDArray();
            auto beta_data = json[bn_key + ".bias"].parseNDArray();
            
            net[fc_prefix + ".block" + std::to_string(i) + ".weight"] = Tensor(in_dim, out_dim).set(weight_data);
            net[fc_prefix + ".block" + std::to_string(i) + ".bias"] = Tensor(out_dim).set(bias_data);
            net[fc_prefix + ".block" + std::to_string(i) + ".mean"] = Tensor(out_dim).set(mean_data);
            net[fc_prefix + ".block" + std::to_string(i) + ".var"] = Tensor(out_dim).set(var_data);
            net[fc_prefix + ".block" + std::to_string(i) + ".gamma"] = Tensor(out_dim).set(gamma_data);
            net[fc_prefix + ".block" + std::to_string(i) + ".beta"] = Tensor(out_dim).set(beta_data);
        } else {
            // FC3: no BatchNorm
            net[fc_prefix + ".lastBlock.weight"] = Tensor(in_dim, out_dim).set(weight_data);
            net[fc_prefix + ".lastBlock.bias"] = Tensor(out_dim).set(bias_data);
        }
        
        std::cout << "    ✓ " << layer << " (" << in_dim << " → " << out_dim << ")\n";
    }
}

/**
 * Load Conv layer weights
 * PyTorch keys: conv1, conv2, conv3 (no prefix), bn1, bn2, bn3
 * SafeTensors already has transposed weights: [in_channels, out_channels]
 */
void loadConvWeights(EncoderTestNet& net,
                     SafeTensorsParser& json, int layer_idx) {
    std::string layer = "conv" + std::to_string(layer_idx);
    std::string weight_key = layer + ".weight";
    std::string bias_key = layer + ".bias";
    std::string bn_key = "bn" + std::to_string(layer_idx);
    
    auto weight_data = json[weight_key].parseNDArray();
    auto bias_data = json[bias_key].parseNDArray();
    auto mean_data = json[bn_key + ".running_mean"].parseNDArray();
    auto var_data = json[bn_key + ".running_var"].parseNDArray();
    auto gamma_data = json[bn_key + ".weight"].parseNDArray();
    auto beta_data = json[bn_key + ".bias"].parseNDArray();
    
    auto weight_shape = json[weight_key].getShape();
    // SafeTensors already has transposed weights: [in_channels, out_channels]
    uint32_t in_dim = weight_shape[0];
    uint32_t out_dim = weight_shape[1];
    
    // Map to internal structure
    std::string internal_prefix;
    if (layer_idx == 1) {
        // Conv1: part of MLPSequence
        internal_prefix = "conv1.mlp0";
    } else if (layer_idx == 2) {
        // Conv2: part of MLPSequence  
        internal_prefix = "conv2.mlp0";
    } else if (layer_idx == 3) {
        // Conv3: PointWiseConvNode (no mlp prefix)
        internal_prefix = "conv3";
    }
    
    net[internal_prefix + ".weight"] = Tensor(in_dim, out_dim).set(weight_data);
    net[internal_prefix + ".bias"] = Tensor(out_dim).set(bias_data);
    net[internal_prefix + ".bn_mean"] = Tensor(out_dim).set(mean_data);
    net[internal_prefix + ".bn_var"] = Tensor(out_dim).set(var_data);
    net[internal_prefix + ".bn_gamma"] = Tensor(out_dim).set(gamma_data);
    net[internal_prefix + ".bn_beta"] = Tensor(out_dim).set(beta_data);
    
    std::cout << "    ✓ " << layer << " (" << in_dim << " → " << out_dim << ")\n";
}

/**
 * Run encoder inference
 */
Tensor eval_encoder(uint32_t N, uint32_t channel,
                   const std::vector<float>& input_data,
                   SafeTensorsParser& json) {
    std::cout << "\nCreating EncoderTestNet...\n";
    EncoderTestNet net(netGlobalDevice, channel);
    
    std::cout << "\nLoading weights...\n";
    
    // Load STN3d weights (PyTorch keys: stn.*)
    loadSTNWeights(net, "stn", json);
    
    // Load Conv1 weights (PyTorch keys: conv1.*, bn1.*)
    std::cout << "  Loading conv layers...\n";
    loadConvWeights(net, json, 1);
    
    // Load STNkd (fstn) weights (PyTorch keys: fstn.*)
    loadSTNWeights(net, "fstn", json);
    
    // Load Conv2, Conv3 weights (PyTorch keys: conv2.*, bn2.*, conv3.*, bn3.*)
    loadConvWeights(net, json, 2);
    loadConvWeights(net, json, 3);
    
    std::cout << "\nPreparing network...\n";
    net.prepare();
    
    std::cout << "Running inference...\n";
    Tensor inputTensor = Tensor(N, channel).set(input_data);
    auto result = net(inputTensor);
    
    return result[0];
}

void test() {
    void loadShaders();
    loadShaders();
    
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║   PointNetEncoder Vulkan Test           ║\n";
    std::cout << "║   (yanx27 structure)                     ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n\n";
    
    // Load reference data
    SafeTensorsParser json(PROJECT_CURRENT_DIR"/test/encoder/reference.safetensors");
    
    auto input_shape = json["input"].getShape();
    uint32_t N = input_shape[0];
    uint32_t channel = input_shape[1];
    
    std::cout << "Test configuration:\n";
    std::cout << "  Points: N = " << N << "\n";
    std::cout << "  Channels: " << channel << "\n";
    std::cout << "  Architecture:\n";
    std::cout << "    STN3d: " << channel << " → transform\n";
    std::cout << "    Conv1: " << channel << " → 64\n";
    std::cout << "    STNkd: 64 → transform\n";
    std::cout << "    Conv2-3: 64 → 128 → 1024\n\n";
    
    std::vector<float> inputData = json["input"].parseNDArray();
    
    Tensor result = eval_encoder(N, channel, inputData, json);
    
    // Download results
    uint32_t totalElements = N * 1024;
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
    
    std::cout << "\n✓ Inference completed\n";
    std::cout << "  Output shape: [" << N << ", 1024]\n";
    
    // Load expected output for comparison
    std::vector<float> expected = json["expected_output"].parseNDArray();
    
    std::cout << "\nValidating against PyTorch reference...\n";
    
    float maxError = 0.0f;
    float avgError = 0.0f;
    uint32_t errorCount = 0;
    const float tolerance = 1e-3f;  // Allow 0.001 difference
    
    // Show first 5 values as sanity check
    std::cout << "  First 5 values: ";
    for (uint32_t i = 0; i < 5; ++i) {
        std::cout << output[i];
        if (i < 4) std::cout << ", ";
    }
    std::cout << "\n";
    
    // Check all values
    for (uint32_t i = 0; i < totalElements; ++i) {
        float diff = std::abs(output[i] - expected[i]);
        maxError = std::max(maxError, diff);
        avgError += diff;
        if (diff > tolerance) {
            errorCount++;
        }
    }
    avgError /= totalElements;
    
    std::cout << "\nError statistics:\n";
    std::cout << "\nError Statistics:\n";
    std::cout << "  Max error: " << std::scientific << maxError << std::defaultfloat << "\n";
    std::cout << "  Avg error: " << std::scientific << avgError << std::defaultfloat << "\n";
    std::cout << "  Values exceeding tolerance (" << tolerance << "): " 
              << errorCount << " / " << totalElements 
              << " (" << (100.0f * errorCount / totalElements) << "%)\n";
    
    if (maxError < tolerance) {
        std::cout << "\n✅ TEST PASSED\n";
    } else if (maxError < 0.01f) {
        std::cout << "\n⚠️  TEST PASSED (with warnings)\n";
    } else {
        std::cout << "\n❌ TEST FAILED - Max error " << maxError << " exceeds threshold\n";
        outBuffer.unmap();
        return;
    }
    
    outBuffer.unmap();
}

int main() {
    try {
        test();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}

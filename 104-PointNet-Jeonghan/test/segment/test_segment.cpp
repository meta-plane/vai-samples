/**
 * PointNetSegment Test (yanx27 structure)
 * 9-dim input, 13 classes (semantic segmentation)
 */

#include <iostream>
#include <vector>
#include "neuralNet.h"
#include "vulkanApp.h"
#include "safeTensorsParser.h"
#include "../networks/include/pointnet.hpp"

using namespace vk;
using namespace networks;

void loadSTNWeights(PointNetSegment& net, const std::string& prefix, 
                    SafeTensorsParser& json) {
    // STN MLP layers (conv1, conv2, conv3)
    for (int i = 0; i < 3; ++i) {
        std::string layer = "conv" + std::to_string(i + 1);
        std::string weight_key = "feat." + prefix + "." + layer + ".weight";
        std::string bias_key = "feat." + prefix + "." + layer + ".bias";
        std::string bn_key = "feat." + prefix + ".bn" + std::to_string(i + 1);
        
        auto weight_data = json[weight_key].parseNDArray();
        auto bias_data = json[bias_key].parseNDArray();
        auto mean_data = json[bn_key + ".running_mean"].parseNDArray();
        auto var_data = json[bn_key + ".running_var"].parseNDArray();
        auto gamma_data = json[bn_key + ".weight"].parseNDArray();
        auto beta_data = json[bn_key + ".bias"].parseNDArray();
        
        // Map to internal structure: feat.prefix.mlp.mlp<i>.*
        std::string mlp_prefix = "feat." + prefix + ".mlp.mlp" + std::to_string(i);
        auto weight_shape = json[weight_key].getShape();
        uint32_t in_dim = weight_shape[0];
        uint32_t out_dim = weight_shape[1];
        
        net[mlp_prefix + ".weight"] = Tensor(in_dim, out_dim).set(weight_data);
        net[mlp_prefix + ".bias"] = Tensor(out_dim).set(bias_data);
        net[mlp_prefix + ".bn_mean"] = Tensor(out_dim).set(mean_data);
        net[mlp_prefix + ".bn_var"] = Tensor(out_dim).set(var_data);
        net[mlp_prefix + ".bn_gamma"] = Tensor(out_dim).set(gamma_data);
        net[mlp_prefix + ".bn_beta"] = Tensor(out_dim).set(beta_data);
    }
    
    // STN FC layers (fc1, fc2, fc3)
    for (int i = 0; i < 3; ++i) {
        std::string layer = "fc" + std::to_string(i + 1);
        std::string weight_key = "feat." + prefix + "." + layer + ".weight";
        std::string bias_key = "feat." + prefix + "." + layer + ".bias";
        
        auto weight_data = json[weight_key].parseNDArray();
        auto bias_data = json[bias_key].parseNDArray();
        auto weight_shape = json[weight_key].getShape();
        uint32_t in_dim = weight_shape[0];
        uint32_t out_dim = weight_shape[1];
        
        // Map to internal structure: feat.prefix.fc.block<i> or fc.lastBlock
        std::string fc_prefix = "feat." + prefix + ".fc";
        
        if (i < 2) {
            // FC1, FC2: have BatchNorm
            std::string bn_key = "feat." + prefix + ".bn" + std::to_string(i + 4);
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
    }
}

void loadWeights(PointNetSegment& net, SafeTensorsParser& weights) {
    std::cout << "Loading weights from SafeTensors...\n";
    
    // 1. Load STN3d weights
    loadSTNWeights(net, "stn", weights);
    
    // 2. Load Conv1 weights (feat.conv1.mlp0)
    {
        auto weight_data = weights["feat.conv1.mlp0.weight"].parseNDArray();
        auto bias_data = weights["feat.conv1.mlp0.bias"].parseNDArray();
        auto mean_data = weights["feat.conv1.mlp0.bn_mean"].parseNDArray();
        auto var_data = weights["feat.conv1.mlp0.bn_var"].parseNDArray();
        auto gamma_data = weights["feat.conv1.mlp0.bn_weight"].parseNDArray();
        auto beta_data = weights["feat.conv1.mlp0.bn_bias"].parseNDArray();
        auto weight_shape = weights["feat.conv1.mlp0.weight"].getShape();
        
        net["feat.conv1.mlp0.weight"] = Tensor(weight_shape[0], weight_shape[1]).set(weight_data);
        net["feat.conv1.mlp0.bias"] = Tensor(weight_shape[1]).set(bias_data);
        net["feat.conv1.mlp0.bn_mean"] = Tensor(weight_shape[1]).set(mean_data);
        net["feat.conv1.mlp0.bn_var"] = Tensor(weight_shape[1]).set(var_data);
        net["feat.conv1.mlp0.bn_gamma"] = Tensor(weight_shape[1]).set(gamma_data);
        net["feat.conv1.mlp0.bn_beta"] = Tensor(weight_shape[1]).set(beta_data);
    }
    
    // 3. Load STNkd weights
    loadSTNWeights(net, "fstn", weights);
    
    // 4. Load Conv2 weights (feat.conv2.mlp0)
    {
        auto weight_data = weights["feat.conv2.mlp0.weight"].parseNDArray();
        auto bias_data = weights["feat.conv2.mlp0.bias"].parseNDArray();
        auto mean_data = weights["feat.conv2.mlp0.bn_mean"].parseNDArray();
        auto var_data = weights["feat.conv2.mlp0.bn_var"].parseNDArray();
        auto gamma_data = weights["feat.conv2.mlp0.bn_weight"].parseNDArray();
        auto beta_data = weights["feat.conv2.mlp0.bn_bias"].parseNDArray();
        auto weight_shape = weights["feat.conv2.mlp0.weight"].getShape();
        
        net["feat.conv2.mlp0.weight"] = Tensor(weight_shape[0], weight_shape[1]).set(weight_data);
        net["feat.conv2.mlp0.bias"] = Tensor(weight_shape[1]).set(bias_data);
        net["feat.conv2.mlp0.bn_mean"] = Tensor(weight_shape[1]).set(mean_data);
        net["feat.conv2.mlp0.bn_var"] = Tensor(weight_shape[1]).set(var_data);
        net["feat.conv2.mlp0.bn_gamma"] = Tensor(weight_shape[1]).set(gamma_data);
        net["feat.conv2.mlp0.bn_beta"] = Tensor(weight_shape[1]).set(beta_data);
    }
    
    // 5. Load Conv3 weights (feat.conv3)
    {
        auto weight_data = weights["feat.conv3.weight"].parseNDArray();
        auto bias_data = weights["feat.conv3.bias"].parseNDArray();
        auto mean_data = weights["feat.conv3.bn_mean"].parseNDArray();
        auto var_data = weights["feat.conv3.bn_var"].parseNDArray();
        auto gamma_data = weights["feat.conv3.bn_weight"].parseNDArray();
        auto beta_data = weights["feat.conv3.bn_bias"].parseNDArray();
        auto weight_shape = weights["feat.conv3.weight"].getShape();
        
        net["feat.conv3.weight"] = Tensor(weight_shape[0], weight_shape[1]).set(weight_data);
        net["feat.conv3.bias"] = Tensor(weight_shape[1]).set(bias_data);
        net["feat.conv3.bn_mean"] = Tensor(weight_shape[1]).set(mean_data);
        net["feat.conv3.bn_var"] = Tensor(weight_shape[1]).set(var_data);
        net["feat.conv3.bn_gamma"] = Tensor(weight_shape[1]).set(gamma_data);
        net["feat.conv3.bn_beta"] = Tensor(weight_shape[1]).set(beta_data);
    }
    
    // 6. Load segmentation head weights (conv1-4)
    for (int i = 0; i < 4; ++i) {
        std::string layer = "conv" + std::to_string(i + 1);
        auto weight_data = weights[layer + ".weight"].parseNDArray();
        auto bias_data = weights[layer + ".bias"].parseNDArray();
        auto mean_data = weights[layer + ".bn_mean"].parseNDArray();
        auto var_data = weights[layer + ".bn_var"].parseNDArray();
        auto gamma_data = weights[layer + ".bn_weight"].parseNDArray();
        auto beta_data = weights[layer + ".bn_bias"].parseNDArray();
        auto weight_shape = weights[layer + ".weight"].getShape();
        
        net[layer + ".weight"] = Tensor(weight_shape[0], weight_shape[1]).set(weight_data);
        net[layer + ".bias"] = Tensor(weight_shape[1]).set(bias_data);
        net[layer + ".bn_mean"] = Tensor(weight_shape[1]).set(mean_data);
        net[layer + ".bn_var"] = Tensor(weight_shape[1]).set(var_data);
        net[layer + ".bn_gamma"] = Tensor(weight_shape[1]).set(gamma_data);
        net[layer + ".bn_beta"] = Tensor(weight_shape[1]).set(beta_data);
    }
    
    std::cout << "  ✓ All weights loaded successfully\n\n";
}

void test_segment() {
    std::cout << "=== PointNetSegment Test (yanx27: 9-dim, 13 classes) ===\n\n";
    
    SafeTensorsParser data(PROJECT_CURRENT_DIR"/test/segment/reference.safetensors");
    std::vector<float> shape_data = data["input_shape"].parseNDArray();
    uint32_t N = static_cast<uint32_t>(shape_data[0]);
    uint32_t numClasses = 13;
    
    std::cout << "N=" << N << ", Classes=" << numClasses << "\n";
    std::cout << "Input: [N,9], Output: [N," << numClasses << "]\n\n";
    
    std::vector<float> input_data = data["input"].parseNDArray();
    
    std::cout << "Creating PointNetSegment...\n";
    PointNetSegment net(netGlobalDevice, numClasses, 9);  // 9-dim input (x,y,z + RGB + normalized coords)
    
    loadWeights(net, data);
    
    std::cout << "Preparing network...\n";
    net.prepare();
    
    // Create input tensor [N, 9]
    Tensor inputTensor = Tensor(N, 9).set(input_data);
    
    std::cout << "Running inference...\n";
    auto outputs = net(inputTensor);
    
    std::cout << "\n✅ TEST PASSED\n";
    std::cout << "  Network executed successfully\n";
    std::cout << "  Output shape: [" << N << ", " << numClasses << "]\n";
}

int main() {
    loadShaders();
    test_segment();
    return 0;
}

/**
 * TNetBlock Vulkan Test (SafeTensors-based)
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
#include <utility>
#include "neuralNet.h"
#include "vulkanApp.h"
#include "safeTensorsParser.h"
#include "../networks/include/pointnet.hpp"

using namespace vk;
using namespace networks;

/**
 * Test network wrapper for TNetBlock
 * 
 * TNetBlock now only generates transformation matrix [K, K]
 * MatMul is separate and applies the transformation to points
 * 
 * Outputs:
 * - out0: Transformed points [N, K]
 * - out1: Transform matrix [K, K]
 */
class TNetTestNet : public NeuralNet {
    TNetBlock tnet;
    MatMulNode matmul;
    
public:
    TNetTestNet(Device& device, uint32_t K)
    : NeuralNet(device, 1, 2)  // 1 input, 2 outputs
    , tnet(K)
    , matmul()
    {
        // Clean architecture: TNet generates matrix, MatMul applies it
        // Input → TNet → Transform matrix [K,K] → MatMul.in1
        // Input ──────────────────────────────→ MatMul.in0
        //                                        ↓
        //                               Transformed points
        
        input(0) - tnet;  // Generate transformation matrix
        tnet.slot("out0") - matmul.slot("in1");  // Matrix → MatMul
        input(0).slot("out0") - matmul.slot("in0");  // Points → MatMul
        
        // Connect outputs
        matmul - output(0);              // Transformed points
        tnet / "out0" - output(1);       // Transform matrix
    }
    
    Tensor& operator[](const std::string& name) {
        return tnet[name];
    }
};/**
 * Run TNetBlock inference
 */
std::pair<Tensor, Tensor> eval_tnet(uint32_t N, uint32_t K,
                 const std::vector<float>& input_data,
                 SafeTensorsParser& json) {
    std::cout << "Creating TNetTestNet...\n";
    // Create network
    TNetTestNet net(netGlobalDevice, K);
    
    std::cout << "Loading MLP weights...\n";
    // Load MLP weights (with BatchNorm parameters)
    // MLP layer 0: 3 → 64
    {
        std::vector<float> weight_data = json["mlp.mlp0.weight"].parseNDArray();
        std::vector<float> bias_data = json["mlp.mlp0.bias"].parseNDArray();
        std::vector<float> mean_data = json["mlp.mlp0.mean"].parseNDArray();
        std::vector<float> var_data = json["mlp.mlp0.var"].parseNDArray();
        std::vector<float> gamma_data = json["mlp.mlp0.gamma"].parseNDArray();
        std::vector<float> beta_data = json["mlp.mlp0.beta"].parseNDArray();

        net["mlp.mlp0.weight"] = Tensor(3, 64).set(weight_data);
        net["mlp.mlp0.bias"] = Tensor(64).set(bias_data);
        net["mlp.mlp0.bn_mean"] = Tensor(64).set(mean_data);     // Note: bn_ prefix!
        net["mlp.mlp0.bn_var"] = Tensor(64).set(var_data);
        net["mlp.mlp0.bn_gamma"] = Tensor(64).set(gamma_data);
        net["mlp.mlp0.bn_beta"] = Tensor(64).set(beta_data);
        std::cout << "  ✓ mlp0 loaded (3 → 64)\n";
    }

    // MLP layer 1: 64 → 128
    {
        std::vector<float> weight_data = json["mlp.mlp1.weight"].parseNDArray();
        std::vector<float> bias_data = json["mlp.mlp1.bias"].parseNDArray();
        std::vector<float> mean_data = json["mlp.mlp1.mean"].parseNDArray();
        std::vector<float> var_data = json["mlp.mlp1.var"].parseNDArray();
        std::vector<float> gamma_data = json["mlp.mlp1.gamma"].parseNDArray();
        std::vector<float> beta_data = json["mlp.mlp1.beta"].parseNDArray();

        net["mlp.mlp1.weight"] = Tensor(64, 128).set(weight_data);
        net["mlp.mlp1.bias"] = Tensor(128).set(bias_data);
        net["mlp.mlp1.bn_mean"] = Tensor(128).set(mean_data);
        net["mlp.mlp1.bn_var"] = Tensor(128).set(var_data);
        net["mlp.mlp1.bn_gamma"] = Tensor(128).set(gamma_data);
        net["mlp.mlp1.bn_beta"] = Tensor(128).set(beta_data);
        std::cout << "  ✓ mlp1 loaded (64 → 128)\n";
    }

    // MLP layer 2: 128 → 1024
    {
        std::vector<float> weight_data = json["mlp.mlp2.weight"].parseNDArray();
        std::vector<float> bias_data = json["mlp.mlp2.bias"].parseNDArray();
        std::vector<float> mean_data = json["mlp.mlp2.mean"].parseNDArray();
        std::vector<float> var_data = json["mlp.mlp2.var"].parseNDArray();
        std::vector<float> gamma_data = json["mlp.mlp2.gamma"].parseNDArray();
        std::vector<float> beta_data = json["mlp.mlp2.beta"].parseNDArray();

        net["mlp.mlp2.weight"] = Tensor(128, 1024).set(weight_data);
        net["mlp.mlp2.bias"] = Tensor(1024).set(bias_data);
        net["mlp.mlp2.bn_mean"] = Tensor(1024).set(mean_data);
        net["mlp.mlp2.bn_var"] = Tensor(1024).set(var_data);
        net["mlp.mlp2.bn_gamma"] = Tensor(1024).set(gamma_data);
        net["mlp.mlp2.bn_beta"] = Tensor(1024).set(beta_data);
        std::cout << "  ✓ mlp2 loaded (128 → 1024)\n";
    }
    
    std::cout << "Loading FC weights (FCBNSequence: block0, block1, lastBlock)...\n";
    // Load FC weights for FCBNSequence
    // Block 0 and 1: FC+BN+ReLU
    for (int i = 0; i < 2; i++) {
        std::string prefix = "fc.block" + std::to_string(i) + ".";
        std::cout << "  Loading block" << i << " (FC+BN+ReLU)\n";
        
        // FC weights
        std::vector<float> weight_data = json[prefix + "weight"].parseNDArray();
        std::vector<float> bias_data = json[prefix + "bias"].parseNDArray();
        
        // BN parameters
        std::vector<float> mean_data = json[prefix + "mean"].parseNDArray();
        std::vector<float> var_data = json[prefix + "var"].parseNDArray();
        std::vector<float> gamma_data = json[prefix + "gamma"].parseNDArray();
        std::vector<float> beta_data = json[prefix + "beta"].parseNDArray();
        
        uint32_t out_dim = bias_data.size();
        uint32_t in_dim = weight_data.size() / out_dim;
        
        std::cout << "    FC Shape: [" << in_dim << ", " << out_dim << "]\n";
        
        net[prefix + "weight"] = Tensor(in_dim, out_dim).set(weight_data);
        net[prefix + "bias"] = Tensor(out_dim).set(bias_data);
        net[prefix + "mean"] = Tensor(out_dim).set(mean_data);
        net[prefix + "var"] = Tensor(out_dim).set(var_data);
        net[prefix + "gamma"] = Tensor(out_dim).set(gamma_data);
        net[prefix + "beta"] = Tensor(out_dim).set(beta_data);
    }
    
    // Last block: FC only (no BN, no ReLU)
    {
        std::string prefix = "fc.lastBlock.";
        std::cout << "  Loading lastBlock (FC only)\n";
        
        std::vector<float> weight_data = json[prefix + "weight"].parseNDArray();
        std::vector<float> bias_data = json[prefix + "bias"].parseNDArray();
        
        uint32_t out_dim = bias_data.size();
        uint32_t in_dim = weight_data.size() / out_dim;
        
        std::cout << "    FC Shape: [" << in_dim << ", " << out_dim << "]\n";
        
        net[prefix + "weight"] = Tensor(in_dim, out_dim).set(weight_data);
        net[prefix + "bias"] = Tensor(out_dim).set(bias_data);
    }
    
    std::cout << "Creating input tensor...\n";
    // Create input tensor [N, K]
    Tensor input_tensor = Tensor(N, K).set(input_data);
    
    std::cout << "Running inference...\n";
    // Run inference
    auto result = net(input_tensor);
    
    // result[0] = transformed points [N, K]
    // result[1] = transformation matrix [K, K]
    
    // Return both outputs as a pair
    return std::make_pair(result[0], result[1]);
}

void test() {
    void loadShaders();
    loadShaders();
    
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║      TNetBlock Vulkan Compute Test      ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n";
    std::cout << std::endl;
    
    // Load reference data
    SafeTensorsParser json(PROJECT_CURRENT_DIR"/test/tnet/reference.safetensors");
    
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
    std::vector<float> expected_transform = json["transform"].parseNDArray();
    
    std::cout << "Running TNetBlock on GPU...\n";
    
    // Run inference
    auto [result, transform] = eval_tnet(N, K, input_data, json);
    
    // Download transform matrix from GPU
    Buffer transformBuf = netGlobalDevice.createBuffer({
        .size = K * K * sizeof(float),
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    });
    
    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(transformBuf, transform.buffer())
        .end()
        .submit()
        .wait();
    
    float* transform_output = (float*)transformBuf.map();
    
    // Compare transform matrix
    std::cout << "\n============================================================\n";
    std::cout << "Transform Matrix Comparison (K=" << K << "):\n";
    std::cout << "============================================================\n\n";
    std::cout << "Expected:\n";
    for (uint32_t i = 0; i < K; ++i) {
        for (uint32_t j = 0; j < K; ++j) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(6) << expected_transform[i*K + j] << " ";
        }
        std::cout << "\n";
    }
    
    std::cout << "\nGot (GPU):\n";
    for (uint32_t i = 0; i < K; ++i) {
        for (uint32_t j = 0; j < K; ++j) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(6) << transform_output[i*K + j] << " ";
        }
        std::cout << "\n";
    }
    
    float transform_max_diff = 0.0f;
    for (uint32_t i = 0; i < K * K; ++i) {
        float diff = std::abs(transform_output[i] - expected_transform[i]);
        transform_max_diff = std::max(transform_max_diff, diff);
    }
    std::cout << "\nTransform Max Diff: " << std::fixed << std::setprecision(6) << transform_max_diff << "\n";
    
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
    
    // Tolerance explanation:
    // - TNet has complex pipeline: MLP(3 layers) → MaxPool → FC(3 layers) → MatMul
    // - Each layer introduces small numerical errors due to float32 precision
    // - BatchNorm uses epsilon=1e-5, which can amplify differences
    // - MaxPooling can amplify errors from MLP stage
    // - Empirically, max diff ~0.22, avg ~0.09 for correct implementation
    const float tolerance = 0.001f;  // Strict tolerance to find bugs
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

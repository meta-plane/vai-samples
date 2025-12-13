/**
 * PointNetEncoder Test
 * Tests full encoder pipeline against PyTorch reference
 * 
 * Architecture:
 * TNet1 (3x3) → MLP1 (3→64→64) → TNet2 (64x64) → MLP2 (64→128→1024) → MaxPool
 * Input: [N, 3] → Output: [1024] global feature
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include "neuralNet.h"
#include "vulkanApp.h"
#include "jsonParser.h"
#include "../networks/include/pointnet.hpp"

using namespace vk;
using namespace networks;

/**
 * Helper: Load TNet weights from JSON
 */
void loadTNetWeights(TNetBlock& tnet, JsonParser& json, const std::string& prefix) {
    std::cout << "Loading " << prefix << " weights...\n";
    
    // MLP layers (3 layers)
    for (int i = 0; i < 3; ++i) {
        std::string accessor = "mlp.mlp" + std::to_string(i);
        
        // Build JSON keys
        std::string w_key = prefix + ".mlp" + std::to_string(i) + ".weight";
        std::string b_key = prefix + ".mlp" + std::to_string(i) + ".bias";
        std::string mean_key = prefix + ".mlp" + std::to_string(i) + ".bn_mean";
        std::string var_key = prefix + ".mlp" + std::to_string(i) + ".bn_var";
        std::string gamma_key = prefix + ".mlp" + std::to_string(i) + ".bn_gamma";
        std::string beta_key = prefix + ".mlp" + std::to_string(i) + ".bn_beta";
        
        tnet[accessor + ".weight"] = Tensor(
            i == 0 ? (prefix == "tnet1" ? 3 : 64) : (i == 1 ? 64 : 128),
            i == 0 ? 64 : (i == 1 ? 128 : 1024)
        ).set(json[w_key.c_str()].parseNDArray());
        
        tnet[accessor + ".bias"] = Tensor(
            i == 0 ? 64 : (i == 1 ? 128 : 1024)
        ).set(json[b_key.c_str()].parseNDArray());
        
        tnet[accessor + ".bn_mean"] = Tensor(
            i == 0 ? 64 : (i == 1 ? 128 : 1024)
        ).set(json[mean_key.c_str()].parseNDArray());
        
        tnet[accessor + ".bn_var"] = Tensor(
            i == 0 ? 64 : (i == 1 ? 128 : 1024)
        ).set(json[var_key.c_str()].parseNDArray());
        
        tnet[accessor + ".bn_gamma"] = Tensor(
            i == 0 ? 64 : (i == 1 ? 128 : 1024)
        ).set(json[gamma_key.c_str()].parseNDArray());
        
        tnet[accessor + ".bn_beta"] = Tensor(
            i == 0 ? 64 : (i == 1 ? 128 : 1024)
        ).set(json[beta_key.c_str()].parseNDArray());
        
        std::cout << "  ✓ " << accessor << " loaded\n";
    }
    
    // FC layers (3 layers with BatchNorm for first 2)
    for (int i = 0; i < 3; ++i) {
        std::string fc_prefix = prefix + ".fc" + std::to_string(i);
        // Last FC block is named "lastBlock", not "block2"
        std::string accessor = (i < 2) ? "fc.block" + std::to_string(i) : "fc.lastBlock";
        
        uint32_t K = (prefix == "tnet1") ? 3 : 64;
        
        tnet[accessor + ".weight"] = Tensor(
            i == 0 ? 1024 : (i == 1 ? 512 : 256),
            i == 0 ? 512 : (i == 1 ? 256 : K*K)
        ).set(json[fc_prefix + ".weight"].parseNDArray());
        
        tnet[accessor + ".bias"] = Tensor(
            i == 0 ? 512 : (i == 1 ? 256 : K*K)
        ).set(json[fc_prefix + ".bias"].parseNDArray());
        
        // First 2 FC layers have BatchNorm
        if (i < 2) {
            tnet[accessor + ".mean"] = Tensor(
                i == 0 ? 512 : 256
            ).set(json[fc_prefix + ".mean"].parseNDArray());
            
            tnet[accessor + ".var"] = Tensor(
                i == 0 ? 512 : 256
            ).set(json[fc_prefix + ".var"].parseNDArray());
            
            tnet[accessor + ".gamma"] = Tensor(
                i == 0 ? 512 : 256
            ).set(json[fc_prefix + ".gamma"].parseNDArray());
            
            tnet[accessor + ".beta"] = Tensor(
                i == 0 ? 512 : 256
            ).set(json[fc_prefix + ".beta"].parseNDArray());
        }
        
        std::cout << "  ✓ " << accessor << " loaded\n";
    }
}

/**
 * Helper: Load MLPSequence weights
 */
void loadMLPSequenceWeights(MLPSequence<2>& mlp, JsonParser& json, 
                            const std::string& prefix,
                            uint32_t in_dim, uint32_t mid_dim, uint32_t out_dim) {
    std::cout << "Loading " << prefix << " weights...\n";
    
    // MLP layer 0
    mlp["mlp0.weight"] = Tensor(in_dim, mid_dim).set(
        json[prefix + ".mlp0.weight"].parseNDArray()
    );
    mlp["mlp0.bias"] = Tensor(mid_dim).set(
        json[prefix + ".mlp0.bias"].parseNDArray()
    );
    mlp["mlp0.bn_mean"] = Tensor(mid_dim).set(
        json[prefix + ".mlp0.bn_mean"].parseNDArray()
    );
    mlp["mlp0.bn_var"] = Tensor(mid_dim).set(
        json[prefix + ".mlp0.bn_var"].parseNDArray()
    );
    mlp["mlp0.bn_gamma"] = Tensor(mid_dim).set(
        json[prefix + ".mlp0.bn_gamma"].parseNDArray()
    );
    mlp["mlp0.bn_beta"] = Tensor(mid_dim).set(
        json[prefix + ".mlp0.bn_beta"].parseNDArray()
    );
    std::cout << "  ✓ mlp0 loaded (" << in_dim << " → " << mid_dim << ")\n";
    
    // MLP layer 1
    mlp["mlp1.weight"] = Tensor(mid_dim, out_dim).set(
        json[prefix + ".mlp1.weight"].parseNDArray()
    );
    mlp["mlp1.bias"] = Tensor(out_dim).set(
        json[prefix + ".mlp1.bias"].parseNDArray()
    );
    mlp["mlp1.bn_mean"] = Tensor(out_dim).set(
        json[prefix + ".mlp1.bn_mean"].parseNDArray()
    );
    mlp["mlp1.bn_var"] = Tensor(out_dim).set(
        json[prefix + ".mlp1.bn_var"].parseNDArray()
    );
    mlp["mlp1.bn_gamma"] = Tensor(out_dim).set(
        json[prefix + ".mlp1.bn_gamma"].parseNDArray()
    );
    mlp["mlp1.bn_beta"] = Tensor(out_dim).set(
        json[prefix + ".mlp1.bn_beta"].parseNDArray()
    );
    std::cout << "  ✓ mlp1 loaded\n";
}

/**
 * Test network wrapper for PointNetEncoder
 */
class EncoderTestNet : public NeuralNet {
    PointNetEncoder encoder;
    MaxPooling1DNode maxpool;
    
public:
    EncoderTestNet(Device& device)
    : NeuralNet(device, 1, 1)
    , encoder()
    , maxpool()
    {
        // Connect input → encoder → maxpool → output
        // Encoder outputs [N, 1024], MaxPool reduces to [1024]
        input(0) - encoder - maxpool - output(0);
    }
    
    // Expose encoder's sub-components for weight loading
    PointNetEncoder& getEncoder() { return encoder; }
};

/**
 * Run PointNetEncoder inference
 */
Tensor eval_encoder(uint32_t N, const std::vector<float>& input_data, JsonParser& json) {
    std::cout << "Creating EncoderTestNet...\n";
    EncoderTestNet net(netGlobalDevice);
    PointNetEncoder& encoder = net.getEncoder();
    
    // Load TNet1 weights (3x3)
    loadTNetWeights(encoder.tnet1, json, "tnet1");
    
    // Load MLP1 weights (3→64→64)
    loadMLPSequenceWeights(encoder.mlp1, json, "mlp1", 3, 64, 64);
    
    // Load TNet2 weights (64x64)
    loadTNetWeights(encoder.tnet2, json, "tnet2");
    
    // Load MLP2 weights (64→128→1024)
    loadMLPSequenceWeights(encoder.mlp2, json, "mlp2", 64, 128, 1024);
    
    std::cout << "\nPreparing network...\n";
    Tensor inputTensor = Tensor(N, 3).set(input_data);
    net.prepare();
    
    std::cout << "Running inference...\n";
    auto outputs = net(inputTensor);
    
    return outputs[0];
}

void test_encoder() {
    std::cout << "=== PointNetEncoder Test ===\n\n";
    
    // Load reference data
    JsonParser json = JsonParser(PROJECT_CURRENT_DIR"/test/encoder/reference.json");
    
    // Parse parameters
    std::vector<float> shape_data = json["input_shape"].parseNDArray();
    uint32_t N = static_cast<uint32_t>(shape_data[0]);
    
    std::cout << "Test parameters:\n";
    std::cout << "  N (points): " << N << "\n";
    std::cout << "  Input dim: 3\n";
    std::cout << "  Output dim: 1024\n\n";
    
    // Load input and expected output
    std::vector<float> input_data = json["input"].parseNDArray();
    std::vector<float> expected = json["expected_output"].parseNDArray();
    
    // Run inference
    Tensor result = eval_encoder(N, input_data, json);
    
    std::cout << "\nDownloading result from GPU...\n";
    // Download result
    Buffer outBuf = netGlobalDevice.createBuffer({
        .size = 1024 * sizeof(float),
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
    std::cout << "\nComparing results...\n";
    bool all_pass = true;
    float max_error = 0.0f;
    const float TOLERANCE = 1e-3f;
    
    int error_count = 0;
    const int MAX_ERRORS_TO_SHOW = 10;
    
    for (uint32_t i = 0; i < 1024; ++i) {
        float error = std::abs(output[i] - expected[i]);
        max_error = std::max(max_error, error);
        
        if (error > TOLERANCE) {
            if (error_count < MAX_ERRORS_TO_SHOW) {
                std::cout << "  ✗ Index " << i << ": "
                          << "got " << output[i] << ", "
                          << "expected " << expected[i] << ", "
                          << "error " << error << "\n";
            }
            error_count++;
            all_pass = false;
        }
    }
    
    if (error_count > MAX_ERRORS_TO_SHOW) {
        std::cout << "  ... and " << (error_count - MAX_ERRORS_TO_SHOW) 
                  << " more errors\n";
    }
    
    std::cout << "\nMax error: " << max_error << "\n";
    std::cout << "Total errors: " << error_count << " / 1024\n";
    
    // Show first 10 values for debugging
    std::cout << "\nFirst 10 values:\n";
    std::cout << "  Expected: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << expected[i] << " ";
    }
    std::cout << "\n  Got:      ";
    for (int i = 0; i < 10; ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << "\n";
    
    if (all_pass) {
        std::cout << "\n✓ All tests passed!\n";
    } else {
        std::cout << "\n✗ Some tests failed\n";
        throw std::runtime_error("Test failed");
    }
}

int main() {
    void loadShaders();
    loadShaders();
    
    try {
        test_encoder();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

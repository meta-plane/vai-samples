/**
 * PointNet Segmentation Test
 * Tests full segmentation pipeline against PyTorch reference
 * 
 * Architecture:
 * Input [N, 3] → Encoder → [N, 1024]
 *   ├→ MaxPool → [1024] → Broadcast → [N, 1024]
 *   └→ Concat → [N, 2048] → SegHead → [N, numClasses]
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

void test_segment() {
    std::cout << "=== PointNet Segmentation Test ===\n\n";
    
    // Load reference data
    JsonParser json(PROJECT_CURRENT_DIR"/test/segment/reference.json");
    
    uint32_t N = 16;
    uint32_t numClasses = 4;
    
    std::cout << "Test parameters:\n";
    std::cout << "  N (points): " << N << "\n";
    std::cout << "  NumClasses: " << numClasses << "\n\n";
    
    // Load input and expected output
    std::vector<float> input_data = json["input"].parseNDArray();
    std::vector<float> expected = json["expected_output"].parseNDArray();
    
    // Create network directly (no wrapper needed!)
    std::cout << "Creating PointNetSegment...\n";
    PointNetSegment net(netGlobalDevice, numClasses);
    
    std::cout << "Loading weights via operator[]...\n";
    
    // Load all encoder weights using operator[] with "encoder." prefix
    // The operator[] in PointNetSegment strips "encoder." and forwards to encoder's operator[]
    
    // Encoder TNet1 weights
    for (int i = 0; i < 3; ++i) {
        std::string json_key = "encoder.tnet1.mlp" + std::to_string(i);
        std::string cpp_key = "encoder.tnet1.mlp.mlp" + std::to_string(i);
        uint32_t in_dim = i == 0 ? 3 : (i == 1 ? 64 : 128);
        uint32_t out_dim = i == 0 ? 64 : (i == 1 ? 128 : 1024);
        
        net[cpp_key + ".weight"] = Tensor(in_dim, out_dim).set(json[json_key + ".weight"].parseNDArray());
        net[cpp_key + ".bias"] = Tensor(out_dim).set(json[json_key + ".bias"].parseNDArray());
        net[cpp_key + ".bn_mean"] = Tensor(out_dim).set(json[json_key + ".bn_mean"].parseNDArray());
        net[cpp_key + ".bn_var"] = Tensor(out_dim).set(json[json_key + ".bn_var"].parseNDArray());
        net[cpp_key + ".bn_gamma"] = Tensor(out_dim).set(json[json_key + ".bn_gamma"].parseNDArray());
        net[cpp_key + ".bn_beta"] = Tensor(out_dim).set(json[json_key + ".bn_beta"].parseNDArray());
    }
    
    // Encoder TNet1 FC weights
    for (int i = 0; i < 3; ++i) {
        std::string json_key = "encoder.tnet1.fc" + std::to_string(i);
        std::string cpp_key = "encoder.tnet1.fc.";
        cpp_key += (i < 2) ? ("block" + std::to_string(i)) : "lastBlock";
        
        net[cpp_key + ".weight"] = Tensor(
            i == 0 ? 1024 : (i == 1 ? 512 : 256),
            i == 0 ? 512 : (i == 1 ? 256 : 9)
        ).set(json[json_key + ".weight"].parseNDArray());
        net[cpp_key + ".bias"] = Tensor(i == 0 ? 512 : (i == 1 ? 256 : 9))
            .set(json[json_key + ".bias"].parseNDArray());
        
        if (i < 2) {
            net[cpp_key + ".mean"] = Tensor(i == 0 ? 512 : 256).set(json[json_key + ".mean"].parseNDArray());
            net[cpp_key + ".var"] = Tensor(i == 0 ? 512 : 256).set(json[json_key + ".var"].parseNDArray());
            net[cpp_key + ".gamma"] = Tensor(i == 0 ? 512 : 256).set(json[json_key + ".gamma"].parseNDArray());
            net[cpp_key + ".beta"] = Tensor(i == 0 ? 512 : 256).set(json[json_key + ".beta"].parseNDArray());
        }
    }
    
    // Encoder MLP1 weights
    net["encoder.mlp1.mlp0.weight"] = Tensor(3, 64).set(json["encoder.mlp1.mlp0.weight"].parseNDArray());
    net["encoder.mlp1.mlp0.bias"] = Tensor(64).set(json["encoder.mlp1.mlp0.bias"].parseNDArray());
    net["encoder.mlp1.mlp0.bn_mean"] = Tensor(64).set(json["encoder.mlp1.mlp0.bn_mean"].parseNDArray());
    net["encoder.mlp1.mlp0.bn_var"] = Tensor(64).set(json["encoder.mlp1.mlp0.bn_var"].parseNDArray());
    net["encoder.mlp1.mlp0.bn_gamma"] = Tensor(64).set(json["encoder.mlp1.mlp0.bn_gamma"].parseNDArray());
    net["encoder.mlp1.mlp0.bn_beta"] = Tensor(64).set(json["encoder.mlp1.mlp0.bn_beta"].parseNDArray());
    
    net["encoder.mlp1.mlp1.weight"] = Tensor(64, 64).set(json["encoder.mlp1.mlp1.weight"].parseNDArray());
    net["encoder.mlp1.mlp1.bias"] = Tensor(64).set(json["encoder.mlp1.mlp1.bias"].parseNDArray());
    net["encoder.mlp1.mlp1.bn_mean"] = Tensor(64).set(json["encoder.mlp1.mlp1.bn_mean"].parseNDArray());
    net["encoder.mlp1.mlp1.bn_var"] = Tensor(64).set(json["encoder.mlp1.mlp1.bn_var"].parseNDArray());
    net["encoder.mlp1.mlp1.bn_gamma"] = Tensor(64).set(json["encoder.mlp1.mlp1.bn_gamma"].parseNDArray());
    net["encoder.mlp1.mlp1.bn_beta"] = Tensor(64).set(json["encoder.mlp1.mlp1.bn_beta"].parseNDArray());
    
    // Encoder TNet2 weights
    for (int i = 0; i < 3; ++i) {
        std::string json_key = "encoder.tnet2.mlp" + std::to_string(i);
        std::string cpp_key = "encoder.tnet2.mlp.mlp" + std::to_string(i);
        uint32_t in_dim = i == 0 ? 64 : (i == 1 ? 64 : 128);
        uint32_t out_dim = i == 0 ? 64 : (i == 1 ? 128 : 1024);
        
        net[cpp_key + ".weight"] = Tensor(in_dim, out_dim).set(json[json_key + ".weight"].parseNDArray());
        net[cpp_key + ".bias"] = Tensor(out_dim).set(json[json_key + ".bias"].parseNDArray());
        net[cpp_key + ".bn_mean"] = Tensor(out_dim).set(json[json_key + ".bn_mean"].parseNDArray());
        net[cpp_key + ".bn_var"] = Tensor(out_dim).set(json[json_key + ".bn_var"].parseNDArray());
        net[cpp_key + ".bn_gamma"] = Tensor(out_dim).set(json[json_key + ".bn_gamma"].parseNDArray());
        net[cpp_key + ".bn_beta"] = Tensor(out_dim).set(json[json_key + ".bn_beta"].parseNDArray());
    }
    
    // Encoder TNet2 FC weights
    for (int i = 0; i < 3; ++i) {
        std::string json_key = "encoder.tnet2.fc" + std::to_string(i);
        std::string cpp_key = "encoder.tnet2.fc.";
        cpp_key += (i < 2) ? ("block" + std::to_string(i)) : "lastBlock";
        
        net[cpp_key + ".weight"] = Tensor(
            i == 0 ? 1024 : (i == 1 ? 512 : 256),
            i == 0 ? 512 : (i == 1 ? 256 : 4096)
        ).set(json[json_key + ".weight"].parseNDArray());
        net[cpp_key + ".bias"] = Tensor(i == 0 ? 512 : (i == 1 ? 256 : 4096))
            .set(json[json_key + ".bias"].parseNDArray());
        
        if (i < 2) {
            net[cpp_key + ".mean"] = Tensor(i == 0 ? 512 : 256).set(json[json_key + ".mean"].parseNDArray());
            net[cpp_key + ".var"] = Tensor(i == 0 ? 512 : 256).set(json[json_key + ".var"].parseNDArray());
            net[cpp_key + ".gamma"] = Tensor(i == 0 ? 512 : 256).set(json[json_key + ".gamma"].parseNDArray());
            net[cpp_key + ".beta"] = Tensor(i == 0 ? 512 : 256).set(json[json_key + ".beta"].parseNDArray());
        }
    }
    
    // Encoder MLP2 weights
    net["encoder.mlp2.mlp0.weight"] = Tensor(64, 128).set(json["encoder.mlp2.mlp0.weight"].parseNDArray());
    net["encoder.mlp2.mlp0.bias"] = Tensor(128).set(json["encoder.mlp2.mlp0.bias"].parseNDArray());
    net["encoder.mlp2.mlp0.bn_mean"] = Tensor(128).set(json["encoder.mlp2.mlp0.bn_mean"].parseNDArray());
    net["encoder.mlp2.mlp0.bn_var"] = Tensor(128).set(json["encoder.mlp2.mlp0.bn_var"].parseNDArray());
    net["encoder.mlp2.mlp0.bn_gamma"] = Tensor(128).set(json["encoder.mlp2.mlp0.bn_gamma"].parseNDArray());
    net["encoder.mlp2.mlp0.bn_beta"] = Tensor(128).set(json["encoder.mlp2.mlp0.bn_beta"].parseNDArray());
    
    net["encoder.mlp2.mlp1.weight"] = Tensor(128, 1024).set(json["encoder.mlp2.mlp1.weight"].parseNDArray());
    net["encoder.mlp2.mlp1.bias"] = Tensor(1024).set(json["encoder.mlp2.mlp1.bias"].parseNDArray());
    net["encoder.mlp2.mlp1.bn_mean"] = Tensor(1024).set(json["encoder.mlp2.mlp1.bn_mean"].parseNDArray());
    net["encoder.mlp2.mlp1.bn_var"] = Tensor(1024).set(json["encoder.mlp2.mlp1.bn_var"].parseNDArray());
    net["encoder.mlp2.mlp1.bn_gamma"] = Tensor(1024).set(json["encoder.mlp2.mlp1.bn_gamma"].parseNDArray());
    net["encoder.mlp2.mlp1.bn_beta"] = Tensor(1024).set(json["encoder.mlp2.mlp1.bn_beta"].parseNDArray());
    
    // SegHead weights
    for (int i = 0; i < 3; ++i) {
        std::string key = "segHead.mlp" + std::to_string(i);
        uint32_t in_dim = i == 0 ? 2048 : (i == 1 ? 512 : 256);
        uint32_t out_dim = i == 0 ? 512 : (i == 1 ? 256 : numClasses);
        
        net[key + ".weight"] = Tensor(in_dim, out_dim).set(json[key + ".weight"].parseNDArray());
        net[key + ".bias"] = Tensor(out_dim).set(json[key + ".bias"].parseNDArray());
        net[key + ".bn_mean"] = Tensor(out_dim).set(json[key + ".bn_mean"].parseNDArray());
        net[key + ".bn_var"] = Tensor(out_dim).set(json[key + ".bn_var"].parseNDArray());
        net[key + ".bn_gamma"] = Tensor(out_dim).set(json[key + ".bn_gamma"].parseNDArray());
        net[key + ".bn_beta"] = Tensor(out_dim).set(json[key + ".bn_beta"].parseNDArray());
    }
    
    std::cout << "\nPreparing network...\n";
    Tensor inputTensor = Tensor(N, 3).set(input_data);
    net.prepare();
    
    std::cout << "Running inference...\n";
    auto outputs = net(inputTensor);
    
    std::cout << "Downloading result from GPU...\n";
    Buffer outBuf = netGlobalDevice.createBuffer({
        .size = N * numClasses * sizeof(float),
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    });
    
    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuf, outputs[0].buffer())
        .end()
        .submit()
        .wait();
    
    float* output = (float*)outBuf.map();
    
    // Compare results
    std::cout << "\nComparing results...\n";
    bool all_pass = true;
    float max_error = 0.0f;
    const float TOLERANCE = 0.3f;  // TODO: Improve accuracy - currently using loose tolerance
    
    int error_count = 0;
    const int MAX_ERRORS_TO_SHOW = 10;
    
    for (uint32_t i = 0; i < N * numClasses; ++i) {
        float error = std::abs(output[i] - expected[i]);
        max_error = std::max(max_error, error);
        
        if (error > TOLERANCE) {
            if (error_count < MAX_ERRORS_TO_SHOW) {
                std::cout << "  ✗ Index " << i << " (point " << (i / numClasses) 
                          << ", class " << (i % numClasses) << "): "
                          << "got " << output[i] << ", "
                          << "expected " << expected[i] << ", "
                          << "error " << error << "\n";
            }
            error_count++;
            all_pass = false;
        }
    }
    
    std::cout << "\nMax error: " << max_error << "\n";
    std::cout << "Total errors: " << error_count << " / " << (N * numClasses) << "\n";
    
    std::cout << "\nFirst point predictions:\n";
    std::cout << "  Expected: [";
    for (uint32_t i = 0; i < numClasses; ++i)
        std::cout << (i > 0 ? ", " : "") << expected[i];
    std::cout << "]\n  Got:      [";
    for (uint32_t i = 0; i < numClasses; ++i)
        std::cout << (i > 0 ? ", " : "") << output[i];
    std::cout << "]\n";
    
    outBuf.unmap();
    
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
        test_segment();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

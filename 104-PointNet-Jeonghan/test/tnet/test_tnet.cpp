/**
 * TNetBlock Vulkan Test (WeightLoader + PyTorch keys)
 * Tests TNetBlock (Spatial Transformer) against PyTorch reference
 *
 * TNetBlock architecture (yanx27 STN3d):
 * - Input: [K, N] point cloud (PyTorch [C, N] format)
 * - MLP: conv1→bn1→relu, conv2→bn2→relu, conv3→bn3→relu
 * - MaxPool: [1024, N] → [1024]
 * - FC: fc1→bn4→relu, fc2→bn5→relu, fc3
 * - Reshape + Identity: [K*K] → [K, K] + I
 * - Output: [K, K] transformation matrix
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <utility>
#include "neuralNet.h"
#include "vulkanApp.h"
#include "safeTensorsParser.h"
#include "../networks/include/pointnet.hpp"
#include "../networks/include/weightLoader.hpp"

using namespace vk;
using namespace networks;

/**
 * Test network wrapper for TNetBlock
 *
 * TNetBlock generates transformation matrix [K, K]
 * MatMul applies the transformation to points
 *
 * Outputs:
 * - out0: Transformed points [K, N]
 * - out1: Transform matrix [K, K]
 */
class TNetTestNet : public NeuralNet {
public:
    TNetBlock tnet;
    MatMulNode matmul;

    TNetTestNet(Device& device, uint32_t K)
    : NeuralNet(device, 1, 2)  // 1 input, 2 outputs
    , tnet(K)
    , matmul()
    {
        // MatMul computes A.T @ B where A=[K,K] (transform), B=[K,N] (points)
        // This implements PyTorch: (points.T @ transform).T = transform.T @ points
        input(0) - tnet;  // Generate transformation matrix [K, K]

        tnet.slot("out0") - matmul.slot("in0");      // Transform [K,K] → MatMul.in0
        input(0).slot("out0") - matmul.slot("in1");  // Points [K,N] → MatMul.in1

        // Connect outputs
        matmul - output(0);              // Transformed points [K, N]
        tnet / "out0" - output(1);       // Transform matrix [K, K]
    }

    TNetBlock& getTNet() { return tnet; }
};

void test() {
    void loadShaders();
    loadShaders();

    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║      TNetBlock Vulkan Test              ║\n";
    std::cout << "║      (WeightLoader + PyTorch keys)      ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n\n";

    // Load reference data
    SafeTensorsParser data(PROJECT_CURRENT_DIR"/test/tnet/reference.safetensors");

    // Parse configuration - shape is [K, N]
    std::vector<float> shape = data["shape"].parseNDArray();
    uint32_t K = static_cast<uint32_t>(shape[0]);  // K = 3 (channels)
    uint32_t N = static_cast<uint32_t>(shape[1]);  // N = 8 (points)

    std::cout << "Test configuration:\n";
    std::cout << "  Input: [K=" << K << ", N=" << N << "]\n";
    std::cout << "  Output: [K=" << K << ", N=" << N << "]\n";
    std::cout << "  Transform: [K=" << K << ", K=" << K << "]\n\n";

    // Parse input and expected output
    std::vector<float> input_data = data["input"].parseNDArray();
    std::vector<float> expected = data["output"].parseNDArray();
    std::vector<float> expected_transform = data["transform"].parseNDArray();

    // Create network
    std::cout << "Creating TNetTestNet...\n";
    TNetTestNet net(netGlobalDevice, K);

    // Load weights using WeightLoader
    std::cout << "\nLoading weights with WeightLoader...\n";
    WeightLoader loader(data);
    loader.loadTNet(net.getTNet(), "");  // No prefix for test reference

    std::cout << "\nPreparing network...\n";
    net.prepare();

    // Create input tensor [K, N]
    Tensor input_tensor = Tensor(K, N).set(input_data);

    std::cout << "Running inference...\n";
    auto result = net(input_tensor);

    // Download transform matrix from GPU
    Buffer transformBuf = netGlobalDevice.createBuffer({
        .size = K * K * sizeof(float),
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    });

    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(transformBuf, result[1].buffer())
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

    transformBuf.unmap();

    // Download result from GPU
    Buffer outBuf = netGlobalDevice.createBuffer({
        .size = K * N * sizeof(float),
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

    // Compare results
    std::cout << "\n============================================================\n";
    std::cout << "Output Comparison:\n";
    std::cout << "============================================================\n\n";

    std::cout << "Index | Expected    | Got         | Diff        | Status\n";
    std::cout << "------------------------------------------------------------\n";

    const float tolerance = 0.001f;
    uint32_t mismatches = 0;
    float max_diff = 0.0f;
    float sum_diff = 0.0f;

    for (uint32_t i = 0; i < K * N; ++i) {
        float diff = std::abs(output[i] - expected[i]);
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;

        if (diff > tolerance)
            mismatches++;

        if (i < 5 || i >= K * N - 5) {
            std::cout << std::setw(5) << i << " | ";
            std::cout << std::setw(11) << std::fixed << std::setprecision(6) << expected[i] << " | ";
            std::cout << std::setw(11) << std::fixed << std::setprecision(6) << output[i] << " | ";
            std::cout << std::setw(11) << std::fixed << std::setprecision(6) << diff << " | ";
            std::cout << (diff <= tolerance ? "✓" : "✗") << "\n";
        } else if (i == 5) {
            std::cout << "  ... (" << (K * N - 10) << " more) ...\n";
        }
    }

    outBuf.unmap();

    std::cout << "\n============================================================\n";
    std::cout << "Results Summary:\n";
    std::cout << "============================================================\n";
    std::cout << "Total values:   " << K * N << "\n";
    std::cout << "Mismatches:     " << mismatches << "\n";
    std::cout << "Max difference: " << std::fixed << std::setprecision(6) << max_diff << "\n";
    std::cout << "Avg difference: " << std::fixed << std::setprecision(6) << (sum_diff / (K * N)) << "\n";
    std::cout << "Tolerance:      " << std::fixed << std::setprecision(6) << tolerance << "\n\n";

    if (mismatches == 0) {
        std::cout << "✅ TEST PASSED! All values match within tolerance.\n";
    } else {
        std::cout << "❌ TEST FAILED! " << mismatches << " values exceed tolerance.\n";
        exit(1);
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

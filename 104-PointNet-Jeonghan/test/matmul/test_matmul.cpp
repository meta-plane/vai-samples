/**
 * MatMul Vulkan Test (JSON-based)
 * Tests MatMulNode against PyTorch reference
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include "neuralNet.h"
#include "neuralNodes.h"
#include "vulkanApp.h"
#include "jsonParser.h"

using namespace vk;

/**
 * Test network with MatMul
 */
class MatMulTestNet : public NeuralNet {
    MatMulNode matmul;
    
public:
    MatMulTestNet(Device& device)
    : NeuralNet(device, 2, 1)  // 2 inputs (A, B), 1 output (C)
    , matmul()
    {
        // Connect inputs to matmul
        // input(0) provides matrix A -> matmul's in0
        // input(1) provides matrix B -> matmul's in1
        input(0) - "in0" / matmul;
        input(1) - "in1" / matmul;

        // Connect matmul output to network output
        matmul - output(0);
    }
};

/**
 * Run MatMul inference
 */
Tensor eval_matmul(uint32_t N, uint32_t K, uint32_t M,
                   const std::vector<float>& A_data,
                   const std::vector<float>& B_data) {
    // Create network
    MatMulTestNet net(netGlobalDevice);
    
    // Create input tensors
    Tensor A = Tensor(N, K).set(A_data);
    Tensor B = Tensor(K, M).set(B_data);
    
    // Run inference - prepare() is called inside run() for each node
    auto result = net(A, B);
    
    return result[0];
}

void test() {
    void loadShaders();
    loadShaders();
    
    // Load reference data
    JsonParser json = JsonParser(PROJECT_CURRENT_DIR"/test/matmul/reference.json");
    
    // Parse dimensions
    std::vector<float> shape = json["shape"].parseNDArray();
    
    uint32_t N = static_cast<uint32_t>(shape[0]);
    uint32_t K = static_cast<uint32_t>(shape[1]);
    uint32_t M = static_cast<uint32_t>(shape[2]);
    
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║      MatMul Vulkan Compute Test         ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n\n";
    
    std::cout << "Test configuration:\n";
    std::cout << "  A: [" << N << ", " << K << "]\n";
    std::cout << "  B: [" << K << ", " << M << "]\n";
    std::cout << "  C = A @ B: [" << N << ", " << M << "]\n\n";
    
    // Get input data
    std::vector<float> A_data = json["A"].parseNDArray();
    std::vector<float> B_data = json["B"].parseNDArray();
    
    // Run inference
    std::cout << "Running MatMul on GPU...\n";
    Tensor result = eval_matmul(N, K, M, A_data, B_data);
    
    // Download result from GPU to CPU
    uint32_t totalElements = N * M;
    
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
    
    // Get expected values
    std::vector<float> expected = json["C"].parseNDArray();
    
    // Compare results
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
        
        // Show first 5, last 5, and all mismatches
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

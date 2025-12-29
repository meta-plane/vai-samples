/**
 * PointNetSegment Test - PyTorch Reference Comparison
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "neuralNet.h"
#include "vulkanApp.h"
#include "safeTensorsParser.h"
#include "../networks/include/pointnet.hpp"
#include "../networks/include/weightLoader.hpp"

using namespace vk;
using namespace networks;

void test_segment() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║     PointNetSegment - PyTorch Reference Test          ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n\n";

    SafeTensorsParser data(PROJECT_CURRENT_DIR"/test/segment/reference.safetensors");

    auto input_shape = data["input"].getShape();
    uint32_t channel = input_shape[0];
    uint32_t N = input_shape[1];
    uint32_t numClasses = 13;

    std::cout << "Configuration:\n";
    std::cout << "  Input:   [" << channel << ", " << N << "] (9-dim: xyz+rgb+normalized)\n";
    std::cout << "  Output:  [" << numClasses << ", " << N << "] (13 semantic classes)\n\n";

    std::vector<float> input_data = data["input"].parseNDArray();
    std::vector<float> expected_output = data["expected_output"].parseNDArray();

    PointNetSegment net(netGlobalDevice, numClasses, channel);
    WeightLoader loader(data, false);
    loader.loadSegment(net);
    net.prepare();

    Tensor inputTensor = Tensor(channel, N).set(input_data);
    auto outputs = net(inputTensor);

    Buffer outBuf = netGlobalDevice.createBuffer({
        .size = numClasses * N * sizeof(float),
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

    float max_diff = 0.0f;
    float total_diff = 0.0f;
    int mismatch_count = 0;
    const float tolerance = 1e-3f;

    for (uint32_t c = 0; c < numClasses; c++) {
        for (uint32_t n = 0; n < N; n++) {
            float vulkan_val = output[c * N + n];
            float pytorch_val = expected_output[c * N + n];
            float diff = std::abs(vulkan_val - pytorch_val);

            max_diff = std::max(max_diff, diff);
            total_diff += diff;

            if (diff > tolerance) {
                mismatch_count++;
            }
        }
    }

    float avg_diff = total_diff / (numClasses * N);

    std::cout << "Comparison (first 5 classes, point 0):\n";
    std::cout << "  ┌─────────┬──────────────┬──────────────┬──────────────┐\n";
    std::cout << "  │ Class   │ Vulkan       │ PyTorch      │ Diff         │\n";
    std::cout << "  ├─────────┼──────────────┼──────────────┼──────────────┤\n";

    for (uint32_t c = 0; c < 5; c++) {
        float vulkan_val = output[c * N + 0];
        float pytorch_val = expected_output[c * N + 0];
        float diff = std::abs(vulkan_val - pytorch_val);
        std::cout << "  │ " << std::setw(7) << c << " │ "
                  << std::setw(12) << std::fixed << std::setprecision(6) << vulkan_val << " │ "
                  << std::setw(12) << pytorch_val << " │ "
                  << std::setw(12) << std::scientific << std::setprecision(2) << diff << " │\n";
    }
    std::cout << "  └─────────┴──────────────┴──────────────┴──────────────┘\n\n";

    outBuf.unmap();

    std::cout << "Results:\n";
    std::cout << "  Max difference: " << std::scientific << std::setprecision(2) << max_diff << "\n";
    std::cout << "  Avg difference: " << avg_diff << "\n";
    std::cout << "  Mismatches:     " << mismatch_count << "/" << (numClasses * N) << "\n\n";

    if (mismatch_count == 0) {
        std::cout << "Status: PASSED (all outputs within tolerance " << std::fixed << tolerance << ")\n";
    } else {
        std::cout << "Status: FAILED (" << mismatch_count << " values exceed tolerance)\n";
    }
    std::cout << "\n";
}

int main() {
    loadShaders();
    test_segment();
    return 0;
}

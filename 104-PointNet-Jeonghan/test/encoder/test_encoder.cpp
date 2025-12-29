/**
 * PointNetEncoder Test (PyTorch Convention + WeightLoader)
 *
 * Uses the complete PointNetEncoder from pointnet.hpp
 * with simplified weight loading via WeightLoader class.
 */

#include "../library/neuralNet.h"
#include "../library/neuralNodes.h"
#include "../library/safeTensorsParser.h"
#include "../networks/include/pointnet.hpp"
#include "../networks/include/weightLoader.hpp"
#include <iostream>
#include <vector>
#include <cmath>

using namespace vk;
using namespace networks;

/**
 * Encoder Test Network - wraps PointNetEncoder with full output pipeline
 *
 * PointNetEncoder produces two outputs:
 *   out0: pointfeat [64, N] (from matmul2, after FSTN)
 *   out1: full features [1024, N] (from conv3)
 *
 * To match PyTorch output [1088, N], we need to:
 *   1. MaxPool out1 → [1024]
 *   2. Reshape → [1024, 1]
 *   3. Broadcast → [1024, N]
 *   4. Concat [global(1024), pointfeat(64)] → [1088, N]
 */
class EncoderTestNet : public NeuralNet {
public:
    PointNetEncoder encoder;
    MaxPooling1DNode maxpool;
    ReShapeNode reshape_global;
    BroadcastNode broadcast;
    ConcatNode concat;

    EncoderTestNet(Device& device, uint32_t channel)
    : NeuralNet(device, 1, 1),
      encoder(channel),
      maxpool(),
      reshape_global({1024, 1}),
      broadcast(),
      concat()
    {
        // Input → Encoder
        input(0) - encoder;

        // PyTorch output order: [global(1024), pointfeat(64)] = [1088, N]
        // Path A: full features [1024, N] → maxpool → reshape → broadcast → concat.in0
        encoder / "out1" - maxpool;                    // [1024, N] → [1024]
        maxpool - reshape_global;                      // [1024] → [1024, 1]
        reshape_global - broadcast;                    // [1024, 1] to broadcast.in0
        encoder / "out1" - "in1" / broadcast;          // [1024, N] shape reference for broadcast
        broadcast - "in0" / concat;                    // Broadcasted [1024, N] to concat.in0

        // Path B: pointfeat [64, N] → concat.in1
        encoder / "out0" - "in1" / concat;             // pointfeat [64, N] to concat.in1

        // Output concatenated result [1088, N] = [global(1024) + pointfeat(64), N]
        concat - output(0);
    }

    Tensor& operator[](const std::string& name) {
        return encoder[name];
    }
};

/**
 * Run encoder inference with WeightLoader
 */
Tensor eval_encoder(uint32_t N, uint32_t channel, const std::vector<float>& input_data, SafeTensorsParser& weights) {
    std::cout << "Creating EncoderTestNet...\n";
    EncoderTestNet net(netGlobalDevice, channel);

    std::cout << "\nLoading weights with WeightLoader...\n";
    WeightLoader loader(weights);
    loader.loadEncoder(net.encoder, "");  // No prefix for test reference

    std::cout << "\nPreparing network...\n";
    net.prepare();

    std::cout << "Running inference...\n";
    Tensor inputTensor = Tensor(channel, N).set(input_data);
    auto result = net(inputTensor);

    return result[0];
}

void test() {
    void loadShaders();
    loadShaders();

    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║   PointNetEncoder Vulkan Test           ║\n";
    std::cout << "║   (WeightLoader + PyTorch Convention)   ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n\n";

    // Load reference data
    SafeTensorsParser weights(PROJECT_CURRENT_DIR"/test/encoder/reference.safetensors");

    auto input_shape = weights["input"].getShape();
    uint32_t channel = input_shape[0];
    uint32_t N = input_shape[1];

    std::cout << "Test configuration:\n";
    std::cout << "  Points: N = " << N << "\n";
    std::cout << "  Channels: " << channel << "\n\n";

    std::vector<float> inputData = weights["input"].parseNDArray();
    std::vector<float> expectedOutput = weights["expected_output"].parseNDArray();

    auto output_shape = weights["expected_output"].getShape();
    uint32_t out_channels = output_shape[0];

    std::cout << "Expected output shape: [" << out_channels << ", " << N << "]\n\n";

    // Run inference
    Tensor result = eval_encoder(N, channel, inputData, weights);

    // Download result
    std::cout << "\nDownloading results from GPU...\n";
    Buffer outBuf = netGlobalDevice.createBuffer({
        .size = out_channels * N * sizeof(float),
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

    // Debug: Print first few values
    std::cout << "\nDebug - First 5 global values (channel 0-4, point 0):\n";
    for (int c = 0; c < 5; ++c) {
        std::cout << "  [" << c << ",0] expected=" << expectedOutput[c * N + 0]
                  << " got=" << output[c * N + 0] << "\n";
    }
    std::cout << "\nDebug - First 5 pointfeat values (channel 1024-1028, point 0):\n";
    for (int c = 1024; c < 1029; ++c) {
        std::cout << "  [" << c << ",0] expected=" << expectedOutput[c * N + 0]
                  << " got=" << output[c * N + 0] << "\n";
    }

    // Compare results
    std::cout << "\nComparing results...\n";
    float max_diff = 0.0f;
    int mismatch_count = 0;
    const float tolerance = 1e-3f;

    for (uint32_t i = 0; i < out_channels * N; ++i) {
        float diff = std::abs(output[i] - expectedOutput[i]);
        max_diff = std::max(max_diff, diff);
        if (diff > tolerance) {
            mismatch_count++;
            if (mismatch_count <= 5) {
                std::cout << "  Mismatch at [" << i << "]: expected " << expectedOutput[i]
                         << ", got " << output[i] << " (diff: " << diff << ")\n";
            }
        }
    }

    outBuf.unmap();

    std::cout << "\n" << std::string(46, '=') << "\n";
    std::cout << "Test Results\n";
    std::cout << std::string(46, '=') << "\n";
    std::cout << "Total elements: " << (out_channels * N) << "\n";
    std::cout << "Max difference: " << max_diff << "\n";
    std::cout << "Mismatches (>" << tolerance << "): " << mismatch_count << "\n";

    if (mismatch_count == 0) {
        std::cout << "\n✅ TEST PASSED!\n";
    } else {
        std::cout << "\n❌ TEST FAILED!\n";
        exit(1);
    }

    std::cout << std::string(46, '=') << "\n";
}

int main() {
    test();
    return 0;
}

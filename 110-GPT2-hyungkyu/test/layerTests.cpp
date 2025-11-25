/**
 * Unit Tests for Basic Transformer Layers
 * Tests: LayerNorm, GELU, Add, Identity, FeedForward
 */

#include "testFramework.h"
#include "testHelpers.h"
#include "../model/transformerBlock/transformer.h"
#include "../core/neuralNet.h"
#include <cmath>

using namespace vk;

// ============================================================================
// LayerNorm Tests
// ============================================================================

class LayerNormBasicTest : public BaseTest {
public:
    LayerNormBasicTest() : BaseTest("LayerNorm - Basic Functionality") {}

    void run() override {
        // Create network
        NeuralNet net(netGlobalDevice, 1, 1);
        LayerNormNode layer(768, 1e-5f);
        net.input(0) - layer - net.output(0);

        // Create input with known distribution
        std::vector<float> input_data(2 * 4 * 768);
        for (size_t i = 0; i < input_data.size(); ++i) {
            input_data[i] = (float)(i % 100) / 50.0f - 1.0f;  // Range [-1, 1]
        }
        Tensor input = Tensor(2, 4, 768).set(input_data);

        // Run inference
        Tensor output = net(input)[0];

        // Verify output shape
        TestAssert::assertShape(output, {2, 4, 768}, "LayerNorm output shape");

        // Verify normalization properties
        std::vector<float> output_data = readTensorData(output);

        // Check mean ~0 and std ~1 for each sequence position
        for (uint32_t b = 0; b < 2; ++b) {
            for (uint32_t s = 0; s < 4; ++s) {
                float sum = 0.0f;
                float sum_sq = 0.0f;

                for (uint32_t d = 0; d < 768; ++d) {
                    float val = output_data[b * 4 * 768 + s * 768 + d];
                    sum += val;
                    sum_sq += val * val;
                }

                float mean = sum / 768.0f;
                float variance = sum_sq / 768.0f - mean * mean;
                float std = std::sqrt(variance);

                TestAssert::assertClose(mean, 0.0f, 1e-4f,
                    "LayerNorm mean for batch=" + std::to_string(b) + " seq=" + std::to_string(s));
                TestAssert::assertClose(std, 1.0f, 1e-4f,
                    "LayerNorm std for batch=" + std::to_string(b) + " seq=" + std::to_string(s));
            }
        }

        std::cout << "  ✓ Output shape verified: [2, 4, 768]" << std::endl;
        std::cout << "  ✓ Normalization verified: mean ~0, std ~1" << std::endl;
    }
};

class LayerNormJSONTest : public BaseTest {
public:
    LayerNormJSONTest() : BaseTest("LayerNorm - PyTorch Reference") {}

    void run() override {
        json test_data = loadTestData(std::string(PROJECT_CURRENT_DIR) + "/assets/test_data/layer_norm_test_data.json");

        uint32_t B = test_data["config"]["batch_size"];
        uint32_t S = test_data["config"]["seq_len"];
        uint32_t D = test_data["config"]["d_model"];
        float eps = test_data["config"]["eps"];

        // Create network
        NeuralNet net(netGlobalDevice, 1, 1);
        LayerNormNode layer(D, eps);
        net.input(0) - layer - net.output(0);

        // Load parameters
        std::vector<float> scale_data = jsonToVector(test_data["scale"]);
        std::vector<float> shift_data = jsonToVector(test_data["shift"]);
        layer["scale"] = Tensor(D).set(scale_data);
        layer["shift"] = Tensor(D).set(shift_data);

        // Load input and run
        std::vector<float> input_data = jsonToVector(test_data["input"]);
        Tensor input = Tensor(B, S, D).set(input_data);
        Tensor output = net(input)[0];

        // Verify output matches PyTorch
        std::vector<float> expected_data = jsonToVector(test_data["output"]);
        Tensor expected = Tensor(B, S, D).set(expected_data);
        TestAssert::assertEqual(output, expected, 1e-4f, "LayerNorm output vs PyTorch");

        std::cout << "  ✓ Output matches PyTorch reference (tolerance: 1e-4)" << std::endl;
    }
};

// ============================================================================
// GELU Tests
// ============================================================================

class GELUBasicTest : public BaseTest {
    std::unique_ptr<GELUNode> layer;

public:
    GELUBasicTest() : BaseTest("GELU - Basic Functionality") {}

    void run() override {
        layer = std::make_unique<GELUNode>();

        // Test with known input values
        Tensor input(2, 3, 4);
        std::vector<float> input_data = {
            -2.0f, -1.0f, 0.0f, 1.0f,
            -0.5f, 0.5f, -1.5f, 1.5f,
            -3.0f, 3.0f, -0.1f, 0.1f,

            -2.0f, -1.0f, 0.0f, 1.0f,
            -0.5f, 0.5f, -1.5f, 1.5f,
            -3.0f, 3.0f, -0.1f, 0.1f
        };
        input.set(input_data);

        (*layer)["in0"] = input;
        layer->prepare();

        auto cmd = netGlobalDevice.newCommandBuffer(queue_compute);
        cmd.begin();
        layer->run(cmd);
        cmd.end();
        cmd.submit();
        cmd.wait();

        Tensor& output = (*layer)["out0"];

        // Verify shape
        TestAssert::assertShape(output, {2, 3, 4}, "GELU output shape");

        // Verify GELU properties: GELU(0) ≈ 0, GELU(x) ≈ x for large positive x
        std::vector<float> output_data = readTensorData(output);

        // GELU(0) should be close to 0
        TestAssert::assertClose(output_data[2], 0.0f, 1e-5f, "GELU(0) ≈ 0");
        TestAssert::assertClose(output_data[14], 0.0f, 1e-5f, "GELU(0) ≈ 0");

        // GELU(x) ≈ x for large positive x
        TestAssert::assertClose(output_data[9], 3.0f, 0.01f, "GELU(3) ≈ 3");

        std::cout << "  ✓ Output shape verified: [2, 3, 4]" << std::endl;
        std::cout << "  ✓ GELU properties verified" << std::endl;
    }
};

class GELUJSONTest : public BaseTest {
    std::unique_ptr<GELUNode> layer;
    json test_data;

public:
    GELUJSONTest() : BaseTest("GELU - PyTorch Reference") {}

    void setup() override {
        test_data = loadTestData("../assets/test_data/gelu_test_data.json");
        layer = std::make_unique<GELUNode>();
    }

    void run() override {
        uint32_t B = test_data["config"]["batch_size"];
        uint32_t S = test_data["config"]["seq_len"];
        uint32_t D = test_data["config"]["d_model"];

        std::vector<float> input_data = jsonToVector(test_data["input"]);
        Tensor input(B, S, D);
        input.set(input_data);

        (*layer)["in0"] = input;
        layer->prepare();

        auto cmd = netGlobalDevice.newCommandBuffer(queue_compute);
        cmd.begin();
        layer->run(cmd);
        cmd.end();
        cmd.submit();
        cmd.wait();

        Tensor& output = (*layer)["out0"];
        std::vector<float> expected_data = jsonToVector(test_data["output"]);
        Tensor expected(B, S, D);
        expected.set(expected_data);

        TestAssert::assertEqual(output, expected, 1e-4f, "GELU output vs PyTorch");

        std::cout << "  ✓ Output matches PyTorch reference (tolerance: 1e-4)" << std::endl;
    }
};

// ============================================================================
// Add Node Tests
// ============================================================================

class AddNodeTest : public BaseTest {
    std::unique_ptr<AddNode> layer;

public:
    AddNodeTest() : BaseTest("AddNode - Element-wise Addition") {}

    void run() override {
        layer = std::make_unique<AddNode>();

        // Create two input tensors
        Tensor input0(2, 3, 4);
        Tensor input1(2, 3, 4);

        std::vector<float> data0(2 * 3 * 4);
        std::vector<float> data1(2 * 3 * 4);

        for (size_t i = 0; i < data0.size(); ++i) {
            data0[i] = (float)i * 0.1f;
            data1[i] = (float)i * 0.2f;
        }

        input0.set(data0);
        input1.set(data1);

        (*layer)["in0"] = input0;
        (*layer)["in1"] = input1;
        layer->prepare();

        auto cmd = netGlobalDevice.newCommandBuffer(queue_compute);
        cmd.begin();
        layer->run(cmd);
        cmd.end();
        cmd.submit();
        cmd.wait();

        Tensor& output = (*layer)["out0"];

        // Verify shape
        TestAssert::assertShape(output, {2, 3, 4}, "AddNode output shape");

        // Verify values (should be element-wise sum)
        std::vector<float> output_data = readTensorData(output);
        for (size_t i = 0; i < output_data.size(); ++i) {
            float expected = data0[i] + data1[i];
            TestAssert::assertClose(output_data[i], expected, 1e-5f,
                "AddNode element " + std::to_string(i));
        }

        std::cout << "  ✓ Output shape verified: [2, 3, 4]" << std::endl;
        std::cout << "  ✓ Element-wise addition verified" << std::endl;
    }
};

// ============================================================================
// Identity Node Tests
// ============================================================================

class IdentityNodeTest : public BaseTest {
    std::unique_ptr<IdentityNode> layer;

public:
    IdentityNodeTest() : BaseTest("IdentityNode - Pass-through") {}

    void run() override {
        layer = std::make_unique<IdentityNode>();

        Tensor input(2, 3, 4);
        std::vector<float> input_data(2 * 3 * 4);
        for (size_t i = 0; i < input_data.size(); ++i) {
            input_data[i] = (float)i * 0.5f;
        }
        input.set(input_data);

        (*layer)["in0"] = input;
        layer->prepare();

        auto cmd = netGlobalDevice.newCommandBuffer(queue_compute);
        cmd.begin();
        layer->run(cmd);
        cmd.end();
        cmd.submit();
        cmd.wait();

        Tensor& output = (*layer)["out0"];

        // Verify shape
        TestAssert::assertShape(output, {2, 3, 4}, "IdentityNode output shape");

        // Verify output equals input (identity operation)
        TestAssert::assertEqual(output, input, 1e-7f, "IdentityNode output == input");

        std::cout << "  ✓ Output shape verified: [2, 3, 4]" << std::endl;
        std::cout << "  ✓ Identity operation verified (output == input)" << std::endl;
    }
};

// ============================================================================
// FeedForward Tests
// ============================================================================

class FeedForwardBasicTest : public BaseTest {
    std::unique_ptr<FeedForwardNode> layer;

public:
    FeedForwardBasicTest() : BaseTest("FeedForward - Basic Functionality") {}

    void run() override {
        uint32_t d_model = 64;
        layer = std::make_unique<FeedForwardNode>(d_model);

        Tensor input(2, 4, d_model);
        std::vector<float> input_data(2 * 4 * d_model, 0.1f);
        input.set(input_data);

        // Initialize weights
        std::vector<float> w1_data(4 * d_model * d_model, 0.01f);
        std::vector<float> w2_data(d_model * 4 * d_model, 0.01f);
        std::vector<float> b1_data(4 * d_model, 0.0f);
        std::vector<float> b2_data(d_model, 0.0f);

        (*layer)["in0"] = input;
        (*layer)["weight1"] = Tensor(4 * d_model, d_model).set(w1_data).setConstant();
        (*layer)["weight2"] = Tensor(d_model, 4 * d_model).set(w2_data).setConstant();
        (*layer)["bias1"] = Tensor(4 * d_model).set(b1_data).setConstant();
        (*layer)["bias2"] = Tensor(d_model).set(b2_data).setConstant();

        layer->prepare();

        auto cmd = netGlobalDevice.newCommandBuffer(queue_compute);
        cmd.begin();
        layer->run(cmd);
        cmd.end();
        cmd.submit();
        cmd.wait();

        Tensor& output = (*layer)["out0"];

        // Verify shape: input [2, 4, 64] -> output [2, 4, 64]
        TestAssert::assertShape(output, {2, 4, d_model}, "FeedForward output shape");

        std::cout << "  ✓ Output shape verified: [2, 4, " << d_model << "]" << std::endl;
        std::cout << "  ✓ FeedForward pipeline executed successfully" << std::endl;
    }
};

class FeedForwardJSONTest : public BaseTest {
    std::unique_ptr<FeedForwardNode> layer;
    json test_data;

public:
    FeedForwardJSONTest() : BaseTest("FeedForward - PyTorch Reference") {}

    void setup() override {
        test_data = loadTestData("../assets/test_data/feedforward_test_data.json");
        uint32_t d_model = test_data["config"]["d_model"];
        layer = std::make_unique<FeedForwardNode>(d_model);
    }

    void run() override {
        uint32_t B = test_data["config"]["batch_size"];
        uint32_t S = test_data["config"]["seq_len"];
        uint32_t D = test_data["config"]["d_model"];
        uint32_t H = test_data["config"]["hidden_dim"];

        // Load input and weights
        std::vector<float> input_data = jsonToVector(test_data["input"]);
        std::vector<float> w1_data = jsonToVector(test_data["weight1"]);
        std::vector<float> w2_data = jsonToVector(test_data["weight2"]);

        Tensor input(B, S, D);
        input.set(input_data);

        (*layer)["in0"] = input;
        (*layer)["weight1"] = Tensor(H, D).set(w1_data).setConstant();
        (*layer)["weight2"] = Tensor(D, H).set(w2_data).setConstant();

        // Initialize biases to zero (PyTorch test data doesn't include bias)
        std::vector<float> b1_data(H, 0.0f);
        std::vector<float> b2_data(D, 0.0f);
        (*layer)["bias1"] = Tensor(H).set(b1_data).setConstant();
        (*layer)["bias2"] = Tensor(D).set(b2_data).setConstant();

        layer->prepare();

        auto cmd = netGlobalDevice.newCommandBuffer(queue_compute);
        cmd.begin();
        layer->run(cmd);
        cmd.end();
        cmd.submit();
        cmd.wait();

        Tensor& output = (*layer)["out0"];
        std::vector<float> expected_data = jsonToVector(test_data["output"]);
        Tensor expected(B, S, D);
        expected.set(expected_data);

        TestAssert::assertEqual(output, expected, 1e-3f, "FeedForward output vs PyTorch");

        std::cout << "  ✓ Output matches PyTorch reference (tolerance: 1e-3)" << std::endl;
    }
};

// ============================================================================
// Test Suite
// ============================================================================

void runLayerTests() {
    TestSuite suite("Basic Transformer Layers");

    // LayerNorm tests
    suite.addTest([]() { LayerNormBasicTest test; return test.execute(); });
    suite.addTest([]() { LayerNormJSONTest test; return test.execute(); });

    // GELU tests
    suite.addTest([]() { GELUBasicTest test; return test.execute(); });
    suite.addTest([]() { GELUJSONTest test; return test.execute(); });

    // Add/Identity tests
    suite.addTest([]() { AddNodeTest test; return test.execute(); });
    suite.addTest([]() { IdentityNodeTest test; return test.execute(); });

    // FeedForward tests
    suite.addTest([]() { FeedForwardBasicTest test; return test.execute(); });
    suite.addTest([]() { FeedForwardJSONTest test; return test.execute(); });

    suite.runAll();
}

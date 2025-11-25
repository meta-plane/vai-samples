/**
 * Unit Tests for Attention Components
 * Tests: LinearNode, SoftmaxNode, MultiHeadAttentionNode
 */

#include "testFramework.h"
#include "testHelpers.h"
#include "../model/attention/attentionNode.h"
#include "../model/cache/kvCache.h"

using namespace vk;

// ============================================================================
// Linear Node Tests
// ============================================================================

class LinearNodeTest : public BaseTest {
    std::unique_ptr<LinearNode> layer;

public:
    LinearNodeTest() : BaseTest("LinearNode - Matrix Multiplication") {}

    void run() override {
        uint32_t in_features = 64;
        uint32_t out_features = 128;
        layer = std::make_unique<LinearNode>(in_features, out_features);

        Tensor input(2, 4, in_features);
        std::vector<float> input_data(2 * 4 * in_features, 0.1f);
        input.set(input_data);

        // Initialize weight
        std::vector<float> weight_data(out_features * in_features, 0.01f);

        (*layer)["in0"] = input;
        (*layer)["weight"] = Tensor(out_features, in_features).set(weight_data).setConstant();

        layer->prepare();

        auto cmd = netGlobalDevice.newCommandBuffer(queue_compute);
        cmd.begin();
        layer->run(cmd);
        cmd.end();
        cmd.submit();
        cmd.wait();

        Tensor& output = (*layer)["out0"];

        // Verify shape: [2, 4, 64] @ [128, 64]^T -> [2, 4, 128]
        TestAssert::assertShape(output, {2, 4, out_features}, "Linear output shape");

        std::cout << "  ✓ Output shape verified: [2, 4, " << out_features << "]" << std::endl;
    }
};

// ============================================================================
// Softmax Node Tests
// ============================================================================

class SoftmaxNodeTest : public BaseTest {
    std::unique_ptr<SoftmaxNode> layer;

public:
    SoftmaxNodeTest() : BaseTest("SoftmaxNode - Normalization") {}

    void run() override {
        layer = std::make_unique<SoftmaxNode>();

        Tensor input(2, 3, 4);
        std::vector<float> input_data = {
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,

            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f
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
        TestAssert::assertShape(output, {2, 3, 4}, "Softmax output shape");

        // Verify softmax properties: sum of each row should be 1.0
        std::vector<float> output_data = readTensorData(output);

        for (uint32_t b = 0; b < 2; ++b) {
            for (uint32_t s = 0; s < 3; ++s) {
                float sum = 0.0f;
                for (uint32_t d = 0; d < 4; ++d) {
                    float val = output_data[b * 3 * 4 + s * 4 + d];
                    sum += val;
                    TestAssert::assertTrue(val >= 0.0f && val <= 1.0f,
                        "Softmax output in range [0, 1]");
                }
                TestAssert::assertClose(sum, 1.0f, 1e-5f,
                    "Softmax sum = 1.0 for batch=" + std::to_string(b) + " seq=" + std::to_string(s));
            }
        }

        std::cout << "  ✓ Output shape verified: [2, 3, 4]" << std::endl;
        std::cout << "  ✓ Softmax properties verified (sum = 1.0, values in [0,1])" << std::endl;
    }
};

// ============================================================================
// MultiHeadAttention Tests
// ============================================================================

class AttentionBasicTest : public BaseTest {
    std::unique_ptr<MultiHeadAttentionNode> layer;

public:
    AttentionBasicTest() : BaseTest("MultiHeadAttention - Basic Functionality") {}

    void run() override {
        uint32_t d_in = 64;
        uint32_t d_out = 64;
        uint32_t num_heads = 4;

        layer = std::make_unique<MultiHeadAttentionNode>(d_in, d_out, num_heads);

        Tensor input(2, 8, d_in);
        std::vector<float> input_data(2 * 8 * d_in, 0.1f);
        input.set(input_data);

        // Initialize weights
        std::vector<float> wq_data(d_out * d_in, 0.01f);
        std::vector<float> wk_data(d_out * d_in, 0.01f);
        std::vector<float> wv_data(d_out * d_in, 0.01f);
        std::vector<float> wout_data(d_out * d_out, 0.01f);
        std::vector<float> bias_data(d_out, 0.0f);

        (*layer)["in0"] = input;
        (*layer)["W_query"] = Tensor(d_out, d_in).set(wq_data).setConstant();
        (*layer)["W_key"] = Tensor(d_out, d_in).set(wk_data).setConstant();
        (*layer)["W_value"] = Tensor(d_out, d_in).set(wv_data).setConstant();
        (*layer)["W_out"] = Tensor(d_out, d_out).set(wout_data).setConstant();
        (*layer)["B_query"] = Tensor(d_out).set(bias_data).setConstant();
        (*layer)["B_key"] = Tensor(d_out).set(bias_data).setConstant();
        (*layer)["B_value"] = Tensor(d_out).set(bias_data).setConstant();
        (*layer)["B_out"] = Tensor(d_out).set(bias_data).setConstant();

        layer->prepare();

        auto cmd = netGlobalDevice.newCommandBuffer(queue_compute);
        cmd.begin();
        layer->run(cmd);
        cmd.end();
        cmd.submit();
        cmd.wait();

        Tensor& output = (*layer)["out0"];

        // Verify shape
        TestAssert::assertShape(output, {2, 8, d_out}, "Attention output shape");

        std::cout << "  ✓ Output shape verified: [2, 8, " << d_out << "]" << std::endl;
    }
};

class AttentionJSONTest : public BaseTest {
    std::unique_ptr<MultiHeadAttentionNode> layer;
    json test_data;

public:
    AttentionJSONTest() : BaseTest("MultiHeadAttention - PyTorch Reference") {}

    void setup() override {
        test_data = loadTestData("../assets/test_data/mha_test_data.json");
        uint32_t d_in = test_data["config"]["d_in"];
        uint32_t d_out = test_data["config"]["d_out"];
        uint32_t num_heads = test_data["config"]["num_heads"];

        layer = std::make_unique<MultiHeadAttentionNode>(d_in, d_out, num_heads);
    }

    void run() override {
        uint32_t B = test_data["config"]["batch_size"];
        uint32_t S = test_data["config"]["seq_len"];
        uint32_t D_in = test_data["config"]["d_in"];
        uint32_t D_out = test_data["config"]["d_out"];

        // Load input and weights
        std::vector<float> input_data = jsonToVector(test_data["input"]);
        std::vector<float> wq_data = jsonToVector(test_data["weights"]["W_query"]);
        std::vector<float> wk_data = jsonToVector(test_data["weights"]["W_key"]);
        std::vector<float> wv_data = jsonToVector(test_data["weights"]["W_value"]);
        std::vector<float> wout_data = jsonToVector(test_data["weights"]["W_out"]);

        Tensor input(B, S, D_in);
        input.set(input_data);

        (*layer)["in0"] = input;
        (*layer)["W_query"] = Tensor(D_out, D_in).set(wq_data).setConstant();
        (*layer)["W_key"] = Tensor(D_out, D_in).set(wk_data).setConstant();
        (*layer)["W_value"] = Tensor(D_out, D_in).set(wv_data).setConstant();
        (*layer)["W_out"] = Tensor(D_out, D_out).set(wout_data).setConstant();

        // Initialize biases to zero (test data doesn't include bias)
        std::vector<float> bias_data(D_out, 0.0f);
        (*layer)["B_query"] = Tensor(D_out).set(bias_data).setConstant();
        (*layer)["B_key"] = Tensor(D_out).set(bias_data).setConstant();
        (*layer)["B_value"] = Tensor(D_out).set(bias_data).setConstant();
        (*layer)["B_out"] = Tensor(D_out).set(bias_data).setConstant();

        layer->prepare();

        auto cmd = netGlobalDevice.newCommandBuffer(queue_compute);
        cmd.begin();
        layer->run(cmd);
        cmd.end();
        cmd.submit();
        cmd.wait();

        Tensor& output = (*layer)["out0"];
        std::vector<float> expected_data = jsonToVector(test_data["output"]);
        Tensor expected(B, S, D_out);
        expected.set(expected_data);

        TestAssert::assertEqual(output, expected, 1e-3f, "Attention output vs PyTorch");

        std::cout << "  ✓ Output matches PyTorch reference (tolerance: 1e-3)" << std::endl;
    }
};

// ============================================================================
// KV Cache Unit Tests
// ============================================================================

class AttentionCacheTest : public BaseTest {
    std::unique_ptr<MultiHeadAttentionNode> layer;
    LayerKVCache cache;

public:
    AttentionCacheTest() : BaseTest("MultiHeadAttention - KV Cache") {}

    void setup() override {
        layer = std::make_unique<MultiHeadAttentionNode>(64, 64, 4);

        // Initialize cache
        cache.max_len = 128;
        cache.current_len = 0;
        cache.K = Tensor(1, 4, 128, 16);  // [B=1, H=4, max_len=128, head_dim=16]
        cache.V = Tensor(1, 4, 128, 16);
    }

    void run() override {
        // Initialize weights
        std::vector<float> w_data(64 * 64, 0.01f);
        std::vector<float> b_data(64, 0.0f);

        (*layer)["W_query"] = Tensor(64, 64).set(w_data).setConstant();
        (*layer)["W_key"] = Tensor(64, 64).set(w_data).setConstant();
        (*layer)["W_value"] = Tensor(64, 64).set(w_data).setConstant();
        (*layer)["W_out"] = Tensor(64, 64).set(w_data).setConstant();
        (*layer)["B_query"] = Tensor(64).set(b_data).setConstant();
        (*layer)["B_key"] = Tensor(64).set(b_data).setConstant();
        (*layer)["B_value"] = Tensor(64).set(b_data).setConstant();
        (*layer)["B_out"] = Tensor(64).set(b_data).setConstant();

        // Step 1: Process initial prompt (no cache)
        std::cout << "\n  Step 1: Initial prompt (cache disabled)" << std::endl;

        Tensor prompt_input(1, 5, 64);
        std::vector<float> prompt_data(5 * 64, 0.1f);
        prompt_input.set(prompt_data);

        (*layer)["in0"] = prompt_input;
        layer->disableCache();
        layer->prepare();

        auto cmd = netGlobalDevice.newCommandBuffer(queue_compute);
        cmd.begin();
        layer->run(cmd);
        cmd.end();
        cmd.submit();
        cmd.wait();

        Tensor& output_no_cache = (*layer)["out0"];
        TestAssert::assertShape(output_no_cache, {1, 5, 64}, "Prompt output shape");
        std::cout << "    ✓ Prompt processed: [1, 5, 64]" << std::endl;

        // Step 2: Enable cache and process new token
        std::cout << "\n  Step 2: New token (cache enabled)" << std::endl;

        layer->setCache(&cache);
        cache.current_len = 5;  // Simulate 5 cached tokens

        Tensor new_token(1, 1, 64);
        std::vector<float> token_data(64, 0.2f);
        new_token.set(token_data);

        (*layer)["in0"] = new_token;
        layer->prepare();

        auto cmd2 = netGlobalDevice.newCommandBuffer(queue_compute);
        cmd2.begin();
        layer->run(cmd2);
        cmd2.end();
        cmd2.submit();
        cmd2.wait();

        Tensor& output_cached = (*layer)["out0"];
        TestAssert::assertShape(output_cached, {1, 1, 64}, "Cached output shape");
        TestAssert::assertTrue(cache.current_len == 6, "Cache length updated to 6");

        std::cout << "    ✓ New token processed: [1, 1, 64]" << std::endl;
        std::cout << "    ✓ Cache length: 5 → 6" << std::endl;

        // Step 3: Verify cache shape remains correct
        std::cout << "\n  Step 3: Cache state verification" << std::endl;
        TestAssert::assertShape(cache.K, {1, 4, 128, 16}, "Cache K shape");
        TestAssert::assertShape(cache.V, {1, 4, 128, 16}, "Cache V shape");
        std::cout << "    ✓ Cache maintains correct shape" << std::endl;
    }
};

// ============================================================================
// Test Suite
// ============================================================================

void runAttentionTests() {
    TestSuite suite("Attention Components");

    // Linear and Softmax tests
    suite.addTest([]() { LinearNodeTest test; return test.execute(); });
    suite.addTest([]() { SoftmaxNodeTest test; return test.execute(); });

    // Attention tests
    suite.addTest([]() { AttentionBasicTest test; return test.execute(); });
    suite.addTest([]() { AttentionJSONTest test; return test.execute(); });

    // KV Cache tests
    suite.addTest([]() { AttentionCacheTest test; return test.execute(); });

    suite.runAll();
}

/**
 * Test Framework Usage Examples
 *
 * This file demonstrates how to use the test framework to create
 * structured, maintainable tests.
 */

#include "testFramework.h"
#include "../model/transformerBlock/transformer.h"
#include "../model/attention/attentionNode.h"
#include <memory>

using namespace vk;

// ============================================================================
// Example 1: Simple LayerNorm Test
// ============================================================================

class LayerNormTest : public BaseTest {
private:
    std::unique_ptr<LayerNormNode> layer;
    Tensor input;
    Tensor expected_output;

public:
    LayerNormTest() : BaseTest("LayerNorm Basic Functionality") {}

    void setup() override {
        // Create layer
        layer = std::make_unique<LayerNormNode>(768, 1e-5f);

        // Prepare input
        input = Tensor(2, 4, 768);
        std::vector<float> input_data(2 * 4 * 768);
        for (size_t i = 0; i < input_data.size(); ++i) {
            input_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        }
        input.set(input_data);
    }

    void run() override {
        // Connect input
        (*layer)["in0"] = input;

        // Prepare layer
        layer->prepare();

        // Execute
        cmd_buffer.begin();
        layer->run(cmd_buffer);
        submitAndWait();

        // Verify output shape
        Tensor& output = (*layer)["out0"];
        TestAssert::assertShape(output, {2, 4, 768}, "LayerNorm output shape");

        // Verify normalization (mean ~0, std ~1)
        std::vector<float> output_data = output.get();

        // Calculate mean and variance over last dimension for first sample
        for (uint32_t b = 0; b < 2; ++b) {
            for (uint32_t s = 0; s < 4; ++s) {
                float sum = 0.0f;
                for (uint32_t d = 0; d < 768; ++d) {
                    sum += output_data[b * 4 * 768 + s * 768 + d];
                }
                float mean = sum / 768.0f;
                TestAssert::assertClose(mean, 0.0f, 1e-3f, "LayerNorm mean");
            }
        }

        std::cout << "  ✓ Output shape verified: [2, 4, 768]" << std::endl;
        std::cout << "  ✓ Normalization verified (mean ~0)" << std::endl;
    }
};

// ============================================================================
// Example 2: FeedForward Test with JSON Verification
// ============================================================================

class FeedForwardJSONTest : public BaseTest {
private:
    std::unique_ptr<FeedForwardNode> layer;
    json test_data;

public:
    FeedForwardJSONTest() : BaseTest("FeedForward vs PyTorch Reference") {}

    void setup() override {
        // Load test data
        test_data = loadTestData("../assets/test_data/feedforward_test_data.json");

        // Extract config
        uint32_t d_model = test_data["config"]["d_model"];

        // Create layer
        layer = std::make_unique<FeedForwardNode>(d_model);
    }

    void run() override {
        uint32_t B = test_data["config"]["batch_size"];
        uint32_t S = test_data["config"]["seq_len"];
        uint32_t D = test_data["config"]["d_model"];
        uint32_t H = test_data["config"]["hidden_dim"];

        // Prepare input
        std::vector<float> input_data = jsonToVector(test_data["input"]);
        Tensor input(B, S, D);
        input.set(input_data);

        // Set weights
        std::vector<float> weight1_data = jsonToVector(test_data["weight1"]);
        std::vector<float> weight2_data = jsonToVector(test_data["weight2"]);

        (*layer)["weight1"] = Tensor(H, D).set(weight1_data).setConstant();
        (*layer)["weight2"] = Tensor(D, H).set(weight2_data).setConstant();
        (*layer)["in0"] = input;

        // Prepare and run
        layer->prepare();

        cmd_buffer.begin();
        layer->run(cmd_buffer);
        submitAndWait();

        // Verify output
        Tensor& output = (*layer)["out0"];
        std::vector<float> expected_output = jsonToVector(test_data["output"]);
        Tensor expected(B, S, D);
        expected.set(expected_output);

        TestAssert::assertEqual(output, expected, 1e-3f, "FeedForward output");

        std::cout << "  ✓ Output matches PyTorch reference (within 1e-3 tolerance)" << std::endl;
    }
};

// ============================================================================
// Example 3: Attention Cache Test (Complex Test)
// ============================================================================

class AttentionCacheTest : public BaseTest {
private:
    std::unique_ptr<MultiHeadAttentionNode> layer;
    LayerKVCache cache;

public:
    AttentionCacheTest() : BaseTest("MultiHeadAttention KV Cache Functionality") {}

    void setup() override {
        // Create layer
        layer = std::make_unique<MultiHeadAttentionNode>(768, 768, 12);

        // Initialize cache
        cache.max_len = 1024;
        cache.current_len = 0;
        cache.K = Tensor(1, 12, 1024, 64);  // [B, H, max_len, head_dim]
        cache.V = Tensor(1, 12, 1024, 64);
    }

    void run() override {
        // Step 1: Process initial prompt (cache disabled)
        std::cout << "\n  Step 1: Initial prompt (no cache)" << std::endl;

        Tensor prompt_input(1, 5, 768);  // Batch=1, SeqLen=5
        std::vector<float> prompt_data(5 * 768, 0.1f);
        prompt_input.set(prompt_data);

        (*layer)["in0"] = prompt_input;
        layer->disableCache();
        layer->prepare();

        cmd_buffer.begin();
        layer->run(cmd_buffer);
        submitAndWait();

        Tensor& output_no_cache = (*layer)["out0"];
        TestAssert::assertShape(output_no_cache, {1, 5, 768}, "Prompt output shape");
        std::cout << "    ✓ Prompt processed: [1, 5, 768]" << std::endl;

        // Step 2: Enable cache and process new token
        std::cout << "\n  Step 2: New token with cache enabled" << std::endl;

        layer->setCache(&cache);
        cache.current_len = 5;  // Simulate 5 cached tokens

        Tensor new_token_input(1, 1, 768);  // Batch=1, SeqLen=1 (single new token)
        std::vector<float> token_data(768, 0.2f);
        new_token_input.set(token_data);

        (*layer)["in0"] = new_token_input;
        layer->prepare();

        cmd_buffer.begin();
        layer->run(cmd_buffer);
        submitAndWait();

        Tensor& output_cached = (*layer)["out0"];
        TestAssert::assertShape(output_cached, {1, 1, 768}, "Cached output shape");
        TestAssert::assertTrue(cache.current_len == 6, "Cache length updated");

        std::cout << "    ✓ New token processed with cache: [1, 1, 768]" << std::endl;
        std::cout << "    ✓ Cache length updated: 5 → 6" << std::endl;

        // Step 3: Verify cache state
        std::cout << "\n  Step 3: Cache state verification" << std::endl;
        TestAssert::assertShape(cache.K, {1, 12, 1024, 64}, "Cache K shape");
        TestAssert::assertShape(cache.V, {1, 12, 1024, 64}, "Cache V shape");
        std::cout << "    ✓ Cache maintains correct shape" << std::endl;
    }
};

// ============================================================================
// Example 4: Using Test Suite to Group Tests
// ============================================================================

void runTransformerLayerTests() {
    TestSuite suite("Transformer Layer Components");

    // Add tests to suite
    suite.addTest([]() {
        LayerNormTest test;
        return test.execute();
    });

    suite.addTest([]() {
        FeedForwardJSONTest test;
        return test.execute();
    });

    suite.addTest([]() {
        AttentionCacheTest test;
        return test.execute();
    });

    // Run all tests
    suite.runAll();
}

// ============================================================================
// Example 5: Parameterized Test (Multiple Configurations)
// ============================================================================

class ParameterizedLayerNormTest : public BaseTest {
private:
    uint32_t batch_size;
    uint32_t seq_len;
    uint32_t d_model;

public:
    ParameterizedLayerNormTest(uint32_t B, uint32_t S, uint32_t D)
        : BaseTest("LayerNorm [" + std::to_string(B) + "," +
                   std::to_string(S) + "," + std::to_string(D) + "]"),
          batch_size(B), seq_len(S), d_model(D) {}

    void run() override {
        LayerNormNode layer(d_model);

        Tensor input(batch_size, seq_len, d_model);
        std::vector<float> input_data(batch_size * seq_len * d_model, 0.5f);
        input.set(input_data);

        layer["in0"] = input;
        layer.prepare();

        cmd_buffer.begin();
        layer.run(cmd_buffer);
        submitAndWait();

        Tensor& output = layer["out0"];
        TestAssert::assertShape(output, {batch_size, seq_len, d_model});

        std::cout << "  ✓ Test passed for shape [" << batch_size << ","
                  << seq_len << "," << d_model << "]" << std::endl;
    }
};

void runParameterizedTests() {
    TestSuite suite("Parameterized Tests");

    // Test various configurations
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> configs = {
        {1, 10, 768},
        {2, 20, 768},
        {4, 50, 768},
        {8, 100, 768}
    };

    for (const auto& [B, S, D] : configs) {
        suite.addTest([B, S, D]() {
            ParameterizedLayerNormTest test(B, S, D);
            return test.execute();
        });
    }

    suite.runAll();
}

// ============================================================================
// Main: Run Example Tests
// ============================================================================

int main() {
    std::cout << "╔════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Test Framework Examples               ║" << std::endl;
    std::cout << "╚════════════════════════════════════════╝\n" << std::endl;

    // Example 1: Single test
    std::cout << "\n=== Example 1: Single Test ===" << std::endl;
    LayerNormTest test1;
    auto result1 = test1.execute();
    result1.print();

    // Example 2: Test suite
    std::cout << "\n\n=== Example 2: Test Suite ===" << std::endl;
    runTransformerLayerTests();

    // Example 3: Parameterized tests
    std::cout << "\n\n=== Example 3: Parameterized Tests ===" << std::endl;
    runParameterizedTests();

    return 0;
}

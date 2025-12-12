/**
 * Test Runner - GPT-2 Layer Unit Tests
 */

#include "../core/globalContext.h"
#include "../core/vulkanApp.h"
#include "graphTest.h"
#include "../model/attention/attentionNode.h"
#include "../model/transformerBlock/transformer.h"
#include "../model/cache/kvCache.h"
#include <iostream>
#include <vector>
#include <memory>

using namespace vk;

// Global test container
static std::vector<std::unique_ptr<ITest>> tests;

// ============================================================================
// Template Specialization for MultiHeadAttentionNode (KV Cache Support)
// ============================================================================

template<>
void GraphTest<MultiHeadAttentionNode>::runSequence() {
    std::cout << "  Running sequence test with " << sequenceSteps.size() << " steps..." << std::endl;
    std::cout << "  Enabling KV cache for autoregressive generation..." << std::endl;

    actualOutputTensors.clear();

    // Initialize KV cache (following KVCacheManager::initialize pattern)
    LayerKVCache kvCache;
    uint32_t num_heads = targetGraph->getNumHeads();
    uint32_t head_dim = targetGraph->getHeadDim();
    uint32_t batch = 1;

    // Create cache tensors [batch, num_heads, max_len, head_dim]
    size_t cache_bytes = batch * num_heads * maxCacheLength * head_dim * sizeof(float);
    BufferPool& pool = BufferPool::get();

    kvCache.K = Tensor(batch, num_heads, maxCacheLength, head_dim);
    kvCache.K.bindBuffer(pool.requestBuffer(
        netGlobalDevice,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        cache_bytes
    ));
    kvCache.K.setConstant(true);  // Prevent BufferPool from reusing

    kvCache.V = Tensor(batch, num_heads, maxCacheLength, head_dim);
    kvCache.V.bindBuffer(pool.requestBuffer(
        netGlobalDevice,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        cache_bytes
    ));
    kvCache.V.setConstant(true);  // Prevent BufferPool from reusing

    kvCache.current_len = 0;
    kvCache.max_len = maxCacheLength;

    // Enable KV cache for the attention layer
    targetGraph->setCache(&kvCache);

    std::cout << "    Cache initialized: max_len=" << maxCacheLength
              << ", heads=" << num_heads << ", head_dim=" << head_dim << std::endl;

    // Process sequence step-by-step with KV cache
    for (size_t i = 0; i < sequenceSteps.size(); ++i) {
        auto& step = sequenceSteps[i];

        std::cout << "    Step " << (i+1) << ": cache_len=" << step.cacheLength
                  << ", new_tokens=1..." << std::endl;

        try {
            // Update cache length to match this step
            kvCache.current_len = step.cacheLength;

            // Create input tensor and assign to slot
            Tensor inputTensor = createInputTensor(step.input.shape, step.input.data);
            network->input(0)["in0"] = inputTensor;

            // Run inference with KV cache
            Tensor& inputRef = network->input(0)["in0"];
            auto outputs = (*network)(inputRef);

            // After processing, cache should have grown by 1
            kvCache.current_len = step.cacheLength + 1;

            // Store output for verification
            actualOutputTensors.push_back(outputs[0]);
            std::cout << "      Step " << (i+1) << " completed (cache now at "
                      << kvCache.current_len << " tokens)" << std::endl;

        } catch (const std::exception& e) {
            std::cout << "      Step " << (i+1) << " FAILED: " << e.what() << std::endl;
            throw;
        }
    }

    // Disable cache after test
    targetGraph->disableCache();
}

// Test registration helper function
template<typename NodeType, typename... Args>
void addTest(const std::string& testName, const std::string& jsonPath, Args&&... args) {
    tests.push_back(std::make_unique<GraphTest<NodeType>>(testName, jsonPath, std::forward<Args>(args)...));
}

void registerTests() {
    addTest<LinearNode>(
        "Linear - Forward Pass (2x4x768 -> 2x4x768)",
        PROJECT_CURRENT_DIR "/assets/test_data/linear_test.json",
        768, 768);

    addTest<LayerNormNode>(
        "LayerNorm - Standard (2x4x768)",
        PROJECT_CURRENT_DIR "/assets/test_data/layernorm_test.json",
        768);

    addTest<GELUNode>(
        "GELU - Standard (2x3x8)",
        PROJECT_CURRENT_DIR "/assets/test_data/gelu_test.json");

    addTest<AddNode>(
        "AddNode - Residual Connection (2x4x768)",
        PROJECT_CURRENT_DIR "/assets/test_data/add_test.json");

    addTest<MultiHeadAttentionNode>(
        "MultiHeadAttention - Self-Attention (1x4x768, 12 heads)",
        PROJECT_CURRENT_DIR "/assets/test_data/attention_test.json",
        768, 768, 12);

    addTest<FeedForwardNode>(
        "FeedForward - MLP (2x4x768, hidden=3072)",
        PROJECT_CURRENT_DIR "/assets/test_data/feedforward_test.json",
        768);

    addTest<TransformerBlock>(
        "TransformerBlock - Full Block (1x4x768, 12 heads)",
        PROJECT_CURRENT_DIR "/assets/test_data/transformer_test.json",
        768, 12);

    addTest<MultiHeadAttentionNode>(
        "MultiHeadAttention - Autoregressive Sequence (4 steps, KV Cache)",
        PROJECT_CURRENT_DIR "/assets/test_data/attention_sequence_test.json",
        768, 768, 12);
}

int main() {
    try {
        std::cout << "╔════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║               Unit Tests - Layer Testing               ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════╝\n" << std::endl;

        // Pre-compile all shaders to eliminate runtime compilation latency
        std::cout << "Loading shaders..." << std::endl;
        loadAllShaders();
        std::cout << "All shaders loaded.\n" << std::endl;

        registerTests();

        int total_tests = 0;
        int passed_tests = 0;

        for (auto& test : tests) {
            if (test->execute()) {
                passed_tests++;
            }
            total_tests++;
        }

        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "OVERALL TEST SUMMARY" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "Total tests run: " << total_tests << std::endl;
        std::cout << "Tests passed: " << passed_tests << std::endl;
        std::cout << "Tests failed: " << (total_tests - passed_tests) << std::endl;

        if (passed_tests == total_tests) {
            std::cout << "\n✓ ALL TESTS PASSED!" << std::endl;
            return 0;
        } else {
            std::cout << "\n✗ SOME TESTS FAILED" << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cout << "\n✗ Fatal error: " << e.what() << std::endl;
        return 1;
    }
}

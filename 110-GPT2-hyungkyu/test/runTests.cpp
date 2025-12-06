/**
 * Test Runner - GPT-2 Layer Unit Tests
 */

#include "../core/globalContext.h"
#include "../core/vulkanApp.h"
#include "graphTest.h"
#include "../model/attention/attentionNode.h"
#include "../model/transformerBlock/transformer.h"
#include <iostream>
#include <vector>
#include <memory>

using namespace vk;

// Global test container
static std::vector<std::unique_ptr<ITest>> tests;

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
}

int main() {
    try {
        std::cout << "╔════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║               Unit Tests - Layer Testing               ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════╝\n" << std::endl;

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

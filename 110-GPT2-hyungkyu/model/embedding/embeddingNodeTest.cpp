#include "embeddingNode.h"
#include "../../core/neuralNet.h"
#include "../../core/error.h"
#include <iostream>
#include <iomanip>
#include <cstring>
#include <random>
#include <cmath>

using namespace vk;

// Global device for neural network operations (defined in neuralNodes.cpp pattern)
Device netGlobalDevice = VulkanApp::get().device();

// Global descriptor pool
DescriptorPool gDestSetPool = netGlobalDevice.createDescriptorPool({
    .maxTypes = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER <= 200},
    .maxSets = 100
});

void testTokenEmbedding()
{
    std::cout << "\n========== Test: Token Embedding Node ==========" << std::endl;

    const uint32_t vocab_size = 100;  // Smaller vocab for testing
    const uint32_t embedding_dim = 8;  // Smaller dim for easier verification
    const uint32_t batch_size = 2;
    const uint32_t seq_length = 3;

    std::cout << "Creating neural network with Token Embedding..." << std::endl;

    // Create neural net
    NeuralNet net(netGlobalDevice, 1, 1);

    // Create embedding node
    TokenEmbeddingNode tokenEmb(vocab_size, embedding_dim);

    net.input(0) - tokenEmb - net.output(0);

    // Prepare token IDs (as float, since Tensor uses float)
    std::vector<float> token_ids_float = {
        10, 20, 30,  // First batch
        40, 50, 60   // Second batch
    };

    std::cout << "Token IDs: ";
    for (auto id : token_ids_float) std::cout << (int)id << " ";
    std::cout << std::endl;

    // Create input tensor
    Tensor inputTensor = Tensor(batch_size, seq_length).set(token_ids_float);

    // Initialize weights with known pattern
    // For token_id=10, embedding will be [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
    // For token_id=20, embedding will be [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7]
    std::vector<float> weights(vocab_size * embedding_dim);
    for (uint32_t v = 0; v < vocab_size; ++v) {
        for (uint32_t e = 0; e < embedding_dim; ++e) {
            weights[v * embedding_dim + e] = v / 10.0f + e / 10.0f;
        }
    }

    std::cout << "Sample weights:" << std::endl;
    std::cout << "  Token 10: ";
    for (uint32_t e = 0; e < embedding_dim; ++e) {
        std::cout << weights[10 * embedding_dim + e] << " ";
    }
    std::cout << std::endl;

    std::cout << "  Token 20: ";
    for (uint32_t e = 0; e < embedding_dim; ++e) {
        std::cout << weights[20 * embedding_dim + e] << " ";
    }
    std::cout << std::endl;

    tokenEmb["weight"] = Tensor(vocab_size, embedding_dim).set(weights);

    std::cout << "Running inference..." << std::endl;

    // Run inference
    Tensor result = net(inputTensor)[0];

    // Copy result back to host
    Buffer outBuffer = netGlobalDevice.createBuffer({
        .size = batch_size * seq_length * embedding_dim * sizeof(float),
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });

    Buffer resultBuffer = result.buffer();
    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuffer, resultBuffer)
        .end()
        .submit()
        .wait();

    float* data = (float*)outBuffer.map();

    std::cout << "\nVerifying results..." << std::endl;
    std::cout << "  Output shape: [" << batch_size << ", " << seq_length << ", " << embedding_dim << "]" << std::endl;

    // Verify token 10 (first token)
    std::cout << "\n  First token (ID=10) embedding: ";
    bool token10_correct = true;
    for (uint32_t e = 0; e < embedding_dim; ++e) {
        float expected = 1.0f + e / 10.0f;
        float actual = data[0 * embedding_dim + e];
        std::cout << std::fixed << std::setprecision(1) << actual << " ";
        if (std::abs(actual - expected) > 0.01f) {
            token10_correct = false;
        }
    }
    std::cout << (token10_correct ? "✓ PASS" : "✗ FAIL") << std::endl;

    // Verify token 20 (second token)
    std::cout << "  Second token (ID=20) embedding: ";
    bool token20_correct = true;
    for (uint32_t e = 0; e < embedding_dim; ++e) {
        float expected = 2.0f + e / 10.0f;
        float actual = data[1 * embedding_dim + e];
        std::cout << std::fixed << std::setprecision(1) << actual << " ";
        if (std::abs(actual - expected) > 0.01f) {
            token20_correct = false;
        }
    }
    std::cout << (token20_correct ? "✓ PASS" : "✗ FAIL") << std::endl;

    // Verify token 30 (third token)
    std::cout << "  Third token (ID=30) embedding: ";
    bool token30_correct = true;
    for (uint32_t e = 0; e < embedding_dim; ++e) {
        float expected = 3.0f + e / 10.0f;
        float actual = data[2 * embedding_dim + e];
        std::cout << std::fixed << std::setprecision(1) << actual << " ";
        if (std::abs(actual - expected) > 0.01f) {
            token30_correct = false;
        }
    }
    std::cout << (token30_correct ? "✓ PASS" : "✗ FAIL") << std::endl;

    if (token10_correct && token20_correct && token30_correct) {
        std::cout << "\n✓ Token embedding numerical verification PASSED" << std::endl;
    } else {
        std::cout << "\n✗ Token embedding numerical verification FAILED" << std::endl;
        std::cout << "Expected values based on pattern: token_id/10.0 + embedding_index/10.0" << std::endl;
    }
}

void testGPTEmbedding()
{
    std::cout << "\n========== Test: GPT Embedding Node ==========" << std::endl;

    const uint32_t vocab_size = 100;
    const uint32_t max_length = 50;
    const uint32_t embedding_dim = 8;
    const uint32_t batch_size = 2;
    const uint32_t seq_length = 3;

    std::cout << "Creating neural network with GPT Embedding (Token + Positional)..." << std::endl;

    // Create neural net
    NeuralNet net(netGlobalDevice, 1, 1);

    // Create GPT embedding node
    GPTEmbeddingNode gptEmb(vocab_size, max_length, embedding_dim);

    net.input(0) - gptEmb - net.output(0);

    // Prepare token IDs
    std::vector<float> token_ids_float = {
        10, 20, 30,
        40, 50, 60
    };

    std::cout << "Token IDs: ";
    for (auto id : token_ids_float) std::cout << (int)id << " ";
    std::cout << std::endl;

    Tensor inputTensor = Tensor(batch_size, seq_length).set(token_ids_float);

    // Initialize token embedding weights
    // Pattern: token_id/10.0 + embedding_index/10.0
    std::vector<float> token_weights(vocab_size * embedding_dim);
    for (uint32_t v = 0; v < vocab_size; ++v) {
        for (uint32_t e = 0; e < embedding_dim; ++e) {
            token_weights[v * embedding_dim + e] = v / 10.0f + e / 10.0f;
        }
    }

    // Initialize positional embedding weights
    // Pattern: position * 0.01 + embedding_index * 0.001
    std::vector<float> pos_weights(max_length * embedding_dim);
    for (uint32_t p = 0; p < max_length; ++p) {
        for (uint32_t e = 0; e < embedding_dim; ++e) {
            pos_weights[p * embedding_dim + e] = p * 0.01f + e * 0.001f;
        }
    }

    std::cout << "Sample positional embeddings:" << std::endl;
    std::cout << "  Position 0: ";
    for (uint32_t e = 0; e < embedding_dim; ++e) {
        std::cout << pos_weights[0 * embedding_dim + e] << " ";
    }
    std::cout << std::endl;

    std::cout << "  Position 1: ";
    for (uint32_t e = 0; e < embedding_dim; ++e) {
        std::cout << pos_weights[1 * embedding_dim + e] << " ";
    }
    std::cout << std::endl;

    gptEmb["token_weight"] = Tensor(vocab_size, embedding_dim).set(token_weights);
    gptEmb["pos_weight"] = Tensor(max_length, embedding_dim).set(pos_weights);

    std::cout << "Running inference..." << std::endl;

    // Run inference
    Tensor result = net(inputTensor)[0];

    // Copy result back to host
    Buffer outBuffer = netGlobalDevice.createBuffer({
        .size = batch_size * seq_length * embedding_dim * sizeof(float),
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });

    Buffer resultBuffer = result.buffer();
    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuffer, resultBuffer)
        .end()
        .submit()
        .wait();

    float* data = (float*)outBuffer.map();

    std::cout << "\nVerifying results..." << std::endl;
    std::cout << "  Output shape: [" << batch_size << ", " << seq_length << ", " << embedding_dim << "]" << std::endl;

    // Verify first token (ID=10, position=0)
    // Expected: token_emb[10] + pos_emb[0]
    std::cout << "\n  First token (ID=10, pos=0) combined embedding: ";
    bool first_correct = true;
    for (uint32_t e = 0; e < embedding_dim; ++e) {
        float token_part = 1.0f + e / 10.0f;  // token 10
        float pos_part = 0.0f + e * 0.001f;   // position 0
        float expected = token_part + pos_part;
        float actual = data[0 * embedding_dim + e];
        std::cout << std::fixed << std::setprecision(3) << actual << " ";
        if (std::abs(actual - expected) > 0.01f) {
            first_correct = false;
        }
    }
    std::cout << (first_correct ? "✓ PASS" : "✗ FAIL") << std::endl;

    // Verify second token (ID=20, position=1)
    std::cout << "  Second token (ID=20, pos=1) combined embedding: ";
    bool second_correct = true;
    for (uint32_t e = 0; e < embedding_dim; ++e) {
        float token_part = 2.0f + e / 10.0f;   // token 20
        float pos_part = 0.01f + e * 0.001f;   // position 1
        float expected = token_part + pos_part;
        float actual = data[1 * embedding_dim + e];
        std::cout << std::fixed << std::setprecision(3) << actual << " ";
        if (std::abs(actual - expected) > 0.01f) {
            second_correct = false;
        }
    }
    std::cout << (second_correct ? "✓ PASS" : "✗ FAIL") << std::endl;

    // Verify third token (ID=30, position=2)
    std::cout << "  Third token (ID=30, pos=2) combined embedding: ";
    bool third_correct = true;
    for (uint32_t e = 0; e < embedding_dim; ++e) {
        float token_part = 3.0f + e / 10.0f;   // token 30
        float pos_part = 0.02f + e * 0.001f;   // position 2
        float expected = token_part + pos_part;
        float actual = data[2 * embedding_dim + e];
        std::cout << std::fixed << std::setprecision(3) << actual << " ";
        if (std::abs(actual - expected) > 0.01f) {
            third_correct = false;
        }
    }
    std::cout << (third_correct ? "✓ PASS" : "✗ FAIL") << std::endl;

    if (first_correct && second_correct && third_correct) {
        std::cout << "\n✓ GPT embedding numerical verification PASSED" << std::endl;
        std::cout << "  Token + Positional addition working correctly!" << std::endl;
    } else {
        std::cout << "\n✗ GPT embedding numerical verification FAILED" << std::endl;
    }
}

void embeddingNodeTest()
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "Embedding Node (Vulkan) - Numerical Verification Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        testTokenEmbedding();
        testGPTEmbedding();

        std::cout << "\n========================================" << std::endl;
        std::cout << "All Embedding Node tests completed!" << std::endl;
        std::cout << "========================================" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with exception: " << e.what() << std::endl;
    }
}

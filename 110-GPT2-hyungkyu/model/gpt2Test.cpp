#include "gpt2Net.h"
#include "gpt2Generation.h"
#include "gpt2Weights.h"
#include "../core/globalContext.h"
#include "../core/error.h"
#include "../tokenizer/bpeTokenizer.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <ctime>

using namespace vk;

void testGPT2()
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "GPT-2 Simple Test" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Create tiny GPT-2 for testing
    GPT2Config config = GPT2TinyConfig();
    GPT2Net gpt2Net(netGlobalDevice, config);

    std::cout << "✓ Network created" << std::endl;
    std::cout << "✓ All tests passed (network construction successful)" << std::endl;
}

// Helper: Initialize embedding weights (token + position)
static void initializeEmbeddingWeights(GPT2Net& gpt2Net, const GPT2Config& config)
{
    // Token embeddings
    std::vector<float> token_emb_data(config.vocab_size * config.d_model);
    for (auto& val : token_emb_data) {
        val = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    }
    gpt2Net["embedding.token"] = Tensor(config.vocab_size, config.d_model)
        .set(token_emb_data).setConstant();

    // Positional embeddings
    std::vector<float> pos_emb_data(config.max_seq_len * config.d_model);
    for (auto& val : pos_emb_data) {
        val = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    }
    gpt2Net["embedding.position"] = Tensor(config.max_seq_len, config.d_model)
        .set(pos_emb_data).setConstant();
}

// Helper: Initialize weights for a single transformer layer
static void initializeTransformerLayerWeights(
    GPT2Net& gpt2Net,
    const GPT2Config& config,
    uint32_t layer_idx)
{
    std::string prefix = "block." + std::to_string(layer_idx) + ".";

    // Layer norm 1
    std::vector<float> ln_scale(config.d_model, 1.0f);
    std::vector<float> ln_shift(config.d_model, 0.0f);
    gpt2Net[prefix + "norm1_scale"] = Tensor(config.d_model).set(ln_scale).setConstant();
    gpt2Net[prefix + "norm1_shift"] = Tensor(config.d_model).set(ln_shift).setConstant();

    // Attention weights
    uint32_t attn_size = config.d_model * config.d_model;
    std::vector<float> attn_data(attn_size);
    for (auto& val : attn_data) {
        val = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    }

    gpt2Net[prefix + "attn_wq"] = Tensor(config.d_model, config.d_model).set(attn_data).setConstant();
    gpt2Net[prefix + "attn_wk"] = Tensor(config.d_model, config.d_model).set(attn_data).setConstant();
    gpt2Net[prefix + "attn_wv"] = Tensor(config.d_model, config.d_model).set(attn_data).setConstant();
    gpt2Net[prefix + "attn_wout"] = Tensor(config.d_model, config.d_model).set(attn_data).setConstant();

    // Layer norm 2
    gpt2Net[prefix + "norm2_scale"] = Tensor(config.d_model).set(ln_scale).setConstant();
    gpt2Net[prefix + "norm2_shift"] = Tensor(config.d_model).set(ln_shift).setConstant();

    // Feed-forward weights
    uint32_t ff1_size = (4 * config.d_model) * config.d_model;
    uint32_t ff2_size = config.d_model * (4 * config.d_model);

    std::vector<float> ff1_data(ff1_size);
    std::vector<float> ff2_data(ff2_size);
    for (auto& val : ff1_data) val = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    for (auto& val : ff2_data) val = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;

    gpt2Net[prefix + "ff_w1"] = Tensor(4 * config.d_model, config.d_model).set(ff1_data).setConstant();
    gpt2Net[prefix + "ff_w2"] = Tensor(config.d_model, 4 * config.d_model).set(ff2_data).setConstant();
}

// Helper: Initialize all network weights
static void initializeAllWeights(GPT2Net& gpt2Net, const GPT2Config& config)
{
    std::cout << "Initializing random weights for testing..." << std::endl;

    // Embeddings
    initializeEmbeddingWeights(gpt2Net, config);

    // Transformer blocks
    for (uint32_t layer = 0; layer < config.num_layers; ++layer) {
        initializeTransformerLayerWeights(gpt2Net, config, layer);
    }

    // Final layer norm
    std::vector<float> final_ln_scale(config.d_model, 1.0f);
    std::vector<float> final_ln_shift(config.d_model, 0.0f);
    gpt2Net["finalNorm.weight"] = Tensor(config.d_model).set(final_ln_scale).setConstant();
    gpt2Net["finalNorm.bias"] = Tensor(config.d_model).set(final_ln_shift).setConstant();

    // LM head (weight tying)
    gpt2Net.lmHead["weight"] = gpt2Net["embedding.token"];

    std::cout << "✓ Weights initialized (including " << config.num_layers << " transformer layers)\n" << std::endl;
}

// Helper: Run generation for a single prompt
static void runPromptGeneration(
    GPT2Net& gpt2Net,
    BPETokenizer& tokenizer,
    const std::string& prompt_text,
    uint32_t num_tokens_to_generate,
    float temperature = 1.0f,
    int top_k = 50,
    int seed = -1)
{
    std::cout << "\n----------------------------------------" << std::endl;
    std::cout << "Prompt: \"" << prompt_text << "\"" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // Encode prompt
    std::vector<int> prompt_ids = tokenizer.encode(prompt_text);
    std::cout << "Encoded to " << prompt_ids.size() << " tokens: ";
    for (int id : prompt_ids) {
        std::cout << id << " ";
    }
    std::cout << std::endl;

    // Generate tokens
    std::cout << "\nGenerating " << num_tokens_to_generate << " new tokens..." << std::endl;

    // Measure generation time
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<int> generated_ids = generate_gpt2(
        gpt2Net,
        prompt_ids,
        num_tokens_to_generate,
        temperature,
        top_k,
        seed
    );

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Decode result
    std::string generated_text = tokenizer.decode(generated_ids);

    std::cout << "\n--- Generated Text ---" << std::endl;
    std::cout << generated_text << std::endl;
    std::cout << "--- End of Generation ---\n" << std::endl;

    // Print statistics
    uint32_t new_tokens = generated_ids.size() - prompt_ids.size();
    double generation_time_sec = duration.count() / 1000.0;
    double tokens_per_sec = new_tokens / generation_time_sec;

    std::cout << "Generated " << new_tokens << " new tokens (total: " << generated_ids.size() << " tokens)" << std::endl;
    std::cout << "Generation time: " << duration.count() << " ms (" << std::fixed << std::setprecision(2) << generation_time_sec << " sec)" << std::endl;
    std::cout << "Generation speed: " << std::fixed << std::setprecision(2) << tokens_per_sec << " tokens/sec" << std::endl;
}

// Helper: Create and initialize network with random weights
static GPT2Net createNetworkWithRandomWeights(const GPT2Config& config)
{
    std::cout << "Creating GPT-2 network..." << std::endl;
    std::cout << "  Config: d_model=" << config.d_model
              << ", num_heads=" << config.num_heads
              << ", num_layers=" << config.num_layers << std::endl;
    GPT2Net gpt2Net(netGlobalDevice, config);
    std::cout << "✓ Network created\n" << std::endl;

    // Initialize weights
    initializeAllWeights(gpt2Net, config);

    return gpt2Net;
}

// Helper: Create and initialize network with pretrained weights
static GPT2Net createNetworkWithPretrainedWeights(
    const GPT2Config& config,
    const std::string& weights_file)
{
    std::cout << "Creating GPT-2 network..." << std::endl;
    std::cout << "  Config: d_model=" << config.d_model
              << ", num_heads=" << config.num_heads
              << ", num_layers=" << config.num_layers << std::endl;
    GPT2Net gpt2Net(netGlobalDevice, config);
    std::cout << "✓ Network created\n" << std::endl;

    // Load pretrained weights
    std::cout << "Loading pretrained weights from: " << weights_file << std::endl;
    loadGPT2Weights(gpt2Net, weights_file);
    std::cout << std::endl;

    return gpt2Net;
}

// Helper: Load tokenizer
static BPETokenizer loadTokenizer()
{
    std::cout << "Loading tokenizer..." << std::endl;
    BPETokenizer tokenizer("110-GPT2-hyungkyu/assets/vocab.json",
                          "110-GPT2-hyungkyu/assets/merges.txt");
    std::cout << "✓ Tokenizer loaded\n" << std::endl;
    return tokenizer;
}

// Helper: Run generation tests on multiple prompts
static void runGenerationTests(
    GPT2Net& gpt2Net,
    BPETokenizer& tokenizer,
    const std::vector<std::string>& prompts,
    uint32_t num_tokens)
{
    for (const auto& prompt_text : prompts) {
        try {
            runPromptGeneration(gpt2Net, tokenizer, prompt_text, num_tokens);
        } catch (const std::exception& e) {
            std::cout << "\n✗ Generation failed: " << e.what() << std::endl;
            throw;  // Re-throw to let caller handle
        }
    }
}

// Common function: Run complete text generation test with a network
static void runGPT2TextGenerationTest(
    GPT2Net& gpt2Net,
    const std::vector<std::string>& test_prompts,
    uint32_t num_tokens_to_generate)
{
    // Load tokenizer
    BPETokenizer tokenizer = loadTokenizer();

    // Run generation tests
    runGenerationTests(gpt2Net, tokenizer, test_prompts, num_tokens_to_generate);

    std::cout << "\n========================================" << std::endl;
    std::cout << "✓ Text generation test completed!" << std::endl;
    std::cout << "========================================" << std::endl;
}

void testGPT2Generation()
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "GPT-2 Text Generation Test (Random Weights)" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Initialize random seed
    srand((unsigned int)time(nullptr));

    // Create network with random weights
    GPT2Config config = GPT2TinyConfig();
    GPT2Net gpt2Net = createNetworkWithRandomWeights(config);

    // Test prompts
    std::vector<std::string> test_prompts = {
        "Hello",
        "Once upon a time",
        "The quick brown fox"
    };

    // Run generation test
    runGPT2TextGenerationTest(gpt2Net, test_prompts, 5);
}

void testGPT2Pretrained(const std::string& prompt, uint32_t max_tokens, bool use_cache)
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "GPT-2 Text Generation Test (Pretrained Weights)" << std::endl;
    if (use_cache) {
        std::cout << "Mode: KV Cache Enabled (Fast)" << std::endl;
    } else {
        std::cout << "Mode: Standard Generation (No Cache)" << std::endl;
    }
    std::cout << "========================================\n" << std::endl;

    // Check if weights exist
    const std::string weights_dir = "110-GPT2-hyungkyu/assets/weights/124M/";
    const std::string config_file = weights_dir + "gpt2_config.txt";
    const std::string weights_file = weights_dir + "gpt2_weights.bin";

    std::ifstream test_file(weights_file);
    if (!test_file) {
        std::cout << "⚠ Pretrained weights not found at: " << weights_file << std::endl;
        std::cout << "  Run download_gpt2_weights.py to download pretrained weights" << std::endl;
        std::cout << "  Skipping pretrained weights test" << std::endl;
        return;
    }
    test_file.close();

    try {
        // Load configuration
        std::cout << "Loading configuration from: " << config_file << std::endl;
        GPT2Config config = loadGPT2Config(config_file);
        std::cout << "  Config: vocab_size=" << config.vocab_size
                  << ", d_model=" << config.d_model
                  << ", num_heads=" << config.num_heads
                  << ", num_layers=" << config.num_layers << std::endl;
        std::cout << "✓ Configuration loaded\n" << std::endl;

        // Create network with pretrained weights
        std::cout << "Creating GPT-2 network..." << std::endl;
        GPT2Net gpt2Net = createNetworkWithPretrainedWeights(config, weights_file);

        // Load tokenizer
        std::cout << "\nLoading tokenizer..." << std::endl;
        BPETokenizer tokenizer = loadTokenizer();
        std::cout << "✓ Tokenizer loaded\n" << std::endl;

        // Encode prompt
        std::vector<int> prompt_ids = tokenizer.encode(prompt);
        std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
        std::cout << "Prompt encoded to " << prompt_ids.size() << " tokens" << std::endl;
        std::cout << "Max tokens to generate: " << max_tokens << std::endl;

        // Generate text
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<int> generated_ids;

        if (use_cache) {
            std::cout << "\nGenerating with KV cache..." << std::endl;
            generated_ids = generate_gpt2_with_cache(
                gpt2Net,
                prompt_ids,
                max_tokens,
                0.8f,  // temperature
                40,    // top_k
                42     // seed for reproducibility
            );
        } else {
            std::cout << "\nGenerating without cache..." << std::endl;
            generated_ids = generate_gpt2(
                gpt2Net,
                prompt_ids,
                max_tokens,
                0.8f,  // temperature
                40,    // top_k
                42     // seed for reproducibility
            );
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Decode and display
        std::string generated_text = tokenizer.decode(generated_ids);
        std::cout << "\n--- Generated Text ---" << std::endl;
        std::cout << generated_text << std::endl;
        std::cout << "--- End ---\n" << std::endl;

        // Performance stats
        double time_sec = duration.count() / 1000.0;
        double speed = max_tokens / time_sec;
        std::cout << "Generation time: " << duration.count() << " ms ("
                  << std::fixed << std::setprecision(2) << time_sec << " sec)" << std::endl;
        std::cout << "Generation speed: " << std::fixed << std::setprecision(2)
                  << speed << " tokens/sec" << std::endl;

        // Second test disabled due to GPU memory constraints (BufferPool accumulation)
        // TODO: Fix BufferPool to release memory between generations
        /*
        std::cout << "\n=== Sampling with Random Seed (Reproducible) ===" << std::endl;
        {
            BPETokenizer tokenizer = loadTokenizer();
            runPromptGeneration(gpt2Net, tokenizer,
                "Hello, I'm a language model,",
                20,
                1.0f,  // temperature
                50,    // top_k
                42     // seed for reproducibility
            );
        }
        */

    } catch (const std::exception& e) {
        std::cout << "\n✗ Error during pretrained weights test: " << e.what() << std::endl;
        std::cout << "This may be due to GPU memory constraints." << std::endl;
        std::cout << "Try running this test alone (comment out runBasicTests in main.cpp)" << std::endl;
    }
}

/**
 * Test GPT-2 generation with KV cache
 * Compares standard vs cached generation performance
 */
void testGPT2WithCache(const std::string& prompt, uint32_t max_tokens)
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "GPT-2 KV Cache Test (Pretrained Weights)" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Check if weights exist
    const std::string weights_dir = "110-GPT2-hyungkyu/assets/weights/124M/";
    const std::string config_file = weights_dir + "gpt2_config.txt";
    const std::string weights_file = weights_dir + "gpt2_weights.bin";

    std::ifstream test_file(weights_file);
    if (!test_file) {
        std::cout << "⚠ Pretrained weights not found at: " << weights_file << std::endl;
        std::cout << "  Run download_gpt2_weights.py to download pretrained weights" << std::endl;
        std::cout << "  Skipping KV cache test" << std::endl;
        return;
    }
    test_file.close();

    try {
        // Load configuration
        std::cout << "Loading configuration from: " << config_file << std::endl;
        GPT2Config config = loadGPT2Config(config_file);
        std::cout << "  Config: vocab_size=" << config.vocab_size
                  << ", d_model=" << config.d_model
                  << ", num_heads=" << config.num_heads
                  << ", num_layers=" << config.num_layers << std::endl;
        std::cout << "✓ Configuration loaded\n" << std::endl;

        // Load tokenizer
        BPETokenizer tokenizer = loadTokenizer();

        std::cout << "\n========================================" << std::endl;
        std::cout << "Performance Comparison: Standard vs Cached" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
        std::cout << "Max tokens: " << max_tokens << "\n" << std::endl;

        // Encode prompt
        std::vector<int> prompt_ids = tokenizer.encode(prompt);
        std::cout << "Prompt encoded to " << prompt_ids.size() << " tokens\n" << std::endl;

        // Test 1: Standard generation (baseline) - Create fresh network
        std::cout << "=== Test 1: Standard Generation (No Cache) ===" << std::endl;
        std::cout << "[DEBUG] Creating fresh network for standard generation..." << std::endl;
        GPT2Net gpt2Net_standard = createNetworkWithPretrainedWeights(config, weights_file);

        auto start_standard = std::chrono::high_resolution_clock::now();

        std::vector<int> generated_standard = generate_gpt2(
            gpt2Net_standard,
            prompt_ids,
            max_tokens,
            0.0f,  // temperature (greedy - deterministic)
            0,     // top_k (not used with greedy)
            -1     // seed (not needed with greedy)
        );

        auto end_standard = std::chrono::high_resolution_clock::now();
        auto duration_standard = std::chrono::duration_cast<std::chrono::milliseconds>(end_standard - start_standard);

        std::string text_standard = tokenizer.decode(generated_standard);
        std::cout << "\n--- Standard Generated Text ---" << std::endl;
        std::cout << text_standard << std::endl;
        std::cout << "--- End ---\n" << std::endl;

        double time_standard_sec = duration_standard.count() / 1000.0;
        double speed_standard = max_tokens / time_standard_sec;
        std::cout << "Standard generation time: " << duration_standard.count() << " ms ("
                  << std::fixed << std::setprecision(2) << time_standard_sec << " sec)" << std::endl;
        std::cout << "Standard generation speed: " << std::fixed << std::setprecision(2)
                  << speed_standard << " tokens/sec\n" << std::endl;

        // Test 2: Cached generation - Create fresh network
        std::cout << "\n=== Test 2: Cached Generation (With KV Cache) ===" << std::endl;
        std::cout << "[DEBUG] Creating fresh network for cached generation..." << std::endl;
        GPT2Net gpt2Net_cached = createNetworkWithPretrainedWeights(config, weights_file);

        auto start_cached = std::chrono::high_resolution_clock::now();

        std::vector<int> generated_cached = generate_gpt2_with_cache(
            gpt2Net_cached,
            prompt_ids,
            max_tokens,
            0.0f,  // temperature (greedy - deterministic)
            0,     // top_k (not used with greedy)
            -1     // seed (not needed with greedy)
        );

        auto end_cached = std::chrono::high_resolution_clock::now();
        auto duration_cached = std::chrono::duration_cast<std::chrono::milliseconds>(end_cached - start_cached);

        std::string text_cached = tokenizer.decode(generated_cached);
        std::cout << "\n--- Cached Generated Text ---" << std::endl;
        std::cout << text_cached << std::endl;
        std::cout << "--- End ---\n" << std::endl;

        double time_cached_sec = duration_cached.count() / 1000.0;
        double speed_cached = max_tokens / time_cached_sec;
        std::cout << "Cached generation time: " << duration_cached.count() << " ms ("
                  << std::fixed << std::setprecision(2) << time_cached_sec << " sec)" << std::endl;
        std::cout << "Cached generation speed: " << std::fixed << std::setprecision(2)
                  << speed_cached << " tokens/sec\n" << std::endl;

        // Compare results
        std::cout << "\n========================================" << std::endl;
        std::cout << "Performance Summary" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Standard: " << duration_standard.count() << " ms ("
                  << std::fixed << std::setprecision(2) << speed_standard << " tokens/sec)" << std::endl;
        std::cout << "Cached:   " << duration_cached.count() << " ms ("
                  << std::fixed << std::setprecision(2) << speed_cached << " tokens/sec)" << std::endl;

        double speedup = (double)duration_standard.count() / duration_cached.count();
        std::cout << "\nSpeedup: " << std::fixed << std::setprecision(2) << speedup << "x faster with cache!" << std::endl;

        // Verify same output (with same seed, should be identical)
        bool outputs_match = (generated_standard == generated_cached);
        std::cout << "Output verification: " << (outputs_match ? "✓ MATCH" : "✗ MISMATCH") << std::endl;

        if (!outputs_match) {
            std::cout << "\n⚠ Warning: Outputs don't match (expected with same seed)" << std::endl;
            std::cout << "  This might indicate an issue with cache implementation" << std::endl;
        }

        std::cout << "\n========================================" << std::endl;
        std::cout << "✓ KV Cache Test Completed!" << std::endl;
        std::cout << "========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "\n✗ Error during KV cache test: " << e.what() << std::endl;
    }
}

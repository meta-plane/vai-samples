#include "inference.h"
#include "gpt2Weights.h"
#include "../core/globalContext.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>

using namespace vk;

namespace GPT2Inference {

GPT2Net* loadPretrainedModel(const InferenceConfig& config) {
    try {
        const std::string config_file = config.weights_dir + "gpt2_config.txt";
        const std::string weights_file = config.weights_dir + "gpt2_weights.bin";

        // Check if weights exist
        std::ifstream test_file(weights_file);
        if (!test_file) {
            std::cout << "⚠ Pretrained weights not found at: " << weights_file << std::endl;
            std::cout << "  Run utils/download_gpt2_weights.py to download weights" << std::endl;
            return nullptr;
        }
        test_file.close();

        // Load configuration
        std::cout << "Loading configuration from: " << config_file << std::endl;
        GPT2Config gpt2_config = loadGPT2Config(config_file);
        std::cout << "  Config: vocab_size=" << gpt2_config.vocab_size
                  << ", d_model=" << gpt2_config.d_model
                  << ", num_heads=" << gpt2_config.num_heads
                  << ", num_layers=" << gpt2_config.num_layers << std::endl;
        std::cout << "✓ Configuration loaded\n" << std::endl;

        // Create network
        std::cout << "Creating GPT-2 network..." << std::endl;
        GPT2Net* model = new GPT2Net(netGlobalDevice, gpt2_config);
        std::cout << "✓ Network created\n" << std::endl;

        // Load weights
        std::cout << "Loading pretrained weights from: " << weights_file << std::endl;
        loadGPT2Weights(*model, weights_file);
        std::cout << "✓ Weights loaded\n" << std::endl;

        return model;

    } catch (const std::exception& e) {
        std::cout << "✗ Error loading pretrained model: " << e.what() << std::endl;
        return nullptr;
    }
}

GenerationResult generate(
    GPT2Net& model,
    BPETokenizer& tokenizer,
    const std::string& prompt,
    uint32_t max_tokens,
    const InferenceConfig& config)
{
    GenerationResult result;

    try {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Text Generation" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
        std::cout << "Max tokens: " << max_tokens << std::endl;
        std::cout << "Mode: " << (config.use_cache ? "KV Cache Enabled" : "Standard (No Cache)") << std::endl;
        std::cout << "Temperature: " << config.temperature << std::endl;
        std::cout << "Top-k: " << config.top_k << std::endl;
        std::cout << "========================================\n" << std::endl;

        // Encode prompt
        std::vector<int> prompt_ids = tokenizer.encode(prompt);
        std::cout << "Prompt encoded to " << prompt_ids.size() << " tokens" << std::endl;

        // Generate
        std::cout << "\nGenerating..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<int> generated_ids;
        if (config.use_cache) {
            generated_ids = generate_gpt2_with_cache(
                model, prompt_ids, max_tokens,
                config.temperature, config.top_k, config.seed,
                config.enable_profiling
            );
        } else {
            generated_ids = generate_gpt2(
                model, prompt_ids, max_tokens,
                config.temperature, config.top_k, config.seed
            );
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Decode result
        result.generated_text = tokenizer.decode(generated_ids);
        result.token_ids = generated_ids;
        result.num_tokens_generated = generated_ids.size() - prompt_ids.size();
        result.generation_time_sec = duration.count() / 1000.0;
        result.tokens_per_sec = result.num_tokens_generated / result.generation_time_sec;
        result.success = true;

        // Display results
        std::cout << "\n--- Generated Text ---" << std::endl;
        std::cout << result.generated_text << std::endl;
        std::cout << "--- End ---\n" << std::endl;

        std::cout << "Statistics:" << std::endl;
        std::cout << "  Generated tokens: " << result.num_tokens_generated << std::endl;
        std::cout << "  Total tokens: " << result.token_ids.size() << std::endl;
        std::cout << "  Generation time: " << duration.count() << " ms ("
                  << std::fixed << std::setprecision(2) << result.generation_time_sec << " sec)" << std::endl;
        std::cout << "  Generation speed: " << std::fixed << std::setprecision(2)
                  << result.tokens_per_sec << " tokens/sec" << std::endl;

    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
        std::cout << "\n✗ Generation failed: " << e.what() << std::endl;
    }

    return result;
}

void compareGenerationModes(
    const std::string& prompt,
    uint32_t max_tokens,
    const InferenceConfig& config)
{
    std::cout << "\n╔════════════════════════════════════════╗" << std::endl;
    std::cout << "║  KV Cache Performance Comparison       ║" << std::endl;
    std::cout << "╚════════════════════════════════════════╝\n" << std::endl;

    std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
    std::cout << "Max tokens: " << max_tokens << "\n" << std::endl;

    // Load tokenizer
    std::cout << "Loading tokenizer..." << std::endl;
    BPETokenizer tokenizer(config.vocab_file, config.merges_file);
    std::cout << "✓ Tokenizer loaded\n" << std::endl;

    // Test 1: Standard generation (no cache)
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Test 1: Standard Generation (No Cache)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    GPT2Net* model_standard = loadPretrainedModel(config);
    if (!model_standard) {
        std::cout << "✗ Failed to load model for standard generation" << std::endl;
        return;
    }

    InferenceConfig config_no_cache = config;
    config_no_cache.use_cache = false;
    config_no_cache.temperature = 0.0f;  // Greedy for deterministic results
    config_no_cache.top_k = 0;

    GenerationResult result_standard = generate(*model_standard, tokenizer, prompt, max_tokens, config_no_cache);

    delete model_standard;

    if (!result_standard.success) {
        std::cout << "✗ Standard generation failed" << std::endl;
        return;
    }

    // Test 2: Cached generation
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Test 2: Cached Generation (With KV Cache)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    GPT2Net* model_cached = loadPretrainedModel(config);
    if (!model_cached) {
        std::cout << "✗ Failed to load model for cached generation" << std::endl;
        return;
    }

    InferenceConfig config_cached = config;
    config_cached.use_cache = true;
    config_cached.temperature = 0.0f;  // Greedy for deterministic results
    config_cached.top_k = 0;

    GenerationResult result_cached = generate(*model_cached, tokenizer, prompt, max_tokens, config_cached);

    delete model_cached;

    if (!result_cached.success) {
        std::cout << "✗ Cached generation failed" << std::endl;
        return;
    }

    // Compare results
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Performance Comparison Summary" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    std::cout << "\nStandard Generation:" << std::endl;
    std::cout << "  Time: " << std::fixed << std::setprecision(2)
              << result_standard.generation_time_sec << " sec" << std::endl;
    std::cout << "  Speed: " << std::fixed << std::setprecision(2)
              << result_standard.tokens_per_sec << " tokens/sec" << std::endl;

    std::cout << "\nCached Generation:" << std::endl;
    std::cout << "  Time: " << std::fixed << std::setprecision(2)
              << result_cached.generation_time_sec << " sec" << std::endl;
    std::cout << "  Speed: " << std::fixed << std::setprecision(2)
              << result_cached.tokens_per_sec << " tokens/sec" << std::endl;

    double speedup = result_standard.generation_time_sec / result_cached.generation_time_sec;
    std::cout << "\nSpeedup: " << std::fixed << std::setprecision(2)
              << speedup << "x faster with cache" << std::endl;

    // Verify outputs match (with greedy sampling, they should be identical)
    bool outputs_match = (result_standard.token_ids == result_cached.token_ids);
    std::cout << "Output verification: " << (outputs_match ? "✓ MATCH" : "✗ MISMATCH") << std::endl;

    if (!outputs_match) {
        std::cout << "\n⚠ Warning: Outputs don't match" << std::endl;
        std::cout << "  This may indicate an issue with cache implementation" << std::endl;
    }

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "✓ Comparison completed" << std::endl;
    std::cout << std::string(80, '=') << "\n" << std::endl;
}

void runInteractive(const InferenceConfig& config) {
    std::cout << "\n╔════════════════════════════════════════╗" << std::endl;
    std::cout << "║  GPT-2 Interactive Text Generation     ║" << std::endl;
    std::cout << "╚════════════════════════════════════════╝\n" << std::endl;

    // Load model
    GPT2Net* model = loadPretrainedModel(config);
    if (!model) {
        std::cout << "✗ Failed to load pretrained model" << std::endl;
        return;
    }

    // Load tokenizer
    std::cout << "Loading tokenizer..." << std::endl;
    BPETokenizer tokenizer(config.vocab_file, config.merges_file);
    std::cout << "✓ Tokenizer loaded\n" << std::endl;

    std::cout << "Ready for text generation!" << std::endl;
    std::cout << "Type your prompt (or 'quit' to exit)\n" << std::endl;

    std::string prompt;
    while (true) {
        std::cout << "\n> ";
        std::getline(std::cin, prompt);

        if (prompt == "quit" || prompt == "exit" || prompt == "q") {
            break;
        }

        if (prompt.empty()) {
            continue;
        }

        // Generate with default settings
        generate(*model, tokenizer, prompt, 50, config);
    }

    delete model;
    std::cout << "\nGoodbye!" << std::endl;
}

} // namespace GPT2Inference

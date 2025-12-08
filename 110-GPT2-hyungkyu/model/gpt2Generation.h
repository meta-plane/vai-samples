#ifndef GPT2_GENERATION_H
#define GPT2_GENERATION_H

#include "gpt2Net.h"
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <cstring>  // memcpy
#include <chrono>   // profiling

/**
 * Sample a token from logits using temperature and top-k sampling
 * If temperature is 0.0, uses greedy decoding (deterministic)
 */
inline int sampleToken(const std::vector<float>& logits, float temperature, int top_k)
{
    // Greedy decoding: always pick the token with highest probability
    if (temperature == 0.0f) {
        auto max_it = std::max_element(logits.begin(), logits.end());
        return (int)std::distance(logits.begin(), max_it);
    }

    std::vector<float> probs = logits;

    // Apply temperature
    if (temperature != 1.0f) {
        for (float& logit : probs) {
            logit /= temperature;
        }
    }

    // Softmax
    float max_logit = *std::max_element(probs.begin(), probs.end());
    for (float& p : probs) {
        p = std::exp(p - max_logit);
    }

    float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    for (float& p : probs) {
        p /= sum;
    }

    // Top-k filtering
    if (top_k > 0 && top_k < (int)probs.size()) {
        std::vector<std::pair<float, int>> prob_idx;
        for (size_t i = 0; i < probs.size(); ++i) {
            prob_idx.push_back({probs[i], (int)i});
        }

        std::sort(prob_idx.begin(), prob_idx.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        for (size_t i = 0; i < probs.size(); ++i) {
            probs[i] = 0.0f;
        }
        for (int i = 0; i < top_k; ++i) {
            probs[prob_idx[i].second] = prob_idx[i].first;
        }

        sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
        for (float& p : probs) {
            p /= sum;
        }
    }

    // Sample from categorical distribution
    float r = (float)rand() / RAND_MAX;
    float cumsum = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        cumsum += probs[i];
        if (r < cumsum) {
            return (int)i;
        }
    }

    return (int)probs.size() - 1;
}

/**
 * Generate tokens autoregressively (like eval_mnist but for generation)
 *
 * @param gpt2Net The GPT2 network
 * @param prompt_ids Initial token IDs
 * @param max_new_tokens Number of tokens to generate
 * @param temperature Sampling temperature (0.0 for greedy/deterministic)
 * @param top_k Top-k sampling parameter
 * @param seed Random seed (optional, -1 for random)
 * @return Generated token IDs (prompt + new tokens)
 */
inline std::vector<int> generate_gpt2(
    GPT2Net& gpt2Net,
    const std::vector<int>& prompt_ids,
    uint32_t max_new_tokens,
    float temperature = 1.0f,
    int top_k = 0,
    int seed = -1)
{
    const GPT2Config& config = gpt2Net.getConfig();
    Device& device = netGlobalDevice;

    // Set random seed if provided
    if (seed >= 0) {
        srand((unsigned int)seed);
        std::cout << "\nRandom seed set to: " << seed << std::endl;
    }

    std::vector<int> generated = prompt_ids;

    std::cout << "\nGenerating text..." << std::endl;
    std::cout << "  Prompt length: " << prompt_ids.size() << " tokens" << std::endl;
    std::cout << "  Max new tokens: " << max_new_tokens << std::endl;
    std::cout << "  Temperature: " << temperature << (temperature == 0.0f ? " (greedy)" : "") << std::endl;

    // Pre-allocate CPU buffer for logits (reuse across iterations)
    uint32_t max_logits_size = config.max_seq_len * config.vocab_size * sizeof(float);
    Buffer cpu_buffer = device.createBuffer({
        .size = max_logits_size,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    });

    // Generate tokens one by one (like eval_mnist loop)
    for (uint32_t i = 0; i < max_new_tokens; ++i) {
        // Prepare input
        uint32_t seq_len = (uint32_t)generated.size();
        uint32_t start_idx = 0;

        // Truncate if too long
        if (seq_len > config.max_seq_len) {
            start_idx = seq_len - config.max_seq_len;
            seq_len = config.max_seq_len;
        }

        std::vector<float> input_data(seq_len);
        for (uint32_t j = 0; j < seq_len; ++j) {
            input_data[j] = (float)generated[start_idx + j];
        }

        Tensor inputTensor = Tensor(1, seq_len).set(input_data);

        // Forward pass (like eval_mnist)
        std::vector<Tensor> outputs = gpt2Net(inputTensor);

        Tensor logits = outputs[0];  // Returns GPU tensor

        // Copy logits to CPU (reusing pre-allocated buffer)
        uint32_t logits_size = seq_len * config.vocab_size * sizeof(float);

        // Get GPU buffer BEFORE any other operations
        Buffer gpu_buffer = logits.buffer();

        // Copy
        device.newCommandBuffer(queue_compute)
            .begin()
            .copyBuffer(cpu_buffer, gpu_buffer)
            .end()
            .submit()
            .wait();

        // Extract logits
        std::vector<float> logits_data(seq_len * config.vocab_size);
        memcpy(logits_data.data(), cpu_buffer.map(), logits_size);
        cpu_buffer.unmap();

        // Get last token logits
        std::vector<float> last_token_logits(config.vocab_size);
        uint32_t last_token_offset = (seq_len - 1) * config.vocab_size;
        for (uint32_t j = 0; j < config.vocab_size; ++j) {
            last_token_logits[j] = logits_data[last_token_offset + j];
        }

        // Debug: Print logits statistics for first and second iteration
        if (i == 0 || i == 1) {
            // Calculate statistics
            float min_logit = *std::min_element(last_token_logits.begin(), last_token_logits.end());
            float max_logit = *std::max_element(last_token_logits.begin(), last_token_logits.end());
            float sum_logit = std::accumulate(last_token_logits.begin(), last_token_logits.end(), 0.0f);
            float mean_logit = sum_logit / last_token_logits.size();

            std::cout << "\n  [Debug - Standard Gen Token " << (i+1) << "] Logits statistics:" << std::endl;
            std::cout << "    Min: " << std::fixed << std::setprecision(2) << min_logit << std::endl;
            std::cout << "    Max: " << std::fixed << std::setprecision(2) << max_logit << std::endl;
            std::cout << "    Mean: " << std::fixed << std::setprecision(2) << mean_logit << std::endl;
            std::cout << "    Range: " << std::fixed << std::setprecision(2) << (max_logit - min_logit) << std::endl;

            // Top 5 logits
            std::vector<std::pair<float, int>> top_logits;
            for (size_t j = 0; j < last_token_logits.size(); ++j) {
                top_logits.push_back({last_token_logits[j], (int)j});
            }
            std::sort(top_logits.begin(), top_logits.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });

            std::cout << "    Top 5 logits: ";
            for (int k = 0; k < 5; ++k) {
                std::cout << "(token=" << top_logits[k].second << ", logit="
                          << std::fixed << std::setprecision(2) << top_logits[k].first << ") ";
            }
            std::cout << std::endl;

            // Bottom 5 logits
            if (i == 0) {
                std::cout << "    Bottom 5 logits: ";
                for (int k = (int)top_logits.size() - 5; k < (int)top_logits.size(); ++k) {
                    std::cout << "(token=" << top_logits[k].second << ", logit="
                              << std::fixed << std::setprecision(2) << top_logits[k].first << ") ";
                }
            }
            std::cout << "\n" << std::endl;
        }

        // Sample next token
        int next_token = sampleToken(last_token_logits, temperature, top_k);
        generated.push_back(next_token);

        std::cout << "  Generated " << (i + 1) << "/" << max_new_tokens << " tokens..." << std::endl;
    }

    std::cout << "  Generation complete! Total tokens: " << generated.size() << std::endl;
    return generated;
}

/**
 * Generate tokens autoregressively WITH KV CACHE (O(n) complexity)
 *
 * This function uses KV caching to avoid recomputing attention for previous tokens.
 * - Prompt phase: Process all prompt tokens at once, fill the cache
 * - Generation phase: Process one token at a time using the cache
 *
 * @param gpt2Net The GPT2 network
 * @param prompt_ids Initial token IDs
 * @param max_new_tokens Number of tokens to generate
 * @param temperature Sampling temperature (0.0 for greedy/deterministic)
 * @param top_k Top-k sampling parameter
 * @param seed Random seed (optional, -1 for random)
 * @return Generated token IDs (prompt + new tokens)
 */
inline std::vector<int> generate_gpt2_with_cache(
    GPT2Net& gpt2Net,
    const std::vector<int>& prompt_ids,
    uint32_t max_new_tokens,
    float temperature = 1.0f,
    int top_k = 0,
    int seed = -1,
    bool enable_profiling = false)
{
    const GPT2Config& config = gpt2Net.getConfig();
    Device& device = netGlobalDevice;

    // Set random seed if provided
    if (seed >= 0) {
        srand((unsigned int)seed);
        std::cout << "\nRandom seed set to: " << seed << std::endl;
    }

    std::vector<int> generated = prompt_ids;

    std::cout << "\nGenerating text WITH KV CACHE..." << std::endl;
    std::cout << "  Prompt length: " << prompt_ids.size() << " tokens" << std::endl;
    std::cout << "  Max new tokens: " << max_new_tokens << std::endl;
    std::cout << "  Temperature: " << temperature << (temperature == 0.0f ? " (greedy)" : "") << std::endl;

    // Enable KV cache
    gpt2Net.enableCache();
    gpt2Net.resetCache();

    // Pre-allocate CPU buffer for logits (reuse across iterations)
    uint32_t max_logits_size = config.max_seq_len * config.vocab_size * sizeof(float);
    Buffer cpu_buffer = device.createBuffer({
        .size = max_logits_size,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    });

    // ========================================================================
    // PROMPT PHASE: Process all prompt tokens at once, fill cache
    // ========================================================================
    std::cout << "\n[Prompt Phase] Processing " << prompt_ids.size() << " tokens..." << std::endl;

    {
        uint32_t prompt_len = (uint32_t)prompt_ids.size();

        // Truncate if too long
        if (prompt_len > config.max_seq_len) {
            prompt_len = config.max_seq_len;
            std::cout << "  Warning: Prompt truncated to " << prompt_len << " tokens" << std::endl;
        }

        std::vector<float> prompt_data(prompt_len);
        for (uint32_t j = 0; j < prompt_len; ++j) {
            prompt_data[j] = (float)prompt_ids[j];
        }

        Tensor promptTensor = Tensor(1, prompt_len).set(prompt_data);

        // Set position offset to 0 for prompt phase
        gpt2Net.setPositionOffset(0);

        // Forward pass - this will fill the cache
        std::vector<Tensor> outputs = gpt2Net(promptTensor);
        Tensor logits = outputs[0];

        // Update cache length
        gpt2Net.updateCacheLength(prompt_len);

        std::cout << "  ✓ Prompt processed. Cache filled with " << gpt2Net.getCacheLength() << " tokens" << std::endl;

        // Get last token logits for first generation
        uint32_t logits_size = prompt_len * config.vocab_size * sizeof(float);
        Buffer gpu_buffer = logits.buffer();

        device.newCommandBuffer(queue_compute)
            .begin()
            .copyBuffer(cpu_buffer, gpu_buffer)
            .end()
            .submit()
            .wait();

        std::vector<float> logits_data(prompt_len * config.vocab_size);
        memcpy(logits_data.data(), cpu_buffer.map(), logits_size);
        cpu_buffer.unmap();

        // Get last token logits
        std::vector<float> last_token_logits(config.vocab_size);
        uint32_t last_token_offset = (prompt_len - 1) * config.vocab_size;
        for (uint32_t j = 0; j < config.vocab_size; ++j) {
            last_token_logits[j] = logits_data[last_token_offset + j];
        }

        // Debug: Print logits statistics
        {
            float min_logit = *std::min_element(last_token_logits.begin(), last_token_logits.end());
            float max_logit = *std::max_element(last_token_logits.begin(), last_token_logits.end());
            float sum_logit = std::accumulate(last_token_logits.begin(), last_token_logits.end(), 0.0f);
            float mean_logit = sum_logit / last_token_logits.size();

            std::cout << "\n  [Debug - Cached] Logits statistics:" << std::endl;
            std::cout << "    Min: " << std::fixed << std::setprecision(2) << min_logit << std::endl;
            std::cout << "    Max: " << std::fixed << std::setprecision(2) << max_logit << std::endl;
            std::cout << "    Mean: " << std::fixed << std::setprecision(2) << mean_logit << std::endl;
            std::cout << "    Range: " << std::fixed << std::setprecision(2) << (max_logit - min_logit) << std::endl;

            // Top 5 logits
            std::vector<std::pair<float, int>> top_logits;
            for (size_t j = 0; j < last_token_logits.size(); ++j) {
                top_logits.push_back({last_token_logits[j], (int)j});
            }
            std::sort(top_logits.begin(), top_logits.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });

            std::cout << "    Top 5 logits: ";
            for (int k = 0; k < 5; ++k) {
                std::cout << "(token=" << top_logits[k].second << ", logit="
                          << std::fixed << std::setprecision(2) << top_logits[k].first << ") ";
            }
            std::cout << std::endl;

            // Bottom 5 logits
            std::cout << "    Bottom 5 logits: ";
            for (int k = (int)top_logits.size() - 5; k < (int)top_logits.size(); ++k) {
                std::cout << "(token=" << top_logits[k].second << ", logit="
                          << std::fixed << std::setprecision(2) << top_logits[k].first << ") ";
            }
            std::cout << "\n" << std::endl;
        }

        // Sample first generated token
        int next_token = sampleToken(last_token_logits, temperature, top_k);
        generated.push_back(next_token);

        std::cout << "  First token generated." << std::endl;
    }

    // ========================================================================
    // GENERATION PHASE: Generate remaining tokens one by one using cache
    // ========================================================================
    std::cout << "\n[Generation Phase] Generating remaining tokens with cache..." << std::endl;

    // Profiling counters
    double total_forward_ms = 0.0;
    double total_wait_ms = 0.0;
    double total_memcpy_ms = 0.0;
    double total_sampling_ms = 0.0;

    for (uint32_t i = 1; i < max_new_tokens; ++i) {
        // Process only the last token (cache has all previous tokens)
        int last_token = generated.back();
        std::vector<float> input_data = {(float)last_token};

        Tensor inputTensor = Tensor(1, 1).set(input_data);  // Single token!

        // Set position offset to cache length (absolute position in sequence)
        uint32_t current_cache_len = gpt2Net.getCacheLength();
        gpt2Net.setPositionOffset(current_cache_len);

        // 1. Forward pass - uses cache, much faster!
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<Tensor> outputs = gpt2Net(inputTensor);
        Tensor logits = outputs[0];
        auto t1 = std::chrono::high_resolution_clock::now();
        total_forward_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Update cache length
        gpt2Net.updateCacheLength(1);

        // 2. Copy logits to CPU
        uint32_t logits_size = 1 * config.vocab_size * sizeof(float);
        Buffer gpu_buffer = logits.buffer();

        auto t2 = std::chrono::high_resolution_clock::now();
        device.newCommandBuffer(queue_compute)
            .begin()
            .copyBuffer(cpu_buffer, gpu_buffer)
            .end()
            .submit()
            .wait();
        auto t3 = std::chrono::high_resolution_clock::now();
        total_wait_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();

        // 3. Memcpy GPU → CPU
        auto t4 = std::chrono::high_resolution_clock::now();
        std::vector<float> last_token_logits(config.vocab_size);
        memcpy(last_token_logits.data(), cpu_buffer.map(), logits_size);
        cpu_buffer.unmap();
        auto t5 = std::chrono::high_resolution_clock::now();
        total_memcpy_ms += std::chrono::duration<double, std::milli>(t5 - t4).count();

        // Debug: Print logits for second generation token
        if (i == 1) {
            float min_logit = *std::min_element(last_token_logits.begin(), last_token_logits.end());
            float max_logit = *std::max_element(last_token_logits.begin(), last_token_logits.end());
            float sum_logit = std::accumulate(last_token_logits.begin(), last_token_logits.end(), 0.0f);
            float mean_logit = sum_logit / last_token_logits.size();

            std::cout << "\n  [Debug - Cached Gen Token 2] Logits statistics:" << std::endl;
            std::cout << "    Min: " << std::fixed << std::setprecision(2) << min_logit << std::endl;
            std::cout << "    Max: " << std::fixed << std::setprecision(2) << max_logit << std::endl;
            std::cout << "    Mean: " << std::fixed << std::setprecision(2) << mean_logit << std::endl;
            std::cout << "    Range: " << std::fixed << std::setprecision(2) << (max_logit - min_logit) << std::endl;

            std::vector<std::pair<float, int>> top_logits;
            for (size_t j = 0; j < last_token_logits.size(); ++j) {
                top_logits.push_back({last_token_logits[j], (int)j});
            }
            std::sort(top_logits.begin(), top_logits.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });

            std::cout << "    Top 5 logits: ";
            for (int k = 0; k < 5; ++k) {
                std::cout << "(token=" << top_logits[k].second << ", logit="
                          << std::fixed << std::setprecision(2) << top_logits[k].first << ") ";
            }
            std::cout << "\n" << std::endl;
        }

        // 4. Sample next token
        auto t6 = std::chrono::high_resolution_clock::now();
        int next_token = sampleToken(last_token_logits, temperature, top_k);
        generated.push_back(next_token);
        auto t7 = std::chrono::high_resolution_clock::now();
        total_sampling_ms += std::chrono::duration<double, std::milli>(t7 - t6).count();

        // Print per-token timing for analysis (every 50 tokens)
        if (enable_profiling && ((i + 1) % 50 == 0 || i == 1)) {
            double token_time = std::chrono::duration<double, std::milli>(t7 - t0).count();
            std::cout << "  Token " << (i + 1) << "/" << max_new_tokens
                      << " (cache_len=" << gpt2Net.getCacheLength() << ")"
                      << " - Time: " << std::fixed << std::setprecision(2) << token_time << " ms" << std::endl;
        } else {
            std::cout << "  Generated " << (i + 1) << "/" << max_new_tokens << " tokens (cache_len=" << gpt2Net.getCacheLength() << ")..." << std::endl;
        }
    }

    std::cout << "  Generation complete! Total tokens: " << generated.size() << std::endl;
    std::cout << "  Final cache length: " << gpt2Net.getCacheLength() << std::endl;

    // Print profiling results (only if enabled)
    if (enable_profiling) {
        uint32_t num_profile_tokens = max_new_tokens - 1;  // Excluding first token (prompt phase)
        std::cout << "\n╔════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║              Performance Breakdown                     ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════╝" << std::endl;
        std::cout << "Total tokens profiled: " << num_profile_tokens << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\n1. Forward Pass (GPU execution):" << std::endl;
        std::cout << "   Total: " << total_forward_ms << " ms" << std::endl;
        std::cout << "   Avg:   " << (total_forward_ms / num_profile_tokens) << " ms/token" << std::endl;
        std::cout << "   %:     " << (total_forward_ms / (total_forward_ms + total_wait_ms + total_memcpy_ms + total_sampling_ms) * 100) << "%" << std::endl;

        std::cout << "\n2. GPU Sync (.wait()):" << std::endl;
        std::cout << "   Total: " << total_wait_ms << " ms" << std::endl;
        std::cout << "   Avg:   " << (total_wait_ms / num_profile_tokens) << " ms/token" << std::endl;
        std::cout << "   %:     " << (total_wait_ms / (total_forward_ms + total_wait_ms + total_memcpy_ms + total_sampling_ms) * 100) << "%" << std::endl;

        std::cout << "\n3. Memcpy (GPU→CPU):" << std::endl;
        std::cout << "   Total: " << total_memcpy_ms << " ms" << std::endl;
        std::cout << "   Avg:   " << (total_memcpy_ms / num_profile_tokens) << " ms/token" << std::endl;
        std::cout << "   %:     " << (total_memcpy_ms / (total_forward_ms + total_wait_ms + total_memcpy_ms + total_sampling_ms) * 100) << "%" << std::endl;

        std::cout << "\n4. CPU Sampling (softmax+top-k):" << std::endl;
        std::cout << "   Total: " << total_sampling_ms << " ms" << std::endl;
        std::cout << "   Avg:   " << (total_sampling_ms / num_profile_tokens) << " ms/token" << std::endl;
        std::cout << "   %:     " << (total_sampling_ms / (total_forward_ms + total_wait_ms + total_memcpy_ms + total_sampling_ms) * 100) << "%" << std::endl;

        double total_profiled = total_forward_ms + total_wait_ms + total_memcpy_ms + total_sampling_ms;
        std::cout << "\nTotal profiled time: " << total_profiled << " ms" << std::endl;
        std::cout << "════════════════════════════════════════════════════════\n" << std::endl;
    }

    return generated;
}

#endif // GPT2_GENERATION_H

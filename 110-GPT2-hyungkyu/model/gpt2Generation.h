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

        // Debug: Print logits statistics for first iteration
        if (i == 0) {
            // Calculate statistics
            float min_logit = *std::min_element(last_token_logits.begin(), last_token_logits.end());
            float max_logit = *std::max_element(last_token_logits.begin(), last_token_logits.end());
            float sum_logit = std::accumulate(last_token_logits.begin(), last_token_logits.end(), 0.0f);
            float mean_logit = sum_logit / last_token_logits.size();

            std::cout << "\n  [Debug] Logits statistics:" << std::endl;
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

        // Sample next token
        int next_token = sampleToken(last_token_logits, temperature, top_k);
        generated.push_back(next_token);

        std::cout << "  Generated " << (i + 1) << "/" << max_new_tokens << " tokens..." << std::endl;
    }

    std::cout << "  Generation complete! Total tokens: " << generated.size() << std::endl;
    return generated;
}

#endif // GPT2_GENERATION_H

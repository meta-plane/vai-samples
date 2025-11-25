#ifndef GPT2_INFERENCE_H
#define GPT2_INFERENCE_H

#include "gpt2Net.h"
#include "gpt2Generation.h"
#include "../tokenizer/bpeTokenizer.h"
#include <string>
#include <vector>

/**
 * GPT2 Inference Module
 *
 * Handles loading pretrained weights and running text generation.
 * Separate from unit tests for clearer organization.
 */

namespace GPT2Inference {

/**
 * Inference Configuration
 */
struct InferenceConfig {
    // NOTE: Paths are relative to 110-GPT2-hyungkyu directory
    // Run executable from: cd 110-GPT2-hyungkyu && ../bin/debug/gpt2-inference.exe
    std::string weights_dir = "assets/weights/124M/";
    std::string vocab_file = "assets/vocab.json";
    std::string merges_file = "assets/merges.txt";

    float temperature = 0.8f;
    int top_k = 40;
    int seed = 42;
    bool use_cache = true;
};

/**
 * Generation Result
 */
struct GenerationResult {
    std::string generated_text;
    std::vector<int> token_ids;
    uint32_t num_tokens_generated;
    double generation_time_sec;
    double tokens_per_sec;
    bool success = false;
    std::string error_message;
};

/**
 * Load pretrained GPT-2 model
 *
 * @param config Inference configuration
 * @return Initialized GPT2Net with loaded weights, or nullptr on failure
 */
GPT2Net* loadPretrainedModel(const InferenceConfig& config = InferenceConfig());

/**
 * Generate text from a prompt
 *
 * @param model GPT2Net with loaded weights
 * @param tokenizer BPE tokenizer
 * @param prompt Input text prompt
 * @param max_tokens Maximum number of tokens to generate
 * @param config Inference configuration
 * @return Generation result with text, timing, and status
 */
GenerationResult generate(
    GPT2Net& model,
    BPETokenizer& tokenizer,
    const std::string& prompt,
    uint32_t max_tokens,
    const InferenceConfig& config = InferenceConfig()
);

/**
 * Compare standard vs cached generation performance
 *
 * @param prompt Input text prompt
 * @param max_tokens Maximum number of tokens to generate
 * @param config Inference configuration
 */
void compareGenerationModes(
    const std::string& prompt,
    uint32_t max_tokens,
    const InferenceConfig& config = InferenceConfig()
);

/**
 * Run interactive text generation
 *
 * Allows user to input prompts and generate text interactively.
 * Type 'quit' to exit.
 *
 * @param config Inference configuration
 */
void runInteractive(const InferenceConfig& config = InferenceConfig());

} // namespace GPT2Inference

#endif // GPT2_INFERENCE_H

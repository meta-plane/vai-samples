/**
 * GPT-2 Main Entry Point
 *
 * This is the main CLI for GPT-2 inference and testing.
 *
 * Organization:
 * - test/: Unit tests for components (use runAllTests executable)
 * - model/inference: Text generation with pretrained weights (this file)
 */

#include "model/inference.h"
#include "core/globalContext.h"
#include "core/vulkanApp.h"
#include <iostream>
#include <string>

using namespace vk;

// ============================================================================
// Usage and CLI
// ============================================================================

void printUsage(const char* program_name) {
    std::cout << "╔════════════════════════════════════════╗" << std::endl;
    std::cout << "║  GPT-2 Text Generation CLI             ║" << std::endl;
    std::cout << "╚════════════════════════════════════════╝\n" << std::endl;

    std::cout << "Usage: " << program_name << " [MODE] [OPTIONS]" << std::endl;

    std::cout << "\n═══ Modes ═══" << std::endl;
    std::cout << "  generate [prompt] [tokens]   Generate text from prompt (default mode)" << std::endl;
    std::cout << "  compare [prompt] [tokens]    Compare standard vs cached generation" << std::endl;
    std::cout << "  interactive                  Interactive text generation" << std::endl;

    std::cout << "\n═══ Options ═══" << std::endl;
    std::cout << "  -h, --help                   Show this help message" << std::endl;
    std::cout << "  --no-cache                   Disable KV cache (slower)" << std::endl;
    std::cout << "  --profile                    Enable performance profiling output" << std::endl;
    std::cout << "  --temperature <float>        Sampling temperature (default: 0.8)" << std::endl;
    std::cout << "  --top-k <int>                Top-k sampling (default: 40)" << std::endl;
    std::cout << "  --seed <int>                 Random seed for reproducibility (default: 42)" << std::endl;

    std::cout << "\n═══ Arguments ═══" << std::endl;
    std::cout << "  prompt                       Text prompt for generation" << std::endl;
    std::cout << "                               (default: \"The future of artificial intelligence is\")" << std::endl;
    std::cout << "  tokens                       Maximum tokens to generate (1-1000)" << std::endl;
    std::cout << "                               (default: 25)" << std::endl;

    std::cout << "\n═══ Examples ═══" << std::endl;
    std::cout << "  # Basic generation (KV cache enabled by default)" << std::endl;
    std::cout << "  " << program_name << std::endl;
    std::cout << "  " << program_name << " generate \"Once upon a time\"" << std::endl;
    std::cout << "  " << program_name << " generate \"Once upon a time\" 50" << std::endl;
    std::cout << "\n  # Without KV cache (slower)" << std::endl;
    std::cout << "  " << program_name << " --no-cache generate \"Hello\" 30" << std::endl;
    std::cout << "\n  # Compare performance" << std::endl;
    std::cout << "  " << program_name << " compare \"Hello, I'm a language model,\" 50" << std::endl;
    std::cout << "\n  # Interactive mode" << std::endl;
    std::cout << "  " << program_name << " interactive" << std::endl;
    std::cout << "\n  # Advanced: Temperature and top-k" << std::endl;
    std::cout << "  " << program_name << " --temperature 1.0 --top-k 50 generate \"Hello\" 30" << std::endl;
    std::cout << "\n  # Performance profiling" << std::endl;
    std::cout << "  " << program_name << " --profile generate \"Hello\" 200" << std::endl;

    std::cout << "\n═══ Notes ═══" << std::endl;
    std::cout << "  • KV cache is ENABLED by default for faster generation" << std::endl;
    std::cout << "  • Requires pretrained weights in assets/weights/124M/" << std::endl;
    std::cout << "  • Run utils/download_gpt2_weights.py to download weights" << std::endl;
    std::cout << "  • For unit tests, use the 'runAllTests' executable" << std::endl;
    std::cout << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    try {
        // Pre-compile all shaders at startup to eliminate runtime compilation latency
        loadAllShaders();

        // Default values
        std::string mode = "generate";
        std::string prompt = "The future of artificial intelligence is";
        uint32_t max_tokens = 25;

        GPT2Inference::InferenceConfig config;
        config.use_cache = true;
        config.temperature = 0.8f;
        config.top_k = 40;
        config.seed = 42;

        // Parse arguments
        int i = 1;
        while (i < argc) {
            std::string arg = argv[i];

            if (arg == "-h" || arg == "--help") {
                printUsage(argv[0]);
                return 0;
            }
            else if (arg == "--no-cache") {
                config.use_cache = false;
                i++;
            }
            else if (arg == "--profile") {
                config.enable_profiling = true;
                i++;
            }
            else if (arg == "--temperature") {
                if (i + 1 < argc) {
                    config.temperature = std::stof(argv[i + 1]);
                    i += 2;
                } else {
                    std::cerr << "Error: --temperature requires a value" << std::endl;
                    return 1;
                }
            }
            else if (arg == "--top-k") {
                if (i + 1 < argc) {
                    config.top_k = std::stoi(argv[i + 1]);
                    i += 2;
                } else {
                    std::cerr << "Error: --top-k requires a value" << std::endl;
                    return 1;
                }
            }
            else if (arg == "--seed") {
                if (i + 1 < argc) {
                    config.seed = std::stoi(argv[i + 1]);
                    i += 2;
                } else {
                    std::cerr << "Error: --seed requires a value" << std::endl;
                    return 1;
                }
            }
            else if (arg == "generate" || arg == "compare" || arg == "interactive") {
                mode = arg;
                i++;

                // Parse prompt and max_tokens for generate/compare modes
                if (mode != "interactive") {
                    if (i < argc && argv[i][0] != '-') {
                        prompt = argv[i];
                        i++;
                    }
                    if (i < argc && argv[i][0] != '-') {
                        max_tokens = std::stoi(argv[i]);
                        i++;
                    }
                }
            }
            else {
                std::cerr << "Unknown argument: " << arg << std::endl;
                std::cerr << "Use --help for usage information" << std::endl;
                return 1;
            }
        }

        // Vulkan is automatically initialized via global context
        std::cout << "Starting GPT-2 inference...\n" << std::endl;

        // Execute requested mode
        if (mode == "generate") {
            // Load model
            GPT2Net* model = GPT2Inference::loadPretrainedModel(config);
            if (!model) {
                return 1;
            }

            // Load tokenizer
            std::cout << "Loading tokenizer..." << std::endl;
            BPETokenizer tokenizer(config.vocab_file, config.merges_file);
            std::cout << "✓ Tokenizer loaded\n" << std::endl;

            // Generate
            GPT2Inference::generate(*model, tokenizer, prompt, max_tokens, config);

            delete model;
        }
        else if (mode == "compare") {
            GPT2Inference::compareGenerationModes(prompt, max_tokens, config);
        }
        else if (mode == "interactive") {
            GPT2Inference::runInteractive(config);
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n✗ Fatal error: " << e.what() << std::endl;
        return 1;
    }
}

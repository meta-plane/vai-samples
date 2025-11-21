#include <string>
#include <iostream>

// Component tests
void tokenizerTest();
void dataLoaderTest();
void embeddingNodeTest();
void attentionNodeTest();
void transformerNodeTest();
void testGPT2();
void testGPT2Generation();

// KV Cache tests
void testKVCache();
void testKVCacheIntegration();

// Generation tests
void testGPT2Pretrained(const std::string& prompt, uint32_t max_tokens);

// ============================================================================
// Main Test Functions
// ============================================================================

/**
 * Run all basic component tests
 * Tests individual layers: tokenizer, embedding, attention, transformer, etc.
 */
void runBasicTests()
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "Running Basic Component Tests" << std::endl;
    std::cout << "========================================\n" << std::endl;

    tokenizerTest();
    dataLoaderTest();
    embeddingNodeTest();
    attentionNodeTest();
    transformerNodeTest();
    testGPT2();
    testGPT2Generation();
}

/**
 * Test standard generation (without KV cache)
 */
void testStandardGeneration(const std::string& prompt, uint32_t max_tokens)
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "Standard Generation Test (No Cache)" << std::endl;
    std::cout << "========================================\n" << std::endl;

    testGPT2Pretrained(prompt, max_tokens);
}

/**
 * Test KV cache data structure
 */
void testKVCacheStructure()
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "KV Cache Structure Test" << std::endl;
    std::cout << "========================================\n" << std::endl;

    testKVCache();
}

/**
 * Test KV cache integrated with AttentionNode
 * (TODO: Implement after Phase 2 completion)
 */
void testKVCacheWithAttention()
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "KV Cache + Attention Integration Test" << std::endl;
    std::cout << "========================================\n" << std::endl;

    std::cout << "⚠ This test is not yet implemented." << std::endl;
    std::cout << "  Status: Waiting for Phase 2 (AttentionNode modification) completion" << std::endl;
    std::cout << "\nCurrent progress:" << std::endl;
    std::cout << "  ✓ Phase 1: KV Cache data structure (COMPLETE)" << std::endl;
    std::cout << "  ⧗ Phase 2: AttentionNode integration (IN PROGRESS)" << std::endl;
    std::cout << "    - Header modifications: DONE" << std::endl;
    std::cout << "    - Concatenation shader: TODO" << std::endl;
    std::cout << "    - Implementation: TODO" << std::endl;
    std::cout << "  ☐ Phase 3: GPT2Net integration (TODO)" << std::endl;
    std::cout << "  ☐ Phase 4: Generation loop optimization (TODO)" << std::endl;
    std::cout << "\nRun with --test-basic to test KV cache structure only.\n" << std::endl;
}

/**
 * Test full generation with KV cache
 * (TODO: Implement after all phases completion)
 */
void testCachedGeneration(const std::string& prompt, uint32_t max_tokens)
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "Cached Generation Test (With KV Cache)" << std::endl;
    std::cout << "========================================\n" << std::endl;

    std::cout << "⚠ KV cache generation is not yet implemented." << std::endl;
    std::cout << "  Falling back to standard generation for now.\n" << std::endl;

    testStandardGeneration(prompt, max_tokens);
}

/**
 * Benchmark: Compare standard vs cached generation
 * (TODO: Implement after all phases completion)
 */
void benchmarkGenerationMethods(const std::string& prompt, uint32_t max_tokens)
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "Generation Benchmark (Standard vs Cached)" << std::endl;
    std::cout << "========================================\n" << std::endl;

    std::cout << "This will compare:" << std::endl;
    std::cout << "  1. Standard generation (O(n²) complexity)" << std::endl;
    std::cout << "  2. Cached generation (O(n) complexity with KV cache)" << std::endl;
    std::cout << "\n⚠ KV cache not yet implemented. Skipping benchmark.\n" << std::endl;
}

// ============================================================================
// Usage and Main
// ============================================================================

void printUsage(const char* program_name)
{
    std::cout << "Usage: " << program_name << " [OPTIONS] [prompt] [max_tokens]" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  -h, --help              Show this help message" << std::endl;
    std::cout << "  --test-basic            Run basic component tests" << std::endl;
    std::cout << "  --test-kvcache          Test KV cache data structure" << std::endl;
    std::cout << "  --test-attention        Test KV cache with attention (TODO)" << std::endl;
    std::cout << "  --test-cached           Test generation with KV cache (TODO)" << std::endl;
    std::cout << "  --benchmark             Benchmark standard vs cached generation (TODO)" << std::endl;
    std::cout << "\nArguments:" << std::endl;
    std::cout << "  prompt                  Text prompt for generation" << std::endl;
    std::cout << "                          (default: \"The future of artificial intelligence is\")" << std::endl;
    std::cout << "  max_tokens              Maximum tokens to generate (1-500)" << std::endl;
    std::cout << "                          (default: 25)" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << program_name << std::endl;
    std::cout << "  " << program_name << " \"Once upon a time\"" << std::endl;
    std::cout << "  " << program_name << " \"Once upon a time\" 50" << std::endl;
    std::cout << "  " << program_name << " --test-kvcache" << std::endl;
    std::cout << "  " << program_name << " --test-basic" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[])
{
    // Default values
    std::string prompt = "The future of artificial intelligence is";
    uint32_t max_tokens = 25;

    // Parse command-line arguments
    if (argc > 1) {
        std::string arg1 = argv[1];

        // Check for flags
        if (arg1 == "-h" || arg1 == "--help") {
            printUsage(argv[0]);
            return 0;
        }

        if (arg1 == "--test-basic") {
            runBasicTests();
            return 0;
        }

        if (arg1 == "--test-kvcache") {
            testKVCacheStructure();
            return 0;
        }

        if (arg1 == "--test-attention") {
            testKVCacheWithAttention();
            return 0;
        }

        if (arg1 == "--test-cached") {
            // Parse optional prompt and max_tokens
            if (argc > 2) prompt = argv[2];
            if (argc > 3) {
                try {
                    max_tokens = std::stoi(argv[3]);
                } catch (...) {
                    std::cerr << "Invalid max_tokens. Using default (25)." << std::endl;
                }
            }
            testCachedGeneration(prompt, max_tokens);
            return 0;
        }

        if (arg1 == "--benchmark") {
            if (argc > 2) prompt = argv[2];
            if (argc > 3) {
                try {
                    max_tokens = std::stoi(argv[3]);
                } catch (...) {
                    std::cerr << "Invalid max_tokens. Using default (25)." << std::endl;
                }
            }
            benchmarkGenerationMethods(prompt, max_tokens);
            return 0;
        }

        // No flag, treat as prompt
        prompt = argv[1];
    }

    if (argc > 2) {
        // Second argument is max_tokens
        try {
            max_tokens = std::stoi(argv[2]);
            if (max_tokens == 0 || max_tokens > 500) {
                std::cerr << "Warning: max_tokens should be between 1 and 500. Using default (25)." << std::endl;
                max_tokens = 25;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid max_tokens value. Using default (25)." << std::endl;
            max_tokens = 25;
        }
    }

    // Default: Run standard generation
    testStandardGeneration(prompt, max_tokens);

    return 0;
}

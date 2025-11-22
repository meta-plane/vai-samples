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
void testGPT2Pretrained(const std::string& prompt, uint32_t max_tokens, bool use_cache = true);
void testGPT2WithCache(const std::string& prompt, uint32_t max_tokens);

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
 * Run text generation (with optional KV cache)
 */
void runTextGeneration(const std::string& prompt, uint32_t max_tokens, bool use_cache = true)
{
    testGPT2Pretrained(prompt, max_tokens, use_cache);
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
 */
void testKVCacheWithAttention()
{
    testKVCacheIntegration();
}

/**
 * Test full generation with KV cache
 */
void testCachedGeneration(const std::string& prompt, uint32_t max_tokens)
{
    testGPT2WithCache(prompt, max_tokens);
}

/**
 * Benchmark: Compare standard vs cached generation
 */
void benchmarkGenerationMethods(const std::string& prompt, uint32_t max_tokens)
{
    // testGPT2WithCache already does the comparison
    testGPT2WithCache(prompt, max_tokens);
}

// ============================================================================
// Usage and Main
// ============================================================================

void printUsage(const char* program_name)
{
    std::cout << "Usage: " << program_name << " [OPTIONS] [prompt] [max_tokens]" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  -h, --help              Show this help message" << std::endl;
    std::cout << "  --no-cache              Disable KV cache (slower but uses less memory)" << std::endl;
    std::cout << "  --test-basic            Run basic component tests" << std::endl;
    std::cout << "  --test-kvcache          Test KV cache data structure" << std::endl;
    std::cout << "  --test-attention        Test KV cache with attention" << std::endl;
    std::cout << "  --test-cached           Test generation with KV cache" << std::endl;
    std::cout << "  --benchmark             Benchmark standard vs cached generation" << std::endl;
    std::cout << "\nArguments:" << std::endl;
    std::cout << "  prompt                  Text prompt for generation" << std::endl;
    std::cout << "                          (default: \"The future of artificial intelligence is\")" << std::endl;
    std::cout << "  max_tokens              Maximum tokens to generate (1-1000)" << std::endl;
    std::cout << "                          (default: 25)" << std::endl;
    std::cout << "\nNote:" << std::endl;
    std::cout << "  KV cache is ENABLED by default for faster generation." << std::endl;
    std::cout << "  Use --no-cache to disable it." << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << program_name << "                                  # Use cache (fast)" << std::endl;
    std::cout << "  " << program_name << " \"Once upon a time\"               # Use cache (fast)" << std::endl;
    std::cout << "  " << program_name << " \"Once upon a time\" 50            # Use cache (fast)" << std::endl;
    std::cout << "  " << program_name << " --no-cache \"Once upon a time\" 50  # No cache (slow)" << std::endl;
    std::cout << "  " << program_name << " --test-kvcache" << std::endl;
    std::cout << "  " << program_name << " --benchmark \"Hello\" 50" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[])
{
    // Default values
    std::string prompt = "The future of artificial intelligence is";
    uint32_t max_tokens = 25;
    bool use_cache = true;  // KV cache enabled by default

    // Parse command-line arguments
    int arg_offset = 1;
    if (argc > 1) {
        std::string arg1 = argv[1];

        // Check for flags
        if (arg1 == "-h" || arg1 == "--help") {
            printUsage(argv[0]);
            return 0;
        }

        // Check for --no-cache flag
        if (arg1 == "--no-cache") {
            use_cache = false;
            arg_offset = 2;  // Skip --no-cache flag
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

        // Parse prompt (might be after --no-cache flag)
        if (argc > arg_offset) {
            prompt = argv[arg_offset];
        }
    }

    // Parse max_tokens
    if (argc > arg_offset + 1) {
        try {
            max_tokens = std::stoi(argv[arg_offset + 1]);
            if (max_tokens == 0 || max_tokens > 1000) {
                std::cerr << "Warning: max_tokens should be between 1 and 1000. Using default (25)." << std::endl;
                max_tokens = 25;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid max_tokens value. Using default (25)." << std::endl;
            max_tokens = 25;
        }
    }

    // Default: Run text generation with KV cache (unless --no-cache specified)
    runTextGeneration(prompt, max_tokens, use_cache);

    return 0;
}

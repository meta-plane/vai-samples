#include <string>
#include <iostream>

void tokenizerTest();
void dataLoaderTest();
void embeddingNodeTest();
void attentionNodeTest();
void transformerNodeTest();
void testGPT2();
void testGPT2Generation();
void testGPT2Pretrained(const std::string& prompt = "The future of artificial intelligence is", uint32_t max_tokens = 25);
void testKVCache();

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [prompt] [max_tokens]" << std::endl;
    std::cout << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  prompt      : Text prompt for generation (default: \"The future of artificial intelligence is\")" << std::endl;
    std::cout << "  max_tokens  : Maximum number of tokens to generate (default: 25)" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << std::endl;
    std::cout << "  " << program_name << " \"Once upon a time\"" << std::endl;
    std::cout << "  " << program_name << " \"Once upon a time\" 15" << std::endl;
    std::cout << std::endl;
}

// Run all basic component tests
void runBasicTests()
{
    // Run tokenizer tests
    tokenizerTest();

    // Run dataloader tests
    dataLoaderTest();

    // Run embedding node tests (Vulkan version)
    embeddingNodeTest();

    // Run attention node tests (Multi-Head Attention)
    attentionNodeTest();

    // Run transformer node tests (LayerNorm, GELU, FeedForward)
    transformerNodeTest();

    // Run GPT-2 complete test suite
    testGPT2();

    // Run GPT-2 text generation test (random weights)
    testGPT2Generation();
}

int main(int argc, char* argv[])
{
    // Default values
    std::string prompt = "The future of artificial intelligence is";
    uint32_t max_tokens = 25;

    // Parse command-line arguments
    if (argc > 1) {
        std::string arg1 = argv[1];

        // Check for help flag
        if (arg1 == "-h" || arg1 == "--help") {
            printUsage(argv[0]);
            return 0;
        }

        // Check for test flag
        if (arg1 == "--test-kvcache") {
            testKVCache();
            return 0;
        }

        // First argument is the prompt
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

    // Print configuration
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Prompt: \"" << prompt << "\"" << std::endl;
    std::cout << "  Max tokens: " << max_tokens << std::endl;
    std::cout << std::endl;

    // Run pretrained weights test
    testGPT2Pretrained(prompt, max_tokens);

    return 0;
}


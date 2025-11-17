#include "gpt2.h"
#include "../core/error.h"
#include "../tokenizer/bpeTokenizer.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace vk;

// Global device and descriptor pool (defined in embeddingNodeTest.cpp)
extern Device netGlobalDevice;
extern DescriptorPool gDestSetPool;

void testGPT2SmallForward()
{
    std::cout << "\n========== Test: GPT-2 Small Forward Pass ===========" << std::endl;

    // Use a smaller config for testing
    GPT2Config config{
        .vocab_size = 50257,
        .max_seq_len = 1024,
        .d_model = 64,      // Smaller for faster testing
        .num_heads = 4,     // Smaller for faster testing
        .num_layers = 2,    // Only 2 layers for testing
        .dropout = 0.0f
    };

    std::cout << "Config:" << std::endl;
    std::cout << "  Vocab size: " << config.vocab_size << std::endl;
    std::cout << "  Max seq len: " << config.max_seq_len << std::endl;
    std::cout << "  d_model: " << config.d_model << std::endl;
    std::cout << "  Num heads: " << config.num_heads << std::endl;
    std::cout << "  Num layers: " << config.num_layers << std::endl;

    std::cout << "\nCreating GPT-2 model..." << std::endl;
    GPT2 model(netGlobalDevice, gDestSetPool, config);

    // Create dummy input: [batch=2, seq_len=4]
    const uint32_t batch_size = 2;
    const uint32_t seq_len = 4;

    std::vector<int> input_ids_int = {
        15496, 11, 995, 0,      // Batch 0: "Hello, world!"
        40, 1101, 4673, 13      // Batch 1: "I'm learning."
    };

    // Convert to float for Tensor
    std::vector<float> input_ids(input_ids_int.begin(), input_ids_int.end());

    Tensor inputTensor = Tensor(batch_size, seq_len);
    inputTensor.set(input_ids);

    std::cout << "Input shape: [" << batch_size << ", " << seq_len << "]" << std::endl;
    std::cout << "Input token IDs:" << std::endl;
    std::cout << "  Batch 0: ";
    for (int i = 0; i < seq_len; ++i) {
        std::cout << input_ids_int[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "  Batch 1: ";
    for (int i = seq_len; i < 2 * seq_len; ++i) {
        std::cout << input_ids_int[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "\nRunning forward pass..." << std::endl;
    std::cout << "  (This may take a moment...)" << std::endl;

    try {
        Tensor output = model.forward(inputTensor);
        std::cout << "  Forward pass completed successfully!" << std::endl;

        std::cout << "Output shape: [" << output.shape()[0] << ", "
                  << output.shape()[1] << ", " << output.shape()[2] << "]" << std::endl;
        std::cout << "Expected: [" << batch_size << ", " << seq_len << ", " << config.d_model << "]" << std::endl;

        // Copy output back to CPU for inspection
        Buffer outBuffer = netGlobalDevice.createBuffer({
        .size = batch_size * seq_len * config.d_model * sizeof(float),
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });

    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuffer, output.buffer())
        .end()
        .submit()
        .wait();

    float* data = (float*)outBuffer.map();

    // Check basic sanity
    std::cout << "\nVerifying output..." << std::endl;

    bool has_nan = false;
    bool has_inf = false;
    float min_val = data[0];
    float max_val = data[0];

    for (size_t i = 0; i < batch_size * seq_len * config.d_model; ++i) {
        if (std::isnan(data[i])) has_nan = true;
        if (std::isinf(data[i])) has_inf = true;
        min_val = std::min(min_val, data[i]);
        max_val = std::max(max_val, data[i]);
    }

    std::cout << "  Has NaN: " << (has_nan ? "YES ✗" : "NO ✓") << std::endl;
    std::cout << "  Has Inf: " << (has_inf ? "YES ✗" : "NO ✓") << std::endl;
    std::cout << "  Value range: [" << std::fixed << std::setprecision(4)
              << min_val << ", " << max_val << "]" << std::endl;

    // Print sample outputs for first token
    std::cout << "\n  First token output (first 10 values):" << std::endl;
    std::cout << "    ";
    for (int i = 0; i < std::min(10, (int)config.d_model); ++i) {
        std::cout << std::fixed << std::setprecision(4) << data[i] << " ";
    }
    std::cout << std::endl;

        if (!has_nan && !has_inf) {
            std::cout << "\n✓ GPT-2 forward pass PASSED - basic sanity checks OK" << std::endl;
        } else {
            std::cout << "\n✗ GPT-2 forward pass FAILED - numerical issues detected" << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cout << "\n✗ Forward pass failed with exception: " << e.what() << std::endl;
    }
}

void testGPT2LoadWeights()
{
    std::cout << "\n========== Test: GPT-2 Load Pretrained Weights ===========" << std::endl;

    // Use full GPT-2 small config
    GPT2Config config = GPT2SmallConfig();

    std::cout << "Config:" << std::endl;
    std::cout << "  Vocab size: " << config.vocab_size << std::endl;
    std::cout << "  Max seq len: " << config.max_seq_len << std::endl;
    std::cout << "  d_model: " << config.d_model << std::endl;
    std::cout << "  Num heads: " << config.num_heads << std::endl;
    std::cout << "  Num layers: " << config.num_layers << std::endl;

    std::cout << "\nCreating GPT-2 model..." << std::endl;
    GPT2 model(netGlobalDevice, gDestSetPool, config);

    // Try to load weights
    std::string weights_file = "weights/gpt2_weights.bin";
    std::cout << "\nAttempting to load weights from: " << weights_file << std::endl;

    try {
        model.loadWeights(weights_file);
        std::cout << "\n✓ Successfully loaded pretrained GPT-2 weights!" << std::endl;

        // Test with real text
        std::cout << "\nTesting with sample input..." << std::endl;
        std::vector<int> input_ids_int = {
            15496, 11, 995, 0      // "Hello, world!"
        };

        std::vector<float> input_ids(input_ids_int.begin(), input_ids_int.end());
        Tensor inputTensor = Tensor(1, 4).set(input_ids);

        Tensor output = model.forward(inputTensor);
        std::cout << "  Forward pass with pretrained weights completed!" << std::endl;
        std::cout << "  Output shape: [" << output.shape()[0] << ", "
                  << output.shape()[1] << ", " << output.shape()[2] << "]" << std::endl;

    }
    catch (const std::exception& e) {
        std::cout << "\n⚠ Could not load weights: " << e.what() << std::endl;
        std::cout << "  This is expected if you haven't downloaded weights yet." << std::endl;
        std::cout << "\nTo download GPT-2 weights:" << std::endl;
        std::cout << "  1. Install transformers: pip install transformers" << std::endl;
        std::cout << "  2. Run: python download_gpt2_weights.py" << std::endl;
        std::cout << "  3. Weights will be saved to weights/gpt2_weights.bin" << std::endl;
    }
}

void testGPT2Generation()
{
    std::cout << "\n========== Test: GPT-2 Text Generation ===========" << std::endl;

    // Use smaller config for testing
    GPT2Config config{
        .vocab_size = 50257,
        .max_seq_len = 1024,
        .d_model = 64,      // Smaller for testing
        .num_heads = 4,
        .num_layers = 2,
        .dropout = 0.0f
    };

    std::cout << "Creating GPT-2 model..." << std::endl;
    GPT2 model(netGlobalDevice, gDestSetPool, config);

    // Try to load weights if available
    try {
        model.loadWeights("weights/gpt2_weights.bin");
        std::cout << "✓ Loaded pretrained weights" << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "⚠ Using random weights (pretrained weights not found)" << std::endl;
    }

    // Initialize tokenizer
    std::cout << "\nInitializing tokenizer..." << std::endl;
    BPETokenizer tokenizer("vocab.json", "merges.txt");

    // Test prompts
    std::vector<std::string> prompts = {
        "Once upon a time",
        "The quick brown fox",
        "In a galaxy far, far away"
    };

    // Seed random number generator
    srand((unsigned int)time(nullptr));

    for (const auto& prompt : prompts) {
        std::cout << "\n" << std::string(60, '-') << std::endl;
        std::cout << "Prompt: \"" << prompt << "\"" << std::endl;

        try {
            // Encode prompt
            std::vector<int> prompt_ids = tokenizer.encode(prompt);
            std::cout << "Encoded to " << prompt_ids.size() << " tokens: ";
            for (size_t i = 0; i < std::min(size_t(10), prompt_ids.size()); ++i) {
                std::cout << prompt_ids[i] << " ";
            }
            std::cout << std::endl;

            // Generate
            std::vector<int> generated = model.generate(
                prompt_ids,
                20,      // max_new_tokens
                0.8f,    // temperature
                40       // top_k
            );

            // Decode
            std::string generated_text = tokenizer.decode(generated);
            std::cout << "\nGenerated text:" << std::endl;
            std::cout << "\"" << generated_text << "\"" << std::endl;

        }
        catch (const std::exception& e) {
            std::cout << "⚠ Generation failed: " << e.what() << std::endl;
        }
    }

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "✓ Text generation test completed!" << std::endl;
}

void gpt2Test()
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "GPT-2 Model Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    // Run tests individually to avoid one failure stopping others
    try {
        testGPT2SmallForward();
    }
    catch (const std::exception& e) {
        std::cerr << "\n❌ Small forward test failed: " << e.what() << std::endl;
    }

    try {
        testGPT2Generation();
    }
    catch (const std::exception& e) {
        std::cerr << "\n❌ Generation test failed: " << e.what() << std::endl;
    }

    try {
        testGPT2LoadWeights();
    }
    catch (const std::exception& e) {
        std::cerr << "\n❌ Weight loading test failed: " << e.what() << std::endl;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "GPT-2 tests completed!" << std::endl;
    std::cout << "========================================" << std::endl;
}

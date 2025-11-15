#include "gptDataset.h"
#include "gptDataLoader.h"
#include "../tokenizer/bpeTokenizer.h"
#include "../core/error.h"
#include <iostream>
#include <memory>
#include <fstream>
#include <sstream>

// Test sliding window dataset creation
void testDataset(BPETokenizer& tokenizer) {
    std::cout << "\n========== Test: Dataset Creation ==========" << std::endl;

    std::string text = "Hello, world! This is a test for the dataloader implementation.";
    int max_length = 4;
    int stride = 2;

    GPTDataset dataset(text, tokenizer, max_length, stride);

    std::cout << "Dataset size: " << dataset.size() << " samples" << std::endl;

    // Show first few samples
    std::cout << "\nFirst 3 samples:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(3), dataset.size()); i++) {
        auto [input, target] = dataset.get(i);

        std::cout << "Sample " << i << ":" << std::endl;
        std::cout << "  Input:  ";
        for (int id : input) std::cout << id << " ";
        std::cout << std::endl;

        std::cout << "  Target: ";
        for (int id : target) std::cout << id << " ";
        std::cout << std::endl;
    }
}

// Test dataloader batching
void testDataLoader(BPETokenizer& tokenizer) {
    std::cout << "\n========== Test: DataLoader Batching ==========" << std::endl;

    std::string text = "Hello, world! This is a test for the dataloader implementation. "
                       "We need enough text to create multiple batches for testing purposes.";
    int max_length = 4;
    int stride = 2;
    int batch_size = 3;

    auto dataset = std::make_shared<GPTDataset>(text, tokenizer, max_length, stride);
    GPTDataLoader dataloader(dataset, batch_size, /*shuffle=*/false, /*drop_last=*/true);

    std::cout << "Number of batches: " << dataloader.num_batches() << std::endl;

    // Show first batch
    std::cout << "\nFirst batch:" << std::endl;
    Batch batch = dataloader.get_batch(0);
    std::cout << "  Batch size: " << batch.batch_size() << std::endl;
    std::cout << "  Sequence length: " << batch.seq_length() << std::endl;

    for (size_t i = 0; i < batch.batch_size(); i++) {
        std::cout << "  Sample " << i << " input:  ";
        for (int id : batch.inputs[i]) std::cout << id << " ";
        std::cout << std::endl;

        std::cout << "  Sample " << i << " target: ";
        for (int id : batch.targets[i]) std::cout << id << " ";
        std::cout << std::endl;
    }
}

// Test dataloader with shuffling
void testShuffle(BPETokenizer& tokenizer) {
    std::cout << "\n========== Test: Shuffling ==========" << std::endl;

    std::string text = "Hello, world! This is a test for shuffling functionality in the dataloader.";
    int max_length = 4;
    int stride = 2;
    int batch_size = 2;

    auto dataset = std::make_shared<GPTDataset>(text, tokenizer, max_length, stride);
    GPTDataLoader dataloader(dataset, batch_size, /*shuffle=*/true, /*drop_last=*/true);

    std::cout << "First epoch - first batch:" << std::endl;
    Batch batch1 = dataloader.get_batch(0);
    std::cout << "  First sample input: ";
    for (int id : batch1.inputs[0]) std::cout << id << " ";
    std::cout << std::endl;

    dataloader.reset();

    std::cout << "\nSecond epoch - first batch (reshuffled):" << std::endl;
    Batch batch2 = dataloader.get_batch(0);
    std::cout << "  First sample input: ";
    for (int id : batch2.inputs[0]) std::cout << id << " ";
    std::cout << std::endl;

    std::cout << "\n(Note: May be same or different due to random shuffle)" << std::endl;
}

// Test iterator (range-based for loop)
void testIterator(BPETokenizer& tokenizer) {
    std::cout << "\n========== Test: Iterator (Range-based For) ==========" << std::endl;

    std::string text = "Testing iterator functionality for the dataloader.";
    int max_length = 3;
    int stride = 2;
    int batch_size = 2;

    auto dataset = std::make_shared<GPTDataset>(text, tokenizer, max_length, stride);
    GPTDataLoader dataloader(dataset, batch_size, /*shuffle=*/false, /*drop_last=*/true);

    std::cout << "Iterating through all batches:" << std::endl;
    int batch_count = 0;
    for (const Batch& batch : dataloader) {
        std::cout << "  Batch " << batch_count << ": "
                  << batch.batch_size() << " samples, "
                  << "seq_len=" << batch.seq_length() << std::endl;
        batch_count++;
    }
}

// Test with the-verdict.txt - basic stats
void testWithVerdictFile(BPETokenizer& tokenizer) {
    std::cout << "\n========== Test: the-verdict.txt Basic ==========" << std::endl;

    std::ifstream file(PROJECT_CURRENT_DIR "/the-verdict.txt");
    if (!file.is_open()) {
        std::cerr << "Failed to open the-verdict.txt" << std::endl;
        return;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string text = buffer.str();
    file.close();

    int max_length = 256;  // Common context size
    int stride = 128;      // 50% overlap
    int batch_size = 8;

    auto dataset = std::make_shared<GPTDataset>(text, tokenizer, max_length, stride);
    GPTDataLoader dataloader(dataset, batch_size, /*shuffle=*/true, /*drop_last=*/true);

    std::cout << "Dataset size: " << dataset->size() << " samples" << std::endl;
    std::cout << "Number of batches: " << dataloader.num_batches() << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Sequence length: " << max_length << std::endl;
    std::cout << "Total trainable tokens per epoch: "
              << dataloader.num_batches() * batch_size * max_length << std::endl;
}

// Detailed test: verify verdict text is properly loaded and decoded
void testVerdictTextContent(BPETokenizer& tokenizer) {
    std::cout << "\n========== Test: Verdict Text Content Verification ==========" << std::endl;

    // Load the-verdict.txt
    std::ifstream file(PROJECT_CURRENT_DIR "/the-verdict.txt");
    if (!file.is_open()) {
        std::cerr << "✗ Failed to open the-verdict.txt" << std::endl;
        return;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string original_text = buffer.str();
    file.close();

    std::cout << "✓ File loaded successfully" << std::endl;
    std::cout << "  File size: " << original_text.size() << " bytes" << std::endl;

    // Show first 100 characters of original text
    std::cout << "\nFirst 100 characters of original text:" << std::endl;
    std::cout << "  \"" << original_text.substr(0, 100) << "...\"" << std::endl;

    // Tokenize and decode to verify round-trip
    std::cout << "\nTokenizing entire text..." << std::endl;
    auto tokens = tokenizer.encode(original_text);
    std::cout << "  Total tokens: " << tokens.size() << std::endl;

    std::cout << "\nDecoding back to text..." << std::endl;
    std::string decoded_text = tokenizer.decode(tokens);
    std::cout << "  Decoded size: " << decoded_text.size() << " bytes" << std::endl;

    // Verify round-trip
    bool match = (original_text == decoded_text);
    std::cout << "\nRound-trip verification: " << (match ? "✓ PASS" : "✗ FAIL") << std::endl;

    if (!match) {
        std::cout << "  Original size: " << original_text.size() << std::endl;
        std::cout << "  Decoded size:  " << decoded_text.size() << std::endl;
    }

    // Show first 100 characters of decoded text
    std::cout << "\nFirst 100 characters of decoded text:" << std::endl;
    std::cout << "  \"" << decoded_text.substr(0, 100) << "...\"" << std::endl;
}

// Test: verify sliding window overlap
void testSlidingWindowOverlap(BPETokenizer& tokenizer) {
    std::cout << "\n========== Test: Sliding Window Overlap ==========" << std::endl;

    std::ifstream file(PROJECT_CURRENT_DIR "/the-verdict.txt");
    if (!file.is_open()) {
        std::cerr << "Failed to open the-verdict.txt" << std::endl;
        return;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string text = buffer.str();
    file.close();

    int max_length = 10;
    int stride = 5;  // 50% overlap

    auto dataset = std::make_shared<GPTDataset>(text, tokenizer, max_length, stride);

    std::cout << "Testing with max_length=" << max_length << ", stride=" << stride << std::endl;
    std::cout << "Dataset size: " << dataset->size() << " samples" << std::endl;

    if (dataset->size() >= 2) {
        // Get first two samples
        auto [input1, target1] = dataset->get(0);
        auto [input2, target2] = dataset->get(1);

        std::cout << "\nSample 0 input:  ";
        for (int id : input1) std::cout << id << " ";
        std::cout << std::endl;

        std::cout << "Sample 1 input:  ";
        for (int id : input2) std::cout << id << " ";
        std::cout << std::endl;

        // Check overlap: last (max_length - stride) tokens of sample 0
        // should match first (max_length - stride) tokens of sample 1
        int overlap_size = max_length - stride;
        bool has_overlap = true;

        for (int i = 0; i < overlap_size; i++) {
            if (input1[stride + i] != input2[i]) {
                has_overlap = false;
                break;
            }
        }

        std::cout << "\nOverlap verification (" << overlap_size << " tokens): "
                  << (has_overlap ? "✓ PASS" : "✗ FAIL") << std::endl;

        if (has_overlap) {
            std::cout << "  Overlapping tokens: ";
            for (int i = 0; i < overlap_size; i++) {
                std::cout << input2[i] << " ";
            }
            std::cout << std::endl;
        }
    }
}

// Test: decode batch samples to verify content
void testBatchContentDecoding(BPETokenizer& tokenizer) {
    std::cout << "\n========== Test: Batch Content Decoding ==========" << std::endl;

    std::ifstream file(PROJECT_CURRENT_DIR "/the-verdict.txt");
    if (!file.is_open()) {
        std::cerr << "Failed to open the-verdict.txt" << std::endl;
        return;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string text = buffer.str();
    file.close();

    int max_length = 50;
    int stride = 25;
    int batch_size = 2;

    auto dataset = std::make_shared<GPTDataset>(text, tokenizer, max_length, stride);
    GPTDataLoader dataloader(dataset, batch_size, /*shuffle=*/false, /*drop_last=*/true);

    std::cout << "Getting first batch..." << std::endl;
    Batch batch = dataloader.get_batch(0);

    std::cout << "Batch size: " << batch.batch_size() << std::endl;
    std::cout << "Sequence length: " << batch.seq_length() << std::endl;

    // Decode each sample in the batch
    for (size_t i = 0; i < batch.batch_size(); i++) {
        std::string decoded_input = tokenizer.decode(batch.inputs[i]);
        std::string decoded_target = tokenizer.decode(batch.targets[i]);

        std::cout << "\nSample " << i << ":" << std::endl;
        std::cout << "  Input text:  \"" << decoded_input.substr(0, 80) << "...\"" << std::endl;
        std::cout << "  Target text: \"" << decoded_target.substr(0, 80) << "...\"" << std::endl;

        // Verify that target is shifted by 1
        // The first (max_length-1) tokens of target should match last (max_length-1) of input
        bool shift_correct = true;
        for (int j = 0; j < max_length - 1; j++) {
            if (batch.inputs[i][j + 1] != batch.targets[i][j]) {
                shift_correct = false;
                break;
            }
        }
        std::cout << "  Target shift verification: " << (shift_correct ? "✓ PASS" : "✗ FAIL") << std::endl;
    }
}

// Test: Show token-level details to clarify token vs character shift
void testTokenLevelDetails(BPETokenizer& tokenizer) {
    std::cout << "\n========== Test: Token-Level Shift Details ==========" << std::endl;
    std::cout << "This test demonstrates that shift is TOKEN-based, not character-based\n" << std::endl;

    std::string text = "I HAD always thought Jack Gisburn rather a cheap genius";

    std::cout << "Original text: \"" << text << "\"" << std::endl;

    // Tokenize
    auto tokens = tokenizer.encode(text);
    std::cout << "\nTokenized into " << tokens.size() << " tokens:" << std::endl;

    // Show each token
    for (size_t i = 0; i < std::min(size_t(10), tokens.size()); i++) {
        std::vector<int> single_token = {tokens[i]};
        std::string token_text = tokenizer.decode(single_token);
        std::cout << "  Token[" << i << "] = " << tokens[i]
                  << " -> \"" << token_text << "\"" << std::endl;
    }

    // Create simple dataset
    int max_length = 5;
    int stride = 5;
    GPTDataset dataset(text, tokenizer, max_length, stride);

    std::cout << "\nDataset with max_length=" << max_length << ":" << std::endl;

    if (dataset.size() > 0) {
        auto [input, target] = dataset.get(0);

        std::cout << "\nInput tokens (indices 0-4):" << std::endl;
        std::cout << "  IDs: ";
        for (int id : input) std::cout << id << " ";
        std::cout << std::endl;

        std::cout << "  Text per token:" << std::endl;
        for (size_t i = 0; i < input.size(); i++) {
            std::vector<int> single = {input[i]};
            std::cout << "    [" << i << "] \"" << tokenizer.decode(single) << "\"" << std::endl;
        }
        std::cout << "  Full text: \"" << tokenizer.decode(input) << "\"" << std::endl;

        std::cout << "\nTarget tokens (indices 1-5, shifted by 1 TOKEN):" << std::endl;
        std::cout << "  IDs: ";
        for (int id : target) std::cout << id << " ";
        std::cout << std::endl;

        std::cout << "  Text per token:" << std::endl;
        for (size_t i = 0; i < target.size(); i++) {
            std::vector<int> single = {target[i]};
            std::cout << "    [" << i << "] \"" << tokenizer.decode(single) << "\"" << std::endl;
        }
        std::cout << "  Full text: \"" << tokenizer.decode(target) << "\"" << std::endl;

        std::cout << "\n✓ Notice: Target is input shifted by 1 TOKEN (not 1 character)" << std::endl;
        std::cout << "  This is correct for GPT next-token prediction!" << std::endl;
    }
}

void dataLoaderTest() {
    try {
        std::cout << "Initializing GPT-2 BPE Tokenizer..." << std::endl;
        BPETokenizer tokenizer(
            PROJECT_CURRENT_DIR "/vocab.json",
            PROJECT_CURRENT_DIR "/merges.txt"
        );

        // Run all basic tests
        testDataset(tokenizer);
        testDataLoader(tokenizer);
        testShuffle(tokenizer);
        testIterator(tokenizer);
        testWithVerdictFile(tokenizer);

        // Run detailed verdict file tests
        testVerdictTextContent(tokenizer);
        testSlidingWindowOverlap(tokenizer);
        testBatchContentDecoding(tokenizer);

        // Show token-level details
        testTokenLevelDetails(tokenizer);

        std::cout << "\n========================================" << std::endl;
        std::cout << "All DataLoader tests completed!" << std::endl;
        std::cout << "========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

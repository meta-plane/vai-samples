#include "bpeTokenizer.h"
#include "../core/error.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <chrono>

// Run a simple tokenizer test with a single string
void runTokenizerTest(BPETokenizer& tokenizer, const std::string& text,
                      int& passed_tests, int& total_tests) {
    std::cout << "Original: \"" << text << "\"" << std::endl;

    auto tokens = tokenizer.encode(text);
    std::cout << "Token IDs: ";
    for (int id : tokens) {
        std::cout << id << " ";
    }
    std::cout << std::endl;
    std::cout << "Token count: " << tokens.size() << std::endl;

    std::string decoded = tokenizer.decode(tokens);
    std::cout << "Decoded: \"" << decoded << "\"" << std::endl;

    bool match = (text == decoded);
    std::cout << "Match: " << (match ? "✓ PASS" : "✗ FAIL") << std::endl;

    total_tests++;
    if (match) passed_tests++;
}

// Read a file and return its contents
std::string readTextFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();
    return buffer.str();
}

// Test tokenizer with a long text file, measuring performance
void testLongTextFile(BPETokenizer& tokenizer, const std::string& filepath,
                      int& passed_tests, int& total_tests) {
    try {
        std::string text = readTextFile(filepath);
        std::cout << "File size: " << text.size() << " bytes" << std::endl;

        // Measure encoding time
        auto start = std::chrono::high_resolution_clock::now();
        auto tokens = tokenizer.encode(text);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Token count: " << tokens.size() << std::endl;
        std::cout << "Encoding time: " << duration.count() << " ms" << std::endl;

        // Show first 20 tokens
        std::cout << "First 20 token IDs: ";
        for (size_t i = 0; i < std::min(size_t(20), tokens.size()); ++i) {
            std::cout << tokens[i] << " ";
        }
        std::cout << std::endl;

        // Measure decoding time
        start = std::chrono::high_resolution_clock::now();
        std::string decoded = tokenizer.decode(tokens);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Decoding time: " << duration.count() << " ms" << std::endl;

        // Check round-trip match
        bool match = (text == decoded);
        std::cout << "Round-trip match: " << (match ? "✓ PASS" : "✗ FAIL") << std::endl;

        if (!match) {
            std::cout << "Decoded size: " << decoded.size() << " bytes" << std::endl;
            // Show first difference
            for (size_t i = 0; i < std::min(text.size(), decoded.size()); ++i) {
                if (text[i] != decoded[i]) {
                    std::cout << "First difference at position " << i << std::endl;
                    std::cout << "Original: '" << text.substr(i, 50) << "...'" << std::endl;
                    std::cout << "Decoded:  '" << decoded.substr(i, 50) << "...'" << std::endl;
                    break;
                }
            }
        }

        total_tests++;
        if (match) passed_tests++;
    } catch (const std::exception& e) {
        std::cerr << "Error in long text file test: " << e.what() << std::endl;
    }
}

// Run all basic tokenizer tests
void runBasicTests(BPETokenizer& tokenizer, int& passed_tests, int& total_tests) {
    // Test 1: Simple sentence
    std::cout << "========== Test 1: Simple Sentence ==========" << std::endl;
    runTokenizerTest(tokenizer, "Hello, world!", passed_tests, total_tests);

    // Test 2: Complex sentence
    std::cout << "\n========== Test 2: Complex Sentence ==========" << std::endl;
    runTokenizerTest(tokenizer, "Learning to build LLMs from scratch is fascinating!",
                     passed_tests, total_tests);

    // Test 3: Numbers and special characters
    std::cout << "\n========== Test 3: Numbers & Special Chars ==========" << std::endl;
    runTokenizerTest(tokenizer, "GPT-2 was released in 2019!", passed_tests, total_tests);

    // Test 4: Contractions
    std::cout << "\n========== Test 4: Contractions ==========" << std::endl;
    runTokenizerTest(tokenizer, "I'm learning. It's fascinating!", passed_tests, total_tests);

    // Test 5: Unicode characters
    std::cout << "\n========== Test 5: Unicode ==========" << std::endl;
    runTokenizerTest(tokenizer, "Hello world!", passed_tests, total_tests);

    // Test 6: Multiple spaces
    std::cout << "\n========== Test 6: Whitespace ==========" << std::endl;
    runTokenizerTest(tokenizer, "Hello  world", passed_tests, total_tests);
}

// Print final test results
void printTestResults(int passed_tests, int total_tests) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Results: " << passed_tests << "/" << total_tests << " passed" << std::endl;
    std::cout << "========================================" << std::endl;
}

void tokenizerTest() {
    try {
        // Initialize tokenizer
        std::cout << "Initializing GPT-2 BPE Tokenizer..." << std::endl;
        BPETokenizer tokenizer(
            PROJECT_CURRENT_DIR "/vocab.json",
            PROJECT_CURRENT_DIR "/merges.txt"
        );
        std::cout << "\nTokenizer ready! Vocab size: " << tokenizer.vocab_size() << std::endl;

        // Run all tests
        int total_tests = 0;
        int passed_tests = 0;

        // Basic tests
        runBasicTests(tokenizer, passed_tests, total_tests);

        // Long text file test
        std::cout << "\n========== Test 7: the-verdict.txt (Long Text) ==========" << std::endl;
        testLongTextFile(tokenizer, PROJECT_CURRENT_DIR "/the-verdict.txt",
                         passed_tests, total_tests);

        // Print results
        printTestResults(passed_tests, total_tests);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}


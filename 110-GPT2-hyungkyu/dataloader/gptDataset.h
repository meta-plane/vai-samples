#pragma once

#include <vector>
#include <string>

// Forward declaration
class BPETokenizer;

// Dataset class for GPT training data using sliding window approach
// Creates input-target pairs where target is shifted by 1 token
class GPTDataset {
public:
    // Constructor: creates dataset from text using sliding window
    // max_length: length of each sequence (context size)
    // stride: step size for sliding window
    GPTDataset(const std::string& text, BPETokenizer& tokenizer,
               int max_length, int stride);

    // Get dataset size (number of samples)
    size_t size() const;

    // Get a single sample (input, target pair)
    // Returns: pair of (input_tokens, target_tokens)
    std::pair<std::vector<int>, std::vector<int>> get(size_t idx) const;

    // Get all input sequences
    const std::vector<std::vector<int>>& get_inputs() const;

    // Get all target sequences
    const std::vector<std::vector<int>>& get_targets() const;

private:
    std::vector<std::vector<int>> input_ids;   // Input token sequences
    std::vector<std::vector<int>> target_ids;  // Target token sequences (shifted by 1)

    // Create sliding window chunks from token IDs
    void create_chunks(const std::vector<int>& token_ids, int max_length, int stride);
};

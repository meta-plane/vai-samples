#include "gptDataset.h"
#include "../tokenizer/bpeTokenizer.h"
#include "../core/error.h"
#include <iostream>

GPTDataset::GPTDataset(const std::string& text, BPETokenizer& tokenizer,
                       int max_length, int stride) {
    // Tokenize the entire text
    std::vector<int> token_ids = tokenizer.encode(text);

    std::cout << "Total tokens: " << token_ids.size() << std::endl;
    std::cout << "Creating dataset with max_length=" << max_length
              << ", stride=" << stride << std::endl;

    // Create sliding window chunks
    create_chunks(token_ids, max_length, stride);

    std::cout << "Dataset size: " << input_ids.size() << " samples" << std::endl;
}

void GPTDataset::create_chunks(const std::vector<int>& token_ids,
                                int max_length, int stride) {
    // Need at least max_length + 1 tokens to create one sample
    if (token_ids.size() <= static_cast<size_t>(max_length)) {
        std::cerr << "Warning: Not enough tokens to create samples" << std::endl;
        return;
    }

    // Sliding window: step through text with given stride
    // Input: token_ids[i : i + max_length]
    // Target: token_ids[i + 1 : i + max_length + 1] (shifted by 1)
    for (size_t i = 0; i + max_length < token_ids.size(); i += stride) {
        std::vector<int> input_chunk(token_ids.begin() + i,
                                      token_ids.begin() + i + max_length);
        std::vector<int> target_chunk(token_ids.begin() + i + 1,
                                       token_ids.begin() + i + max_length + 1);

        input_ids.push_back(input_chunk);
        target_ids.push_back(target_chunk);
    }
}

size_t GPTDataset::size() const {
    return input_ids.size();
}

std::pair<std::vector<int>, std::vector<int>> GPTDataset::get(size_t idx) const {
    ASSERT_(idx < input_ids.size());
    return {input_ids[idx], target_ids[idx]};
}

const std::vector<std::vector<int>>& GPTDataset::get_inputs() const {
    return input_ids;
}

const std::vector<std::vector<int>>& GPTDataset::get_targets() const {
    return target_ids;
}

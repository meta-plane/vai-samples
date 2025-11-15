#pragma once

#include "gptDataset.h"
#include <vector>
#include <memory>

// Batch structure: holds input and target batches
struct Batch {
    std::vector<std::vector<int>> inputs;   // [batch_size, seq_length]
    std::vector<std::vector<int>> targets;  // [batch_size, seq_length]

    size_t batch_size() const { return inputs.size(); }
    size_t seq_length() const { return inputs.empty() ? 0 : inputs[0].size(); }
};

// DataLoader: creates batches from dataset with optional shuffling
class GPTDataLoader {
public:
    // Constructor
    // batch_size: number of samples per batch
    // shuffle: whether to shuffle data each epoch
    // drop_last: whether to drop incomplete last batch
    GPTDataLoader(std::shared_ptr<GPTDataset> dataset,
                  int batch_size,
                  bool shuffle = true,
                  bool drop_last = true);

    // Get total number of batches
    size_t num_batches() const;

    // Get a single batch by index
    Batch get_batch(size_t batch_idx) const;

    // Reset and shuffle for new epoch
    void reset();

    // Iterator support for range-based for loop
    class Iterator {
    public:
        Iterator(GPTDataLoader* loader, size_t idx);
        Batch operator*() const;
        Iterator& operator++();
        bool operator!=(const Iterator& other) const;

    private:
        GPTDataLoader* loader;
        size_t current_idx;
    };

    Iterator begin();
    Iterator end();

private:
    std::shared_ptr<GPTDataset> dataset;
    int batch_size;
    bool shuffle;
    bool drop_last;
    std::vector<size_t> indices;  // Shuffled indices

    // Initialize/shuffle indices
    void initialize_indices();
    void shuffle_indices();
};

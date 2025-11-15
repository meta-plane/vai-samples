#include "gptDataLoader.h"
#include "../error.h"
#include <algorithm>
#include <random>
#include <iostream>

GPTDataLoader::GPTDataLoader(std::shared_ptr<GPTDataset> dataset,
                             int batch_size,
                             bool shuffle,
                             bool drop_last)
    : dataset(dataset)
    , batch_size(batch_size)
    , shuffle(shuffle)
    , drop_last(drop_last) {

    ASSERT_(dataset != nullptr);
    ASSERT_(batch_size > 0);

    initialize_indices();
}

void GPTDataLoader::initialize_indices() {
    // Create sequential indices
    indices.clear();
    for (size_t i = 0; i < dataset->size(); i++) {
        indices.push_back(i);
    }

    // Shuffle if requested
    if (shuffle) {
        shuffle_indices();
    }
}

void GPTDataLoader::shuffle_indices() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);
}

size_t GPTDataLoader::num_batches() const {
    size_t total_samples = dataset->size();

    if (drop_last) {
        // Drop incomplete last batch
        return total_samples / batch_size;
    } else {
        // Keep incomplete last batch
        return (total_samples + batch_size - 1) / batch_size;
    }
}

Batch GPTDataLoader::get_batch(size_t batch_idx) const {
    ASSERT_(batch_idx < num_batches());

    Batch batch;
    size_t start_idx = batch_idx * batch_size;
    size_t end_idx = std::min(start_idx + batch_size, dataset->size());

    // Collect samples for this batch
    for (size_t i = start_idx; i < end_idx; i++) {
        size_t sample_idx = indices[i];
        auto [input, target] = dataset->get(sample_idx);
        batch.inputs.push_back(input);
        batch.targets.push_back(target);
    }

    return batch;
}

void GPTDataLoader::reset() {
    // Re-shuffle for new epoch
    if (shuffle) {
        shuffle_indices();
    }
}

// Iterator implementation
GPTDataLoader::Iterator::Iterator(GPTDataLoader* loader, size_t idx)
    : loader(loader), current_idx(idx) {}

Batch GPTDataLoader::Iterator::operator*() const {
    return loader->get_batch(current_idx);
}

GPTDataLoader::Iterator& GPTDataLoader::Iterator::operator++() {
    ++current_idx;
    return *this;
}

bool GPTDataLoader::Iterator::operator!=(const Iterator& other) const {
    return current_idx != other.current_idx;
}

GPTDataLoader::Iterator GPTDataLoader::begin() {
    return Iterator(this, 0);
}

GPTDataLoader::Iterator GPTDataLoader::end() {
    return Iterator(this, num_batches());
}

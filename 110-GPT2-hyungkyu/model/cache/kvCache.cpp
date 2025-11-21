#include "kvCache.h"
#include "../../core/error.h"
#include <iostream>

KVCacheManager::KVCacheManager(
    uint32_t num_layers,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t max_seq_len)
    : num_layers(num_layers)
    , batch_size(batch_size)
    , num_heads(num_heads)
    , head_dim(head_dim)
    , max_seq_len(max_seq_len)
{
    // Pre-allocate layer cache structures (but not GPU buffers yet)
    layer_caches.resize(num_layers);

    // Set max length for each layer
    for (auto& cache : layer_caches) {
        cache.max_len = max_seq_len;
        cache.current_len = 0;
    }
}

void KVCacheManager::initialize()
{
    if (is_initialized) {
        std::cout << "Warning: KVCacheManager already initialized" << std::endl;
        return;
    }

    std::cout << "Initializing KV Cache..." << std::endl;
    std::cout << "  Layers: " << num_layers << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Num heads: " << num_heads << std::endl;
    std::cout << "  Head dim: " << head_dim << std::endl;
    std::cout << "  Max seq len: " << max_seq_len << std::endl;

    // Calculate memory requirements
    size_t cache_elements = batch_size * num_heads * max_seq_len * head_dim;
    size_t cache_bytes = cache_elements * sizeof(float);
    size_t total_bytes = cache_bytes * 2 * num_layers;  // 2 for K and V, × num_layers

    std::cout << "  Cache per layer (K or V): " << cache_bytes / 1024.0 / 1024.0 << " MB" << std::endl;
    std::cout << "  Total cache memory: " << total_bytes / 1024.0 / 1024.0 << " MB" << std::endl;

    // Allocate cache tensors for each layer
    for (uint32_t layer = 0; layer < num_layers; ++layer) {
        auto& cache = layer_caches[layer];

        // Create tensors with shape [batch, num_heads, max_seq_len, head_dim]
        cache.K = Tensor(batch_size, num_heads, max_seq_len, head_dim);
        cache.V = Tensor(batch_size, num_heads, max_seq_len, head_dim);

        // Mark as non-constant (these are temporary buffers)
        cache.K.setConstant(false);
        cache.V.setConstant(false);

        cache.current_len = 0;
    }

    is_initialized = true;
    std::cout << "✓ KV Cache initialized successfully\n" << std::endl;
}

void KVCacheManager::reset()
{
    _ASSERT(is_initialized);

    // Reset all cache lengths to 0
    // Note: We don't need to clear the actual tensor data,
    // just reset the length tracker. Old data will be overwritten.
    for (auto& cache : layer_caches) {
        cache.current_len = 0;
    }
}

LayerKVCache& KVCacheManager::getCache(uint32_t layer_idx)
{
    _ASSERT(layer_idx < num_layers);
    return layer_caches[layer_idx];
}

const LayerKVCache& KVCacheManager::getCache(uint32_t layer_idx) const
{
    _ASSERT(layer_idx < num_layers);
    return layer_caches[layer_idx];
}

void KVCacheManager::updateCacheLength(uint32_t layer_idx, uint32_t new_tokens)
{
    _ASSERT(layer_idx < num_layers);
    _ASSERT(layer_caches[layer_idx].current_len + new_tokens <= max_seq_len);

    layer_caches[layer_idx].current_len += new_tokens;
}

uint32_t KVCacheManager::getCurrentLength() const
{
    if (layer_caches.empty()) {
        return 0;
    }

    // All layers should have the same current length
    // (we verify this in debug mode)
    uint32_t first_len = layer_caches[0].current_len;

#ifdef _DEBUG
    for (const auto& cache : layer_caches) {
        _ASSERT(cache.current_len == first_len);
    }
#endif

    return first_len;
}

uint32_t KVCacheManager::getMaxLength() const
{
    return max_seq_len;
}

bool KVCacheManager::isInitialized() const
{
    return is_initialized;
}

bool KVCacheManager::isEmpty() const
{
    return getCurrentLength() == 0;
}

uint32_t KVCacheManager::getNumLayers() const
{
    return num_layers;
}

void KVCacheManager::printInfo() const
{
    std::cout << "\n=== KV Cache Info ===" << std::endl;
    std::cout << "Status: " << (is_initialized ? "Initialized" : "Not initialized") << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Layers: " << num_layers << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Num heads: " << num_heads << std::endl;
    std::cout << "  Head dim: " << head_dim << std::endl;
    std::cout << "  Max seq len: " << max_seq_len << std::endl;
    std::cout << "Current state:" << std::endl;
    std::cout << "  Cached length: " << getCurrentLength() << " / " << max_seq_len << std::endl;
    std::cout << "  Available space: " << (max_seq_len - getCurrentLength()) << " tokens" << std::endl;

    // Per-layer info
    if (is_initialized) {
        std::cout << "Per-layer status:" << std::endl;
        for (uint32_t i = 0; i < num_layers; ++i) {
            const auto& cache = layer_caches[i];
            std::cout << "  Layer " << i << ": "
                      << cache.current_len << " / " << cache.max_len
                      << (cache.isEmpty() ? " (empty)" : "")
                      << (cache.isFull() ? " (full)" : "")
                      << std::endl;
        }
    }
    std::cout << "====================\n" << std::endl;
}

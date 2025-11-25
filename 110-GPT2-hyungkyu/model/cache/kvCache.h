#ifndef KV_CACHE_H
#define KV_CACHE_H

#include "../../core/tensor.h"
#include "../../core/vulkanApp.h"
#include <vector>
#include <cstdint>

/**
 * KV Cache for a single transformer layer
 *
 * Stores Key and Value matrices from previous forward passes
 * to avoid recomputing them during autoregressive generation.
 *
 * Shape: [batch_size, num_heads, seq_len, head_dim]
 */
struct LayerKVCache
{
    Tensor K;  // Cached Key tensor
    Tensor V;  // Cached Value tensor

    uint32_t current_len = 0;  // Current cached sequence length
    uint32_t max_len = 0;      // Maximum capacity

    // Check if cache is empty
    bool isEmpty() const {
        return current_len == 0;
    }

    // Check if cache is at capacity
    bool isFull() const {
        return current_len >= max_len;
    }

    // Get available space
    uint32_t availableSpace() const {
        return max_len - current_len;
    }
};

/**
 * Manages KV caches for all transformer layers
 *
 * Responsibilities:
 * - Allocate and initialize cache buffers for each layer
 * - Track current cache state (length, capacity)
 * - Provide cache access interface
 * - Handle cache reset for new sequences
 *
 * Memory layout per layer:
 *   K: [batch, num_heads, max_seq_len, head_dim]
 *   V: [batch, num_heads, max_seq_len, head_dim]
 *
 * Total memory: 2 × num_layers × batch × num_heads × max_seq_len × head_dim × 4 bytes
 * Example (GPT-2, 400 tokens): 2 × 12 × 1 × 12 × 400 × 64 × 4 = ~30 MB
 */
class KVCacheManager
{
    std::vector<LayerKVCache> layer_caches;

    // Configuration
    uint32_t num_layers;
    uint32_t batch_size;
    uint32_t num_heads;
    uint32_t head_dim;
    uint32_t max_seq_len;

    bool is_initialized = false;

public:
    /**
     * Constructor
     *
     * @param num_layers Number of transformer layers
     * @param batch_size Batch size (usually 1 for generation)
     * @param num_heads Number of attention heads
     * @param head_dim Dimension per head (d_model / num_heads)
     * @param max_seq_len Maximum sequence length to cache
     */
    KVCacheManager(
        uint32_t num_layers,
        uint32_t batch_size,
        uint32_t num_heads,
        uint32_t head_dim,
        uint32_t max_seq_len);

    /**
     * Initialize cache buffers (allocate GPU memory)
     * Should be called once before first use
     */
    void initialize();

    /**
     * Reset all caches to empty state
     * Call this at the beginning of each generation sequence
     */
    void reset();

    /**
     * Get cache for a specific layer
     *
     * @param layer_idx Layer index (0 to num_layers-1)
     * @return Reference to the layer's KV cache
     */
    LayerKVCache& getCache(uint32_t layer_idx);

    /**
     * Get const cache for a specific layer
     */
    const LayerKVCache& getCache(uint32_t layer_idx) const;

    /**
     * Update cache length for a specific layer
     * Called after appending new K, V to the cache
     *
     * @param layer_idx Layer index
     * @param new_tokens Number of new tokens added
     */
    void updateCacheLength(uint32_t layer_idx, uint32_t new_tokens);

    /**
     * Get current cached sequence length
     * (should be same across all layers)
     */
    uint32_t getCurrentLength() const;

    /**
     * Get maximum cache capacity
     */
    uint32_t getMaxLength() const;

    /**
     * Check if cache is initialized
     */
    bool isInitialized() const;

    /**
     * Check if cache is empty
     */
    bool isEmpty() const;

    /**
     * Get number of layers
     */
    uint32_t getNumLayers() const;

    /**
     * Get configuration info for debugging
     */
    void printInfo() const;
};

#endif // KV_CACHE_H

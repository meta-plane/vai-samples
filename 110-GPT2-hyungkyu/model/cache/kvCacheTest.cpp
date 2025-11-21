#include "kvCache.h"
#include "../../core/error.h"
#include <iostream>

/**
 * Test KVCache basic functionality
 */
void testKVCache()
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "KV Cache Test" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Test configuration (GPT-2 124M parameters)
    const uint32_t num_layers = 12;
    const uint32_t batch_size = 1;
    const uint32_t num_heads = 12;
    const uint32_t head_dim = 64;   // d_model (768) / num_heads (12)
    const uint32_t max_seq_len = 512;

    try {
        // 1. Create KVCacheManager
        std::cout << "=== Test 1: Create KVCacheManager ===" << std::endl;
        KVCacheManager cache_manager(num_layers, batch_size, num_heads, head_dim, max_seq_len);
        std::cout << "✓ KVCacheManager created successfully\n" << std::endl;

        // 2. Initialize caches
        std::cout << "=== Test 2: Initialize Caches ===" << std::endl;
        cache_manager.initialize();
        _ASSERT(cache_manager.isInitialized());
        _ASSERT(cache_manager.isEmpty());
        _ASSERT(cache_manager.getCurrentLength() == 0);
        std::cout << "✓ Caches initialized successfully\n" << std::endl;

        // 3. Access layer caches
        std::cout << "=== Test 3: Access Layer Caches ===" << std::endl;
        for (uint32_t i = 0; i < num_layers; ++i) {
            LayerKVCache& layer_cache = cache_manager.getCache(i);
            _ASSERT(layer_cache.isEmpty());
            _ASSERT(layer_cache.max_len == max_seq_len);
            _ASSERT(layer_cache.current_len == 0);
            _ASSERT(layer_cache.K.shape().size() == 4);  // [batch, heads, seq, dim]
            _ASSERT(layer_cache.V.shape().size() == 4);
        }
        std::cout << "✓ All " << num_layers << " layer caches accessible\n" << std::endl;

        // 4. Simulate cache usage (update lengths)
        std::cout << "=== Test 4: Simulate Cache Updates ===" << std::endl;

        // Simulate prompt phase (10 tokens)
        uint32_t prompt_len = 10;
        for (uint32_t i = 0; i < num_layers; ++i) {
            cache_manager.updateCacheLength(i, prompt_len);
        }
        _ASSERT(cache_manager.getCurrentLength() == prompt_len);
        _ASSERT(!cache_manager.isEmpty());
        std::cout << "  After prompt (10 tokens): current_len = " << cache_manager.getCurrentLength() << std::endl;

        // Simulate generation (20 more tokens, one by one)
        for (uint32_t token = 0; token < 20; ++token) {
            for (uint32_t i = 0; i < num_layers; ++i) {
                cache_manager.updateCacheLength(i, 1);
            }
        }
        _ASSERT(cache_manager.getCurrentLength() == 30);
        std::cout << "  After generation (20 tokens): current_len = " << cache_manager.getCurrentLength() << std::endl;
        std::cout << "✓ Cache updates working correctly\n" << std::endl;

        // 5. Reset cache
        std::cout << "=== Test 5: Reset Cache ===" << std::endl;
        cache_manager.reset();
        _ASSERT(cache_manager.isEmpty());
        _ASSERT(cache_manager.getCurrentLength() == 0);
        std::cout << "  After reset: current_len = " << cache_manager.getCurrentLength() << std::endl;
        std::cout << "✓ Cache reset working correctly\n" << std::endl;

        // 6. Print cache info
        std::cout << "=== Test 6: Print Cache Info ===" << std::endl;
        cache_manager.printInfo();
        std::cout << "✓ Cache info printed successfully\n" << std::endl;

        // 7. Test capacity limits
        std::cout << "=== Test 7: Test Capacity Limits ===" << std::endl;
        for (uint32_t i = 0; i < num_layers; ++i) {
            cache_manager.updateCacheLength(i, max_seq_len);
        }
        _ASSERT(cache_manager.getCurrentLength() == max_seq_len);

        for (uint32_t i = 0; i < num_layers; ++i) {
            LayerKVCache& layer_cache = cache_manager.getCache(i);
            _ASSERT(layer_cache.isFull());
            _ASSERT(layer_cache.availableSpace() == 0);
        }
        std::cout << "  Cache at full capacity: " << cache_manager.getCurrentLength()
                  << " / " << cache_manager.getMaxLength() << std::endl;
        std::cout << "✓ Capacity limits working correctly\n" << std::endl;

        std::cout << "\n========================================" << std::endl;
        std::cout << "✓ All KV Cache Tests Passed!" << std::endl;
        std::cout << "========================================\n" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "\n✗ KV Cache Test Failed: " << e.what() << std::endl;
        throw;
    }
}

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

/**
 * Test KV Cache integration with AttentionNode
 * This tests the cache update simulation
 */
void testKVCacheIntegration()
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "KV Cache + AttentionNode Integration Test" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Test configuration
    const uint32_t batch_size = 1;
    const uint32_t num_heads = 4;
    const uint32_t head_dim = 16;
    const uint32_t d_model = num_heads * head_dim;  // 64
    const uint32_t max_seq_len = 128;

    try {
        std::cout << "=== Test 1: Setup AttentionNode and Cache ===" << std::endl;

        // Create a single-layer cache for this test
        LayerKVCache layer_cache;
        layer_cache.max_len = max_seq_len;
        layer_cache.current_len = 0;
        layer_cache.K = Tensor(batch_size, num_heads, max_seq_len, head_dim);
        layer_cache.V = Tensor(batch_size, num_heads, max_seq_len, head_dim);
        layer_cache.K.setConstant(false);
        layer_cache.V.setConstant(false);

        std::cout << "✓ Layer cache created (max_len=" << max_seq_len << ")\n" << std::endl;

        std::cout << "=== Test 2: First Forward Pass (Prompt Phase) ===" << std::endl;
        std::cout << "  Processing 5 tokens without cache..." << std::endl;

        // For now, we just verify the setup works
        // Full integration will be tested when GPT2Net is integrated
        _ASSERT(layer_cache.isEmpty());
        _ASSERT(layer_cache.availableSpace() == max_seq_len);

        std::cout << "  ✓ Initial state: cache empty, space=" << layer_cache.availableSpace() << std::endl;

        // Simulate cache update after forward pass
        uint32_t prompt_len = 5;
        layer_cache.current_len = prompt_len;
        _ASSERT(!layer_cache.isEmpty());
        _ASSERT(layer_cache.current_len == prompt_len);

        std::cout << "  ✓ After prompt: cache_len=" << layer_cache.current_len
                  << ", space=" << layer_cache.availableSpace() << "\n" << std::endl;

        std::cout << "=== Test 3: Second Forward Pass (Generation) ===" << std::endl;
        std::cout << "  Processing 1 token with cache (len=" << layer_cache.current_len << ")..." << std::endl;

        // Simulate generation step
        uint32_t new_tokens = 1;
        layer_cache.current_len += new_tokens;
        _ASSERT(layer_cache.current_len == prompt_len + new_tokens);

        std::cout << "  ✓ After generation: cache_len=" << layer_cache.current_len
                  << ", space=" << layer_cache.availableSpace() << "\n" << std::endl;

        std::cout << "=== Test 4: Multiple Generation Steps ===" << std::endl;
        for (uint32_t i = 0; i < 10; ++i) {
            layer_cache.current_len += 1;
        }
        _ASSERT(layer_cache.current_len == prompt_len + new_tokens + 10);
        std::cout << "  ✓ After 10 more tokens: cache_len=" << layer_cache.current_len << "\n" << std::endl;

        std::cout << "=== Test 5: Cache Reset ===" << std::endl;
        layer_cache.current_len = 0;
        _ASSERT(layer_cache.isEmpty());
        std::cout << "  ✓ Cache reset successful\n" << std::endl;

        std::cout << "\n========================================" << std::endl;
        std::cout << "✓ KV Cache Integration Tests Passed!" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "\nNote: Full AttentionNode integration will be tested in Phase 3" << std::endl;
        std::cout << "      when GPT2Net is modified to use the cache.\n" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "\n✗ KV Cache Integration Test Failed: " << e.what() << std::endl;
        throw;
    }
}

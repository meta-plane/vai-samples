#ifndef GPT2_NET_H
#define GPT2_NET_H

#include "../core/neuralNet.h"
#include "../core/vulkanApp.h"
#include "embedding/embeddingNode.h"
#include "transformerBlock/transformer.h"
#include "lmHeadNode.h"
#include "cache/kvCache.h"
#include <memory>

using namespace vk;

/**
 * GPT-2 Configuration
 */
struct GPT2Config {
    uint32_t vocab_size;      // 50257 for GPT-2
    uint32_t max_seq_len;     // 1024 for GPT-2
    uint32_t d_model;         // 768 for small, 1024 for medium
    uint32_t num_heads;       // 12 for small, 16 for medium
    uint32_t num_layers;      // 12 for small, 24 for medium
    float dropout;            // Not used in inference
};

/**
 * GPT2Net - Direct NeuralNet implementation (like MnistNet)
 *
 * Architecture:
 * input → embedding → transformer_blocks → final_norm → lm_head → output
 */
class GPT2Net : public NeuralNet
{
    GPT2Config config;
    std::unique_ptr<KVCacheManager> cache_manager;  // KV cache for generation

public:
    // Nodes as objects (like MnistNet)
    GPTEmbeddingNode embedding;
    std::vector<std::unique_ptr<TransformerBlock>> transformerBlocks;
    LayerNormNode finalNorm;
    LMHeadNode lmHead;
    GPT2Net(Device& device, const GPT2Config& config)
        : NeuralNet(device, 1, 1)
        , config(config)
        , embedding(config.vocab_size, config.max_seq_len, config.d_model)
        , finalNorm(config.d_model)
        , lmHead(config.d_model, config.vocab_size)
    {
        // 1. Create transformer blocks (dynamic count)
        transformerBlocks.reserve(config.num_layers);
        for (uint32_t i = 0; i < config.num_layers; ++i) {
            transformerBlocks.push_back(
                std::make_unique<TransformerBlock>(config.d_model, config.num_heads)
            );
        }

        // 2. Connect graph: input → embedding → blocks → norm → lm_head → output
        // Note: Weight tying handled in operator[] below
        if (config.num_layers == 1) {
            input(0) - embedding - *transformerBlocks[0] - finalNorm - lmHead - output(0);
        } else if (config.num_layers == 2) {
            input(0) - embedding - *transformerBlocks[0] - *transformerBlocks[1] - finalNorm - lmHead - output(0);
        } else if (config.num_layers >= 3) {
            input(0) - embedding - *transformerBlocks[0];
            for (size_t i = 1; i < transformerBlocks.size(); ++i) {
                *transformerBlocks[i - 1] - *transformerBlocks[i];
            }
            *transformerBlocks.back() - finalNorm - lmHead - output(0);
        }
    }

    // No destructor needed - automatic cleanup!

    // Weight access (like MnistNet)
    Tensor& operator[](const std::string& name)
    {
        // Embedding weights
        if (name == "embedding.token") return embedding["token_weight"];
        if (name == "embedding.position") return embedding["pos_weight"];

        // Transformer block weights (e.g., "block.0.ln1.weight")
        if (name.starts_with("block.")) {
            size_t dot1 = name.find('.', 6);
            if (dot1 != std::string::npos) {
                uint32_t layerIdx = std::stoi(name.substr(6, dot1 - 6));
                std::string rest = name.substr(dot1 + 1);

                if (layerIdx < transformerBlocks.size()) {
                    return (*transformerBlocks[layerIdx])[rest];
                }
            }
        }

        // Final layer norm
        if (name == "finalNorm.weight") return finalNorm["scale"];
        if (name == "finalNorm.bias") return finalNorm["shift"];

        throw std::runtime_error("No such weight in GPT2Net: " + name);
    }

    const GPT2Config& getConfig() const { return config; }

    /**
     * Enable KV caching for efficient autoregressive generation
     * Creates and initializes cache for all transformer layers
     */
    void enableCache()
    {
        if (!cache_manager) {
            uint32_t batch_size = 1;  // Currently only support batch_size=1 for generation
            uint32_t head_dim = config.d_model / config.num_heads;

            cache_manager = std::make_unique<KVCacheManager>(
                config.num_layers,
                batch_size,
                config.num_heads,
                head_dim,
                config.max_seq_len
            );

            cache_manager->initialize();

            // Connect cache to all attention layers
            for (uint32_t i = 0; i < config.num_layers; ++i) {
                transformerBlocks[i]->setAttentionCache(&cache_manager->getCache(i));
            }
        }
    }

    /**
     * Disable KV caching and free cache memory
     */
    void disableCache()
    {
        if (cache_manager) {
            // Disconnect cache from all attention layers
            for (uint32_t i = 0; i < config.num_layers; ++i) {
                transformerBlocks[i]->disableAttentionCache();
            }

            cache_manager.reset();
        }
    }

    /**
     * Reset cache (clear all cached K,V values)
     */
    void resetCache()
    {
        if (cache_manager) {
            cache_manager->reset();
        }
    }

    /**
     * Update cache lengths after forward pass
     * @param new_tokens Number of new tokens processed
     */
    void updateCacheLength(uint32_t new_tokens)
    {
        if (cache_manager) {
            for (uint32_t i = 0; i < config.num_layers; ++i) {
                cache_manager->updateCacheLength(i, new_tokens);
            }
        }
    }

    /**
     * Check if cache is enabled
     */
    bool isCacheEnabled() const { return cache_manager != nullptr; }

    /**
     * Get current cache length
     */
    uint32_t getCacheLength() const
    {
        return cache_manager ? cache_manager->getCurrentLength() : 0;
    }

    /**
     * Get cache manager (for debugging/inspection)
     */
    KVCacheManager* getCacheManager() { return cache_manager.get(); }

    /**
     * Set position offset for embedding layer (for KV cache support)
     * When using KV cache, new tokens need to use their absolute position
     * in the full sequence, not their relative position in the input
     */
    void setPositionOffset(uint32_t offset)
    {
        embedding.setPositionOffset(offset);
    }
};

// Helper function to create common configs
inline GPT2Config GPT2SmallConfig() {
    return {50257, 1024, 768, 12, 12, 0.0f};
}

inline GPT2Config GPT2MediumConfig() {
    return {50257, 1024, 1024, 16, 24, 0.0f};
}

inline GPT2Config GPT2TinyConfig() {
    return {50257, 128, 64, 4, 1, 0.0f};  // For testing - 1 layer only
}

#endif // GPT2_NET_H

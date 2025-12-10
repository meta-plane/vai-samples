#ifndef ATTENTION_NODE_H
#define ATTENTION_NODE_H

#include "../../core/neuralNet.h"
#include "../../core/globalContext.h"
#include "../../core/vulkanApp.h"
#include "../cache/kvCache.h"

using namespace vk;

/**
 * Linear transformation: Y = X @ W^T
 * Input: [batch, seq_len, in_features]
 * Weight: [out_features, in_features]
 * Output: [batch, seq_len, out_features]
 */
class LinearNode : public Node
{
    uint32_t in_features;
    uint32_t out_features;

    ComputePipeline linearPipeline;
    DescriptorSet linearDescSet;

public:
    LinearNode(uint32_t in_features, uint32_t out_features);

    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

/**
 * Softmax along the last dimension
 * Input: [batch, ..., dim]
 * Output: [batch, ..., dim] (sum along last dim = 1.0)
 */
class SoftmaxNode : public Node
{
    ComputePipeline softmaxPipeline;
    DescriptorSet softmaxDescSet;

public:
    SoftmaxNode();

    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

/**
 * Multi-Head Attention
 * Input: [batch, seq_len, d_in]
 * Output: [batch, seq_len, d_out]
 *
 * Internal weights:
 * - W_query: [d_out, d_in] - project input to attention space
 * - W_key: [d_out, d_in] - project input to attention space
 * - W_value: [d_out, d_in] - project input to attention space
 * - W_out: [d_out, d_out] - final transformation in output space
 */
class MultiHeadAttentionNode : public Node
{
    uint32_t d_in;      // Input dimension
    uint32_t d_out;     // Output dimension
    uint32_t num_heads;
    uint32_t head_dim;

    // KV Cache support
    LayerKVCache* kv_cache = nullptr;  // Pointer to external cache (optional)
    bool use_cache = false;             // Whether to use KV caching

    // Pipelines for each stage (standard mode)
    ComputePipeline qkvProjection;        // Project input to Q, K, V
    ComputePipeline attentionScores;      // Q @ K^T / sqrt(head_dim)
    ComputePipeline applyCausalMask;      // Set upper triangle to -inf
    ComputePipeline softmaxPipeline;      // Softmax on attention scores
    ComputePipeline weightedSum;          // attn_weights @ V
    ComputePipeline combineHeads;         // Reshape and concat heads
    ComputePipeline outputProjection;     // Final linear projection

    // Pipelines for cache mode
    ComputePipeline reshapeForHeads;      // Reshape to multi-head format [B,S,D] -> [B,H,S,HD]
    ComputePipeline concatenateKV;        // Concatenate cached K,V with new K,V
    ComputePipeline updateCache;          // Update cache with new K,V values
    ComputePipeline scoresPipelineCached; // Attention scores with different K/V lengths
    ComputePipeline maskPipelineCached;   // Causal mask for cached attention
    ComputePipeline weightedSumPipelineCached; // Weighted sum for cached attention

    // Descriptor sets
    DescriptorSet qkvProjDescSet;
    DescriptorSet reshapeDescSetQ;        // For Q reshape in cache mode
    DescriptorSet reshapeDescSetK;        // For K reshape in cache mode
    DescriptorSet reshapeDescSetV;        // For V reshape in cache mode
    DescriptorSet concatDescSetK;         // For K concatenation in cache mode
    DescriptorSet concatDescSetV;         // For V concatenation in cache mode
    DescriptorSet updateCacheDescSetK;    // For K cache update
    DescriptorSet updateCacheDescSetV;    // For V cache update
    DescriptorSet scoresDescSet;
    DescriptorSet scoresCachedDescSet;    // For cached attention scores
    DescriptorSet maskDescSet;
    DescriptorSet maskCachedDescSet;      // For cached causal mask
    DescriptorSet softmaxDescSet;
    DescriptorSet weightedSumDescSet;
    DescriptorSet weightedSumCachedDescSet; // For cached weighted sum
    DescriptorSet combineDescSet;
    DescriptorSet outProjDescSet;

    // Helper struct for intermediate tensors
    struct IntermediateTensors {
        Tensor Q_flat, K_flat, V_flat;      // Projected Q, K, V
        Tensor K_full, V_full;               // K, V after concatenating with cache (if using cache)
        Tensor scores, attn_weights;
        Tensor context, context_combined;
    };

    // Private helper functions - standard mode
    IntermediateTensors allocateIntermediateBuffers(uint32_t B, uint32_t S, uint32_t D, uint32_t H, uint32_t HD);
    void computeQKVProjection(CommandBuffer& cmdBuff, const Tensor& input, IntermediateTensors& tensors,
                              const Tensor& W_q, const Tensor& W_k, const Tensor& W_v,
                              const Tensor& B_q, const Tensor& B_k, const Tensor& B_v,
                              uint32_t B, uint32_t S, uint32_t D_in, uint32_t D_out);
    void computeAttentionScores(CommandBuffer& cmdBuff, IntermediateTensors& tensors, uint32_t B, uint32_t H, uint32_t S, uint32_t HD);
    void applyCausalMaskToScores(CommandBuffer& cmdBuff, IntermediateTensors& tensors, uint32_t B, uint32_t H, uint32_t S);
    void computeSoftmax(CommandBuffer& cmdBuff, IntermediateTensors& tensors, uint32_t B, uint32_t H, uint32_t S);
    void computeWeightedSum(CommandBuffer& cmdBuff, IntermediateTensors& tensors, uint32_t B, uint32_t H, uint32_t S, uint32_t HD);
    void combineHeadsAndProject(CommandBuffer& cmdBuff, IntermediateTensors& tensors, const Tensor& W_out, const Tensor& B_out, Tensor& output,
                                uint32_t B, uint32_t S, uint32_t D, uint32_t H, uint32_t HD);

    // Private helper functions - cache mode
    IntermediateTensors allocateIntermediateBuffersCached(uint32_t B, uint32_t new_S, uint32_t total_S, uint32_t D, uint32_t H, uint32_t HD);
    void reshapeQKVForCache(CommandBuffer& cmdBuff, IntermediateTensors& tensors,
                            Tensor& Q_reshaped, Tensor& K_reshaped, Tensor& V_reshaped,
                            uint32_t B, uint32_t new_S, uint32_t H, uint32_t HD);
    void reshapeToHeads(CommandBuffer& cmdBuff, const Tensor& flat, Tensor& reshaped, DescriptorSet& descSet, uint32_t B, uint32_t S, uint32_t H, uint32_t HD);
    void concatenateWithCache(CommandBuffer& cmdBuff, IntermediateTensors& tensors,
                              uint32_t B, uint32_t H, uint32_t new_S, uint32_t cache_len, uint32_t HD);
    void updateCacheWithNewKV(CommandBuffer& cmdBuff, const Tensor& K_new, const Tensor& V_new,
                              uint32_t B, uint32_t H, uint32_t new_S, uint32_t cache_offset, uint32_t max_len, uint32_t HD);
    void computeAttentionScoresCached(CommandBuffer& cmdBuff, const Tensor& Q, const Tensor& K, Tensor& scores,
                                      uint32_t B, uint32_t H, uint32_t S_q, uint32_t S_kv, uint32_t HD);
    void applyCausalMaskCached(CommandBuffer& cmdBuff, Tensor& scores, uint32_t B, uint32_t H, uint32_t S_q, uint32_t S_kv, uint32_t cache_len);
    void computeSoftmaxCached(CommandBuffer& cmdBuff, IntermediateTensors& tensors, uint32_t B, uint32_t H, uint32_t S_q, uint32_t S_kv);
    void computeWeightedSumCached(CommandBuffer& cmdBuff, IntermediateTensors& tensors, uint32_t B, uint32_t H, uint32_t S_q, uint32_t S_kv, uint32_t HD);

public:
    MultiHeadAttentionNode(uint32_t d_in, uint32_t d_out, uint32_t num_heads);

    void prepare() override;
    void run(CommandBuffer cmdBuff) override;

    /**
     * Enable KV caching for this attention layer
     *
     * @param cache Pointer to the layer's KV cache (managed externally)
     */
    void setCache(LayerKVCache* cache);

    /**
     * Disable KV caching
     */
    void disableCache();

    /**
     * Check if caching is enabled
     */
    bool isCacheEnabled() const { return use_cache; }
};

#endif // ATTENTION_NODE_H

#include "globalContext.h"
#include "shaders.h"
#include <unordered_map>

using namespace vk;

// Global Vulkan device for all neural network operations
Device netGlobalDevice = VulkanApp::get().device();

// Global descriptor pool for all descriptor set allocations
// Each forward pass creates ~125 descriptor sets (embedding + 12 layers + final_norm + lm_head)
// For 500 token generation: ~500 tokens × 125 sets = 62,500 descriptor sets needed
// Using 100,000 maxSets to support 500+ token generation with safety margin
DescriptorPool gDestSetPool = netGlobalDevice.createDescriptorPool({
    .maxTypes = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER <= 100000},
    .maxSets = 100000
});

// Subgroup size - initialized to 32 as default, then updated by hardware detection
// Actual value is queried from VkPhysicalDeviceSubgroupProperties during device initialization
uint32_t gSubgroupSize = 32;

// ============================================================================
// Global Pipeline Cache
// ============================================================================

ComputePipeline requestPipeline(const char* src)
{
    static std::unordered_map<const char*, ComputePipeline> pipelineCache;

    auto [it, inserted] = pipelineCache.try_emplace(src);
    if (inserted)
        it->second = netGlobalDevice.createComputePipeline({src});
    return it->second;
}

// ============================================================================
// Shader Pre-compilation
// ============================================================================

void loadAllShaders()
{
    // Attention shaders
    requestPipeline(src_linear);
    requestPipeline(src_softmax);
    requestPipeline(src_qkv_projection);
    requestPipeline(src_attention_scores);
    requestPipeline(src_attention_scores_cached);
    requestPipeline(src_causal_mask);
    requestPipeline(src_causal_mask_cached);
    requestPipeline(src_weighted_sum);
    requestPipeline(src_weighted_sum_cached);
    requestPipeline(src_combine_heads);
    requestPipeline(src_reshape_to_heads);
    requestPipeline(src_update_cache);
    requestPipeline(src_concatenate_kv);

    // Transformer block shaders
    requestPipeline(src_layer_norm);
    requestPipeline(src_gelu);
    requestPipeline(src_linear_ff);
    requestPipeline(src_add);

    // Embedding shaders
    requestPipeline(src_token_embedding);
    requestPipeline(src_positional_embedding);
    requestPipeline(src_add_embeddings);

    // LM head shaders
    requestPipeline(src_lm_head);
}

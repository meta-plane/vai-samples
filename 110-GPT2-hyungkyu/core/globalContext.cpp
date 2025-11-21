#include "globalContext.h"

using namespace vk;

// Global Vulkan device for all neural network operations
Device netGlobalDevice = VulkanApp::get().device();

// Global descriptor pool for all descriptor set allocations
// Each forward pass creates ~88 descriptor sets (12 layers × 7 sets/layer + embedding + lm_head)
// For 100 token generation: ~100 tokens × 88 sets = 8,800 descriptor sets needed
// Using 10,000 maxSets to support 100+ token generation with safety margin
DescriptorPool gDestSetPool = netGlobalDevice.createDescriptorPool({
    .maxTypes = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER <= 10000},
    .maxSets = 10000
});

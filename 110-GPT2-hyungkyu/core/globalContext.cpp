#include "globalContext.h"

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

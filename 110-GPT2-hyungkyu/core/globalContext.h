#ifndef GLOBAL_CONTEXT_H
#define GLOBAL_CONTEXT_H

#include "vulkanApp.h"

// Global Vulkan device for all neural network operations
extern vk::Device netGlobalDevice;

// Global descriptor pool for all descriptor set allocations
// Configured to support large models with multiple layers
extern vk::DescriptorPool gDestSetPool;

#endif // GLOBAL_CONTEXT_H

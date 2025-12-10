#ifndef GLOBAL_CONTEXT_H
#define GLOBAL_CONTEXT_H

#include "vulkanApp.h"

// Global Vulkan device for all neural network operations
extern vk::Device netGlobalDevice;

// Global descriptor pool for all descriptor set allocations
// Configured to support large models with multiple layers
extern vk::DescriptorPool gDestSetPool;

// Subgroup size for compute shaders (queried from GPU)
// Typical values: 32 (NVIDIA), 64 (AMD), 16-32 (Intel)
extern uint32_t gSubgroupSize;

// Global shader pipeline cache and management
// Caches compiled compute pipelines to avoid redundant compilations
vk::ComputePipeline requestPipeline(const char* src);

// Pre-compile all shaders used in GPT-2 model
// Call this at startup to eliminate runtime compilation latency
void loadAllShaders();

#endif // GLOBAL_CONTEXT_H

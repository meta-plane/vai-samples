#ifndef SPIRV_HELPERS_H
#define SPIRV_HELPERS_H

#include <array>
#include <vector>
#include <vulkan/vulkan_core.h>
#include "core/vulkanApp.h"

namespace vk {
struct PipelineLayoutDesc;
}

std::vector<uint32_t> glsl2spv(VkShaderStageFlags stage, const char* shaderSource);

void* createReflectShaderModule(const std::vector<uint32_t>& spvBinary);

void destroyReflectShaderModule(void* pModule);

std::array<uint32_t, 3> extractWorkGroupSize(const void* pModule);

vk::PipelineLayoutDesc extractPipelineLayoutDesc(const void* pModule);

#endif // SPIRV_HELPERS_H


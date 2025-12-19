#include "neuralNodes.h"
#include "vulkanApp.h"

Device gDevice = VulkanApp::get().createDevice({.supportPresent = false});
DescriptorPool gDestSetPool = gDevice.createDescriptorPool({
    .maxTypes = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER <= 10000},
    .maxSets = 5000
});

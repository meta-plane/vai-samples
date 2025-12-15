#include "utils.h"
#include "../library/vulkanApp.h"


std::vector<float> downloadTensor(const Tensor& tensor)
{
    if (!tensor.numElements())
        return {};

    auto device = VulkanApp::get().device();
    const size_t byteSize = tensor.numElements() * sizeof(float);

    Buffer staging = device.createBuffer({
        .size = byteSize,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });

    device.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(staging, tensor.buffer())
        .end()
        .submit()
        .wait();

    std::vector<float> host(tensor.numElements());
    std::memcpy(host.data(), staging.map(), byteSize);
    staging.unmap();

    return host;
}
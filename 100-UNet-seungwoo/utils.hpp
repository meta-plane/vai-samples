#pragma once

#include <iostream>
#include <vector>
#include <fstream>
#include "../external/stb/stb_image.h"
#include "../external/stb/stb_image_write.h"
#include "library/vulkanApp.h"

namespace utils 
{
    typedef struct
    {
        uint32_t channels;
        uint32_t height;
        uint32_t width;
    }ImageInfo;

    template<uint32_t Channels>
    auto readImage(const char* filename)
    {
        int w, h, c0, c = Channels;
        std::vector<uint8_t> srcImage;

        if (uint8_t* input = stbi_load(filename, &w, &h, &c0, c))
        {
            srcImage.assign(input, input + w * h * c);
            stbi_image_free(input);
        }
        else
        {
            printf(stbi_failure_reason());
            fflush(stdout);
            throw;
        }

        return std::make_tuple(srcImage, (uint32_t)w, (uint32_t)h);
    }

    inline bool writeImage(const char* filename, std::vector<uint8_t>& raw, ImageInfo info)
    {
        int success = stbi_write_png(filename, info.width, info.height, info.channels, raw.data(), info.width * info.channels);
        if (!success)
        {
            std::cout << "Failed to write image: " << filename << std::endl;
            return false;
        }
		return true;
    }

    inline std::vector<float> tensor_to_cpu(const Tensor& t)
    {
        auto shape = t.shape();
        size_t elemCount = 1;
        for (auto d : shape) elemCount *= size_t(d);

        size_t byteSize = elemCount * sizeof(float);

        vk::Buffer staging = netGlobalDevice.createBuffer({
            byteSize,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            });

        netGlobalDevice.newCommandBuffer(queue_compute)
            .begin()
            .copyBuffer(staging, t.buffer())
            .end()
            .submit()
            .wait();

        std::vector<float> cpu(elemCount);
        void* mapped = staging.map();
        memcpy(cpu.data(), mapped, byteSize);
        staging.unmap();

        return cpu;
    }

    inline void write_tensor_bin(const std::string& path, const Tensor& t)
    {
        auto shape = t.shape();
        std::vector<float> data = tensor_to_cpu(t);

        std::ofstream ofs(path, std::ios::binary);
        uint32_t rank = (uint32_t)shape.size();
        ofs.write((char*)&rank, sizeof(rank));
        for (auto d : shape)
        {
            uint32_t dd = (uint32_t)d;
            ofs.write((char*)&dd, sizeof(dd));
        }
        uint64_t n = (uint64_t)data.size();
        ofs.write((char*)&n, sizeof(n));
        ofs.write((char*)data.data(), sizeof(float) * data.size());
    }

}

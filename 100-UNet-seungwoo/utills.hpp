#pragma once

#include <iostream>
#include <vector>
#include "../external/stb/stb_image.h"

namespace utils 
{
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
}

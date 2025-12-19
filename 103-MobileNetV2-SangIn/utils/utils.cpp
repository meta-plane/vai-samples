#include "utils.h"
#include <stb/stb_image.h>
#include <algorithm>
#include <random>


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
        printf("Failed to load image: %s\n", stbi_failure_reason());
        fflush(stdout);
        throw std::runtime_error("Image load failed");
    }

    return std::make_tuple(srcImage, (uint32_t)w, (uint32_t)h);
}


Tensor makeConstTensor(const std::vector<uint32_t>& shape, float val)
{
    Tensor t(shape);
    t.set(std::vector<float>(t.numElements(), val));
    return t;
}

Tensor makeConstTensor(uint32_t size, float val)
{
    Tensor t(size);
    t.set(std::vector<float>(t.numElements(), val));
    return t;
}


void softmax(float* data, int size) {
    float maxVal = data[0];
    for (int i = 1; i < size; ++i) maxVal = std::max(maxVal, data[i]);
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        data[i] = std::exp(data[i] - maxVal);
        sum += data[i];
    }
    for (int i = 0; i < size; ++i) data[i] /= sum;
}


std::vector<int> getTopK(const float* data, int size, int k) {
    std::vector<std::pair<float, int>> pairs(size);
    for (int i = 0; i < size; ++i) pairs[i] = { data[i], i };
    std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<int> result(k);
    for (int i = 0; i < k; ++i) result[i] = pairs[i].second;
    return result;
}
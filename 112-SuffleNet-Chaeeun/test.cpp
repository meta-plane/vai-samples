#include "neuralNet.h"
#include "neuralNodes.h"
#include "jsonParser.h"
#include "timeChecker.hpp"
#include <stb/stb_image.h>
#include <cstring>  // memcpy

#include "SuffleNet.h"

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

// TODO: weight initialization
Tensor eval_shufflenet(const std::vector<float>& srcImage, const JsonParser& json, uint32_t iter) // srcImage layout: [H][W][C]
{
    SuffleNet suffleNet(netGlobalDevice);

    suffleNet["conv0.weight"] = Tensor(json["layer1.0.weight"]).reshape(32, 1*3*3).permute(1, 0);
    suffleNet["conv0.bias"] = Tensor(json["layer1.0.bias"]);
    suffleNet["conv1.weight"] = Tensor(json["layer2.0.weight"]).reshape(64, 32*3*3).permute(1, 0);
    suffleNet["conv1.bias"] = Tensor(json["layer2.0.bias"]);
    suffleNet["weight"] = Tensor(json["fc.weight"]).reshape(10, 64, 7*7).permute(2, 1, 0).reshape(7*7*64, 10);
    suffleNet["bias"] = Tensor(json["fc.bias"]);
    
    Tensor result;
    Tensor inputTensor = Tensor(28, 28, 1).set(srcImage);

    for (uint32_t i = 0; i < iter; ++i)
        result = suffleNet(inputTensor)[0];

    return result;
}

void Run()
{
    void loadShaders();
    loadShaders();

    const uint32_t channels = 1;
    auto [srcImage, width, height] = readImage<channels>(PROJECT_CURRENT_DIR"/data/0.png");
    _ASSERT(width == 28 && height == 28);
    _ASSERT(width * height * channels == srcImage.size());

    std::vector<float> inputData(width * height * channels);
    for (size_t i = 0; i < srcImage.size(); ++i)
        inputData[i] = srcImage[i] / 255.0f;

    JsonParser json = JsonParser(PROJECT_CURRENT_DIR"/weights.json");

    uint32_t iter = 1;
    Tensor eval;

    {
        TimeChecker timer("(VAI) SuffleNet evaluation: {} iterations", iter);
        eval = eval_shufflenet(inputData, json, iter);
    }

    vk::Buffer outBuffer = netGlobalDevice.createBuffer({
        10 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });

    vk::Buffer evalBuffer = eval.buffer();
    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuffer, evalBuffer)
        .end()
        .submit()
        .wait();

    float data[10];
    memcpy(data, outBuffer.map(), 10 * sizeof(float));

    for(int i=0; i<10; ++i)
        printf("data[%d] = %f\n", i, data[i]);
}

#include "library/neuralNet.h"
#include "library/neuralNodes.h"
#include "networks/efficientNet.h"
#include "library/jsonParser.h"
#include "library/vulkanApp.h"
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

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
        srcImage.resize(224 * 224 * Channels, 0); // Fallback dummy
        w = 224; h = 224;
    }

    return std::make_tuple(srcImage, (uint32_t)w, (uint32_t)h);
}

std::vector<float> makeTestTensor(uint32_t resolution, uint32_t channels)
{
    const size_t numel = static_cast<size_t>(resolution) * resolution * channels;
    std::vector<float> tensor(numel);
    for (size_t i = 0; i < numel; ++i)
        tensor[i] = static_cast<float>(i % 1024) / 1023.0f; // deterministic pattern
    return tensor;
}

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

std::unique_ptr<JsonParser> tryLoadWeights(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.good())
        return nullptr;
    file.close();
    try
    {
        return std::make_unique<JsonParser>(path.c_str());
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Warn] Failed to parse weights at " << path << ": " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "[Warn] Unknown error while parsing weights at " << path << std::endl;
    }
    return nullptr;
}


Tensor eval_efficientnet(const std::vector<float>& srcImage, uint32_t W, uint32_t H, EfficientNetVersion version, const JsonParser* weights, uint32_t iter)
{
    auto device = VulkanApp::get().device();
    
    // 버전별 config 자동 생성
    std::cout << "Creating EfficientNet..." << std::endl;
    EfficientNet net(device, version, 1000);
    std::cout << "EfficientNet created." << std::endl;

    // Load weights (if JSON is valid)
    if (weights)
    {
        // TODO: populate weights
    }

    Tensor inputTensor(H, W, 3);
    inputTensor.set(srcImage);

    Tensor result;
    for (uint32_t i = 0; i < iter; ++i)
    {
        std::cout << "Running iteration " << i << "..." << std::endl;
        result = net(inputTensor)[0];
        std::cout << "Iteration " << i << " done." << std::endl;
    }

    return result;
}


void test()
{
    void loadShaders();
    loadShaders();

    std::string weightsPath = std::string(PROJECT_CURRENT_DIR) + "/weights/efficientnet-b0.json";
    auto weights = tryLoadWeights(weightsPath);
    if (weights)
        std::cout << "Loaded EfficientNet weights from " << weightsPath << std::endl;
    else
        std::cout << "Weights file not found at " << weightsPath << ", running with dummy parameters." << std::endl;

    // 원하는 버전 선택 (B0 ~ B7)
    EfficientNetVersion version = EfficientNetVersion::B0;
    
    auto config = getEfficientNetConfig(version);
    const uint32_t resolution = config.resolution;
    const uint32_t channels = 3;
    const auto floatImage = makeTestTensor(resolution, channels);

    std::cout << "Running EfficientNet " 
              << (version == EfficientNetVersion::B0 ? "B0" :
                  version == EfficientNetVersion::B1 ? "B1" :
                  version == EfficientNetVersion::B2 ? "B2" :
                  version == EfficientNetVersion::B3 ? "B3" :
                  version == EfficientNetVersion::B4 ? "B4" :
                  version == EfficientNetVersion::B5 ? "B5" :
                  version == EfficientNetVersion::B6 ? "B6" : "B7")
              << " (d=" << config.depth_multiplier 
              << ", w=" << config.width_multiplier 
              << ", r=" << config.resolution << ")..." << std::endl;
    
    std::cout << "Calling eval_efficientnet..." << std::endl;
    Tensor output = eval_efficientnet(floatImage, resolution, resolution, version, weights.get(), 1);
    std::cout << "eval_efficientnet returned." << std::endl;
    auto logits = downloadTensor(output);
    std::cout << "downloadTensor returned." << std::endl;

    if (logits.empty())
    {
        std::cout << "No output data returned.\n";
    }
    else
    {
        size_t printCount = logits.size() < 10 ? logits.size() : 10;
        for (size_t i = 0; i < printCount; ++i)
            std::cout << "logit[" << i << "] = " << logits[i] << std::endl;
    }

    std::cout << "Done." << std::endl;
}

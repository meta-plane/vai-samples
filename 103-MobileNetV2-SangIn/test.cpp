#include "library/neuralNet.h"
#include "library/neuralNodes.h"
#include "library/safeTensorsParser.h"
#include "library/vulkanApp.h"
#include "library/timeChecker.hpp"
#include "networks/MobileNetV2.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION

#include <cstring>


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

void loadWeights(MobileNetV2& net, const SafeTensorsParser& weights)
{
    // // MobileNetV2의 각 레이어에 대해 가중치 로드
    // for (uint32_t i = 0; i < net.numInvertedBottlenecks(); ++i)
    // {
    //     auto& ib = net.invertedBottleneck(i);
    //     ib["expand.weight"] = Tensor(weights["features." + std::to_string(i) + ".0.weight"]).permute(1, 0);
    //     ib["depthwise.weight"] = Tensor(weights["features." + std::to_string(i) + ".1.weight"]);
    //     ib["project.weight"] = Tensor(weights["features." + std::to_string(i) + ".2.weight"]).permute(1, 0);
    //     ib["project.bias"] = Tensor(weights["features." + std::to_string(i) + ".2.bias"]);
    // }

    // // 초기 conv 레이어
    // net["conv1.weight"] = Tensor(weights["features.0.0.weight"]).permute(1, 0);
    // net["conv1.bias"] = Tensor(weights["features.0.0.bias"]);

    // // 최종 fc 레이어
    // net["fc.weight"] = Tensor(weights["classifier.1.weight"]).permute(1, 0);
    // net["fc.bias"] = Tensor(weights["classifier.1.bias"]);
}

Tensor eval_ImageNet(const std::vector<float>& srcImage, uint32_t W, uint32_t H, const SafeTensorsParser* weights, uint8_t iter)
{
    auto device = VulkanApp::getGlobalDevice();

    std:cout << "Creating MobileNetV2..." << std::endl;
    MobileNetV2 mobileNetV2(device);
    std::cout << "MobileNetV2 created." << std::endl;

    if (weights)
    {
        loadWeights(mobileNetV2, *weights);
    }


    
    Tensor inputTensor(H, W, 3); // srcImage layout: [H][W][C]
    inputTensor.set(srcImage);   // 데이터 복사
    printf("Input Tensor Shape: [%d, %d, %d]\n", inputTensor.shape()[0], inputTensor.shape()[1], inputTensor.shape()[2]);

    Tensor result;
    for (uint32_t i = 0; i < iter; ++i)
    {
        std::cout << "Running iteration " << i << "..." << std::endl;
        result = mobileNetV2(inputTensor)[0];
        std::cout << "Iteration " << i << " done." << std::endl;
    }

    return result;
}

// smart pointer를 사용하여 SafeTensorsParser 객체를 반환하거나, 실패 시 nullptr를 반환
// smart pointer : 소유권 관리가 자동으로 이루어져 메모리 누수를 방지(명시적으로 메모리를 해제할 필요가 없음)
std::unique_ptr<SafeTensorsParser> tryLoadWeights(const std::string& path)
{
    std::ifstream file(path, std::ios::binary); // check if file exists
    if (!file.good())
        return nullptr;

    file.close();
    try
    {
        printf("✓ SafeTensors file found: %s\n", path.c_str());
        return std::make_unique<SafeTensorsParser>(path.c_str()); // smart pointer 반환
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
void test()
{
    void loadShaders();
    loadShaders();

    // Load model weights
    std::string weightsPath = std::string(PROJECT_CURRENT_DIR) + "/weights/mobelenet_v2_imagenet1k.safetensors";

    auto weights = tryLoadWeights(weightsPath);
    if (weights)
        std::cout << "Loaded EfficientNet weights from " << weightsPath << std::endl;
    else
        std::cout << "Weights file not found at " << weightsPath << ", running with dummy parameters." << std::endl;

    // Load image and normalize
    const uint8_t channels = 3U;
    const uint32_t resolution = 224U;
    std::string imagePath = std::string(PROJECT_CURRENT_DIR) + "/utils/shark.png";
    std::cout << "Loading image from " << imagePath << "..." << std::endl;
    auto [srcImage, width, height] = readImage<channels>(imagePath.c_str(), resolution, resolution); // (H, W, C) == (224, 224, 3)

    std::vector<float> inputData(width * height * channels);
    for (size_t i = 0; i < srcImage.size(); ++i)
        inputData[i] = srcImage[i] / 255.0f;


    // Eval MobileNetV2
    const uint8_t iter = 1U;

    std::cout << "Calling eval_ImageNet..." << std::endl;
    Tensor output = eval_ImageNet(inputData, resolution, resolution, weights.get(), iter);
    std::cout << "eval_efficientnet returned." << std::endl;
    auto logits = downloadTensor(output);
    std::cout << "downloadTensor returned." << std::endl;


    uint32_t iter = 1;
    Tensor eval;

    {
        TimeChecker timer("(VAI) MNIST evaluation: {} iterations", iter);
        eval = eval_ImageNet(inputData, json, iter);
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

    // float data[10];
    // memcpy(data, outBuffer.map(), 10 * sizeof(float));

    // for(int i=0; i<10; ++i)
    //     printf("data[%d] = %f\n", i, data[i]);
}

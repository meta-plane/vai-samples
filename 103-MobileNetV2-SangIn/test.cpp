#include "library/neuralNet.h"
#include "library/neuralNodes.h"
#include "library/safeTensorsParser.h"
#include "library/vulkanApp.h"
#include "library/timeChecker.hpp"
#include "models/MobileNetV2.h"
#include "utils/utils.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION

#include <cstring>
#include <iostream>
#include <filesystem>
#include <chrono>


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

std::vector<float> preprocess(const std::vector<uint8_t>& image, int w, int h) {
    std::vector<float> result(w * h * 3);
    const float mean[] = { 0.485f, 0.456f, 0.406f };
    const float std[] = { 0.229f, 0.224f, 0.225f };

    for (int i = 0; i < w * h; ++i) {
        for (int c = 0; c < 3; ++c) {
            float val = image[i * 3 + c] / 255.0f;
            result[i * 3 + c] = (val - mean[c]) / std[c];
        }
    }

    return result;
}

void loadWeights(MobileNetV2& net, const SafeTensorsParser& weights)
{
    std::cout << "Loading weights into MobileNetV2..." << std::endl;

    // Helper lambda to load tensor by name
    auto loadTensor = [&](const std::string& cppName, const std::string& ptName)
    {
        try {
            auto ref = weights[ptName];
            Tensor tensor(ref);
            auto shape = tensor.shape();
            
            // find는 문자열 내에서 특정 부분 문자열의 위치를 반환, 없으면 npos 반환
            if (cppName.find("depthwiseConv.weight") != std::string::npos) {
                if (shape.size() == 4 && shape[1] == 1) {
                    tensor.reshape(shape[0], shape[2], shape[3]); // [C_out, 1, K, K] -> [C_out, K, K]
                    tensor.permute(1, 2, 0); // make it [K, K, C_out]

                    auto new_shape = tensor.shape();
                    tensor.reshape(new_shape[0] * new_shape[1], new_shape[2]); // flatten to [K*K, C_out]
                }
            }
            else if (cppName.find(".weight") != std::string::npos) {
                if (shape.size() == 4) {
                    tensor.permute(1, 2, 3, 0); // [C_out, C_in, K, K] -> [C_in, K, K, C_out]
                    const auto& newShape = tensor.shape();
                    tensor.reshape(newShape[0] * newShape[1] * newShape[2], newShape[3]); // [C_in, K, K, C_out] -> [C_in * K * K, C_out]
                }
                else if (shape.size() == 2) {
                    tensor.permute(1, 0); // [C_out, C_in] -> [C_in, C_out]
                }
            }
            
            net[cppName] = std::move(tensor);
            net.setNodeNameFromParam(cppName);

        } catch (const std::exception& e) {
            std::cerr << "[Failed to load] " << cppName << " <- " << ptName << ": " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[Failed to load] " << cppName << " to " << ptName << " (unknown error)" << std::endl;
        }
    };
     
    auto loadBN = [&](const std::string& cppPrefix, const std::string& ptPrefix)
    {
        loadTensor(cppPrefix + ".mean", ptPrefix + ".running_mean");
        loadTensor(cppPrefix + ".variance", ptPrefix + ".running_var");
        loadTensor(cppPrefix + ".gamma", ptPrefix + ".weight");
        loadTensor(cppPrefix + ".beta", ptPrefix + ".bias");
    };

    auto loadIRB = [&](int blockIdx)
    {
        const int ptFeatIdx = blockIdx + 1;  // PyTorch features 인덱스
        const std::string cppBase = "features." + std::to_string(blockIdx);
        const std::string ptBase  = "features." + std::to_string(ptFeatIdx) + ".conv";

        // 첫 번째 IRB (features.1): expand 없음
        if (ptFeatIdx == 1)
        {
            // depthwise: conv.0.(0=conv,1=bn)
            loadTensor(cppBase + ".dwConvBNReLU6.depthwiseConv.weight", ptBase  + ".0.0.weight");
            loadBN(cppBase + ".dwConvBNReLU6.bn", ptBase + ".0.1");

            // pointwise: conv.1.(1=conv,2=bn)
            loadTensor(cppBase + ".pwConvBN.pointwiseConv.weight", ptBase  + ".1.weight");
            loadBN(cppBase + ".pwConvBN.bn", ptBase + ".2");
        }

        // 나머지 IRB들 (expand_ratio > 1): expand + depthwise + project
        else
        {             
            // expand: conv.0.(0=conv,1=bn)
            loadTensor(cppBase + ".pwConvBNReLU6.pointwiseConv.weight", ptBase  + ".0.0.weight");
            loadBN(cppBase + ".pwConvBNReLU6.bn", ptBase + ".0.1");

            // depthwise: conv.1.(0=conv,1=bn)
            loadTensor(cppBase + ".dwConvBNReLU6.depthwiseConv.weight", ptBase  + ".1.0.weight");
            loadBN(cppBase + ".dwConvBNReLU6.bn", ptBase + ".1.1");

            // project(pointwise): conv.2, conv.3(BN)
            loadTensor(cppBase + ".pwConvBN.pointwiseConv.weight", ptBase  + ".2.weight");
            loadBN(cppBase + ".pwConvBN.bn", ptBase + ".3");
        }
    };

    // 1) Stem
    loadTensor("stem.conv.weight", "features.0.0.weight");
    loadBN("stem.bn", "features.0.1");

    // 2) Inverted Residual Blocks (net.invertedResidualBlocks.size() == #IRB)
    for (size_t i = 0; i < net.blocks().size(); ++i) {
        loadIRB(i);
    }

    // 3) Final layers
    loadTensor("finalConv.pointwiseConv.weight", "features.18.0.weight");
    loadBN("finalConv.bn", "features.18.1");

    loadTensor("classifier.weight", "classifier.1.weight");
    loadTensor("classifier.bias", "classifier.1.bias");

    std::cout << "Weights loading completed." << std::endl;
}

Tensor eval_ImageNet(const std::vector<float>& srcImage, uint32_t W, uint32_t H, const SafeTensorsParser* weights, uint8_t iter, bool enableLayerTiming = false)
{
    auto device = VulkanApp::get().device();

    std::cout << "Creating MobileNetV2..." << std::endl;
    MobileNetV2 mobileNetV2(device);
    std::cout << "MobileNetV2 created." << std::endl;

    if (weights)
    {
        loadWeights(mobileNetV2, *weights);
    }

    Tensor inputTensor(H, W, 3); // srcImage layout: [H][W][C]
    inputTensor.set(srcImage);   // data copy
    printf("Input Tensor Shape: [%d, %d, %d]\n", inputTensor.shape()[0], inputTensor.shape()[1], inputTensor.shape()[2]);

    // Warmup phase
    uint32_t warmupIter = std::min(3U, static_cast<uint32_t>(iter));
    std::cout << "\n=== Warmup Phase (first " << warmupIter << " iterations) ===" << std::endl;
    Tensor result;
    for (uint32_t i = 0; i < warmupIter; ++i)
    {
        std::cout << "Warmup iteration " << i + 1 << "/" << warmupIter << "..." << std::endl;
        result = mobileNetV2(inputTensor)[0];
    }
    std::cout << "Warmup completed.\n" << std::endl;

    // Benchmark phase
    if (iter > warmupIter)
    {
        const uint32_t benchmarkIter = iter - warmupIter;
        std::cout << "=== Benchmark Phase (" << benchmarkIter << " iterations) ===" << std::endl;

        auto startTime = std::chrono::high_resolution_clock::now();

        for (uint32_t i = 0; i < benchmarkIter; ++i)
        {
            std::cout << "Benchmark iteration " << i + 1 << "/" << benchmarkIter << "..." << std::endl;
            result = mobileNetV2(inputTensor)[0];
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        std::cout << "\n=== Benchmark Results ===" << std::endl;
        std::cout << "Total iterations: " << benchmarkIter << std::endl;
        std::cout << "Total time: " << duration.count() << " ms" << std::endl;
        std::cout << "Average time per iteration: " << (float)duration.count() / benchmarkIter << " ms" << std::endl;
        std::cout << "=========================\n" << std::endl;
    }

    // Layer timing measurement (separate from benchmark)
    if (enableLayerTiming)
    {
        std::cout << "\n=== Running Layer Timing Measurement ===" << std::endl;
        mobileNetV2.setLayerTiming(true);
        result = mobileNetV2(inputTensor)[0];
        mobileNetV2.setLayerTiming(false);

        const auto& timings = mobileNetV2.getLayerTimings();
        if (!timings.empty())
        {
            std::cout << "\n=== Layer-wise Timing ===" << std::endl;
            double totalLayerTime = 0.0;
            for (const auto& [nodeName, timeMs] : timings)
            {
                if (!nodeName.empty() && nodeName != "noname")
                {
                    std::cout << "  " << nodeName << ": " << timeMs << " ms" << std::endl;
                    totalLayerTime += timeMs;
                }
            }
            std::cout << "Total layer time: " << totalLayerTime << " ms" << std::endl;
            std::cout << "=========================\n" << std::endl;
        }
    }

    return result;
}

// smart pointer를 사용하여 SafeTensorsParser 객체를 반환하거나, 실패 시 nullptr를 반환
// smart pointer : 소유권 관리가 자동으로 이루어져 메모리 누수를 방지(명시적으로 메모리를 해제할 필요가 없음)
std::unique_ptr<SafeTensorsParser> tryLoadWeights(const std::string& path)
{
    namespace fs = std::filesystem;

    if (!fs::exists(path)) {
        std::cerr << "[Warn] Weight file not found: " << path << "\n";
        return nullptr;
    }

    try {
        std::cout << "✓ SafeTensors file found: " << path << "\n";
        return std::make_unique<SafeTensorsParser>(path.c_str());
    }
    catch (const std::exception& e) {
        std::cerr << "[Warn] Failed to parse weights at " << path
            << ": " << e.what() << "\n";
    }
    catch (...) {
        std::cerr << "[Warn] Unknown error while parsing weights at "
            << path << "\n";
    }
    return nullptr;
}


void test()
{
    void loadShaders();
    loadShaders();

    // Load model weights
    std::string weightsPath = std::string(PROJECT_CURRENT_DIR) + "/weights/mobilenet_v2_imagenet1k.safetensors";

    auto weights = tryLoadWeights(weightsPath);
    if (weights)
        std::cout << "Loaded EfficientNet weights from " << weightsPath << std::endl;
    else
        std::cout << "Weights file not found at " << weightsPath << ", running with dummy parameters." << std::endl;

    // Load image and normalize
    const uint8_t channels = 3U;
    const uint32_t resolution = 224U;
    std::string imagePath = std::string(PROJECT_CURRENT_DIR) + "/img/shark.png";

    std::cout << "Loading image from " << imagePath << "..." << std::endl;
    auto [srcImage, width, height] = readImage<channels>(imagePath.c_str()); // (H, W, C) == (224, 224, 3)
    std::cout << "Image Size: " << width << "x" << height << std::endl;
    _ASSERT(width == resolution && height == resolution);

    auto inputData = preprocess(srcImage, resolution, resolution);

    // Eval MobileNetV2
    const uint8_t iter = 10U;  // Total iterations (first 3 for warmup, rest for benchmark)
    const bool enableLayerTiming = false;  // Enable layer-wise timing measurements

    std::cout << "Calling eval_ImageNet..." << std::endl;
    Tensor output = eval_ImageNet(inputData, resolution, resolution, weights.get(), iter, enableLayerTiming);
    std::cout << "eval_efficientnet returned." << std::endl;
    auto logits = downloadTensor(output);
    std::cout << "downloadTensor returned." << std::endl;

    if (logits.empty())
    {
        std::cout << "No output data returned.\n";
    }
    else
    {
        // show top-5 idx and logit
        std::vector<std::pair<int, float>> idxLogits;
        for (size_t i = 0; i < logits.size(); ++i) {
            idxLogits.push_back({i, logits[i]});
        }
        std::sort(idxLogits.begin(), idxLogits.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
        
        std::cout << "Top-5 Results:\n";
        for (int i = 0; i < 5 && i < (int)idxLogits.size(); ++i) {
            std::cout << "  " << i+1 << ". Class " << idxLogits[i].first 
                  << ": " << idxLogits[i].second << "\n";
        }
    }

    std::cout << "Done." << std::endl;
}

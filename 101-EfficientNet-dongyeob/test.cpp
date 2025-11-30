#include "library/neuralNet.h"
#include "library/neuralNodes.h"
#include "networks/efficientNet.h"
#include "library/safeTensorsParser.h"
#include "library/vulkanApp.h"
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize2.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <algorithm>

template<uint32_t Channels>
auto readImage(const char* filename, int targetW, int targetH)
{
    int w, h, c;
    std::vector<uint8_t> outputImage(targetW * targetH * Channels);

    if (uint8_t* input = stbi_load(filename, &w, &h, &c, Channels))
    {
        printf("✓ Successfully loaded image: %s (%dx%d, %d channels)\n", filename, w, h, c);
        if (w != targetW || h != targetH) {
            stbir_resize_uint8_linear(input, w, h, 0, outputImage.data(), targetW, targetH, 0, (stbir_pixel_layout)Channels);
            printf("  Resized from %dx%d to %dx%d\n", w, h, targetW, targetH);
        } else {
            std::memcpy(outputImage.data(), input, w * h * Channels);
        }
        stbi_image_free(input);
        
        printf("  First pixel RGB: (%d, %d, %d)\n", outputImage[0], outputImage[1], outputImage[2]);
    }
    else
    {
        printf("✗ Failed to load image: %s (reason: %s)\n", filename, stbi_failure_reason());
        printf("  Using DUMMY DATA fallback\n");
        // Fallback dummy
        for (size_t i = 0; i < outputImage.size(); ++i) outputImage[i] = (uint8_t)(i % 255);
    }

    return outputImage;
}

std::vector<float> preprocess(const std::vector<uint8_t>& image, int w, int h) {
    std::vector<float> result(w * h * 3);
    const float mean[] = {0.485f, 0.456f, 0.406f};
    const float std[] = {0.229f, 0.224f, 0.225f};
    
    for (int i = 0; i < w * h; ++i) {
        for (int c = 0; c < 3; ++c) {
            float val = image[i * 3 + c] / 255.0f;
            result[i * 3 + c] = (val - mean[c]) / std[c];
        }
    }
    return result;
}

std::vector<std::string> loadLabels(const std::string& path) {
    std::vector<std::string> labels;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        labels.push_back(line);
    }
    return labels;
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

std::unique_ptr<SafeTensorsParser> tryLoadWeights(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.good())
        return nullptr;
    file.close();
    try
    {
        printf("✓ SafeTensors file found: %s\n", path.c_str());
        return std::make_unique<SafeTensorsParser>(path.c_str());
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


// Helper to load BN weights
void loadBN(Tensor& bnNodeTensor, const SafeTensorsParser& parser, const std::string& prefix)
{
}

void loadWeights(EfficientNet& net, const SafeTensorsParser& parser, EfficientNetVersion version)
{
    std::cout << "Loading weights..." << std::endl;

    auto loadTensor = [&](const std::string& cppName, const std::string& ptName)
    {
        try {
            auto ref = parser[ptName];
            Tensor t(ref);
            auto shape = t.shape();
            
            if (cppName.find("depthwise.weight") != std::string::npos) {
                if (shape.size() == 4 && shape[1] == 1) {
                    t.reshape(shape[0], shape[2], shape[3]);
                }
            }
            else if (cppName.find(".weight") != std::string::npos) {
                if (shape.size() == 4) {
                    t.permute(1, 2, 3, 0);
                    const auto& newShape = t.shape();
                    t.reshape(newShape[0] * newShape[1] * newShape[2], newShape[3]);
                }
                else if (shape.size() == 2) {
                    t.permute(1, 0);
                }
            }
            
            net[cppName] = std::move(t);
            
            // Debug output
            const auto& finalShape = net[cppName].shape();
            printf("✓ Loaded %-40s <- %-40s shape=[", cppName.c_str(), ptName.c_str());
            for(size_t i=0; i<finalShape.size(); ++i) printf("%d%s", finalShape[i], i<finalShape.size()-1?"x":"");
            printf("]\n");
            
        } catch (const std::exception& e) {
            std::cerr << "Failed to load " << ptName << " to " << cppName << ": " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Failed to load " << ptName << " to " << cppName << " (unknown error)" << std::endl;
        }
    };

    auto loadBN = [&](const std::string& cppPrefix, const std::string& ptPrefix)
    {
        loadTensor(cppPrefix + ".mean", ptPrefix + ".running_mean");
        loadTensor(cppPrefix + ".variance", ptPrefix + ".running_var");
        loadTensor(cppPrefix + ".gamma", ptPrefix + ".weight");
        loadTensor(cppPrefix + ".beta", ptPrefix + ".bias");
    };

    // 1. Stem
    loadTensor("stem.weight", "features.0.0.weight");
    loadBN("stem", "features.0.1");

    // 2. Blocks
    struct BlockConfig { int repeats; int expand_ratio; };
    std::vector<BlockConfig> stages;
    
    // Base B0 repeats
    std::vector<int> baseRepeats = {1, 2, 2, 3, 3, 4, 1};
    std::vector<int> expandRatios = {1, 6, 6, 6, 6, 6, 6};
    
    float depth_mult = 1.0f;
    switch(version) {
        case EfficientNetVersion::B0: depth_mult = 1.0f; break;
        case EfficientNetVersion::B1: depth_mult = 1.1f; break;
        case EfficientNetVersion::B2: depth_mult = 1.2f; break;
        case EfficientNetVersion::B3: depth_mult = 1.4f; break;
        case EfficientNetVersion::B4: depth_mult = 1.8f; break;
        case EfficientNetVersion::B5: depth_mult = 2.0f; break;
        case EfficientNetVersion::B6: depth_mult = 2.2f; break;
        case EfficientNetVersion::B7: depth_mult = 2.6f; break;
    }

    int globalBlockIdx = 0;
    for (size_t i = 0; i < baseRepeats.size(); ++i)
    {
        int repeats = static_cast<int>(std::ceil(baseRepeats[i] * depth_mult));
        int expand_ratio = expandRatios[i];
        
        for (int r = 0; r < repeats; ++r)
        {
            std::string cppBlock = "blocks." + std::to_string(globalBlockIdx);
            std::string ptBlock = "features." + std::to_string(i + 1) + "." + std::to_string(r) + ".block";
            
            int ptLayerIdx = 0;

            // Expand
            if (expand_ratio != 1)
            {
                loadTensor(cppBlock + ".expand.weight", ptBlock + "." + std::to_string(ptLayerIdx) + ".0.weight");
                loadBN(cppBlock + ".expand", ptBlock + "." + std::to_string(ptLayerIdx) + ".1");
                ptLayerIdx++;
            }

            // Depthwise
            loadTensor(cppBlock + ".depthwise.weight", ptBlock + "." + std::to_string(ptLayerIdx) + ".0.weight");
            loadBN(cppBlock + ".depthwise", ptBlock + "." + std::to_string(ptLayerIdx) + ".1");
            ptLayerIdx++;

            // Squeeze Excite
            loadTensor(cppBlock + ".se.weight_reduce", ptBlock + "." + std::to_string(ptLayerIdx) + ".fc1.weight");
            loadTensor(cppBlock + ".se.bias_reduce", ptBlock + "." + std::to_string(ptLayerIdx) + ".fc1.bias");
            loadTensor(cppBlock + ".se.weight_expand", ptBlock + "." + std::to_string(ptLayerIdx) + ".fc2.weight");
            loadTensor(cppBlock + ".se.bias_expand", ptBlock + "." + std::to_string(ptLayerIdx) + ".fc2.bias");
            ptLayerIdx++;

            // Project
            loadTensor(cppBlock + ".project.weight", ptBlock + "." + std::to_string(ptLayerIdx) + ".0.weight");
            loadBN(cppBlock + ".projectBN", ptBlock + "." + std::to_string(ptLayerIdx) + ".1");
            
            globalBlockIdx++;
        }
    }

    // 3. Head
    loadTensor("head.weight", "features.8.0.weight");
    loadBN("head", "features.8.1");

    // 4. Classifier
    loadTensor("classifier.weight", "classifier.1.weight");
    loadTensor("classifier.bias", "classifier.1.bias");
    
    printf("Weights loaded.\n");
    fflush(stdout);
}

Tensor eval_efficientnet(const std::vector<float>& srcImage, uint32_t W, uint32_t H, EfficientNetVersion version, const SafeTensorsParser* weights, uint32_t iter)
{
    auto device = VulkanApp::get().device();
    
    std::cout << "Creating EfficientNet..." << std::endl;
    EfficientNet net(device, version, 1000);
    std::cout << "EfficientNet created." << std::endl;

    if (weights)
    {
        loadWeights(net, *weights, version);
    }

    Tensor inputTensor(H, W, 3);
    inputTensor.set(srcImage);
    printf("Input Tensor Shape: [%d, %d, %d]\n", inputTensor.shape()[0], inputTensor.shape()[1], inputTensor.shape()[2]);

    Tensor result;
    for (uint32_t i = 0; i < iter; ++i)
    {
        std::cout << "Running iteration " << i << "..." << std::endl;
        result = net(inputTensor)[0];
        std::cout << "Iteration " << i << " done." << std::endl;
    }

    return result;
}


void test(const std::string& versionStr)
{
    void loadShaders();
    loadShaders();

    EfficientNetVersion version = EfficientNetVersion::B0;
    std::string selectedVersion = versionStr;

    if (selectedVersion.empty())
    {
        std::cout << "Select EfficientNet version (B0-B7) [default: B0]: ";
        std::string input;
        std::getline(std::cin, input);
        if (!input.empty())
            selectedVersion = input;
    }

    std::transform(selectedVersion.begin(), selectedVersion.end(), selectedVersion.begin(), ::toupper);

    if (selectedVersion == "B0") version = EfficientNetVersion::B0;
    else if (selectedVersion == "B1") version = EfficientNetVersion::B1;
    else if (selectedVersion == "B2") version = EfficientNetVersion::B2;
    else if (selectedVersion == "B3") version = EfficientNetVersion::B3;
    else if (selectedVersion == "B4") version = EfficientNetVersion::B4;
    else if (selectedVersion == "B5") version = EfficientNetVersion::B5;
    else if (selectedVersion == "B6") version = EfficientNetVersion::B6;
    else if (selectedVersion == "B7") version = EfficientNetVersion::B7;
    else 
    {
        if (!selectedVersion.empty())
            std::cout << "Invalid version '" << selectedVersion << "', defaulting to B0." << std::endl;
        version = EfficientNetVersion::B0;
        selectedVersion = "B0";
    }

    std::string weightsPath = std::string(PROJECT_CURRENT_DIR) + "/weights/efficientnet-" + (selectedVersion.empty() ? "b0" : selectedVersion) + ".safetensors";
    std::transform(weightsPath.begin(), weightsPath.end(), weightsPath.begin(), ::tolower);
    
    auto weights = tryLoadWeights(weightsPath);
    if (weights)
        std::cout << "Loaded EfficientNet weights from " << weightsPath << std::endl;
    else
        std::cout << "Weights file not found at " << weightsPath << ", running with dummy parameters." << std::endl;
    
    auto config = getEfficientNetConfig(version);
    const uint32_t resolution = config.resolution;
    
    std::string imagePath = std::string(PROJECT_CURRENT_DIR) + "/assets/panda.jpg";
    std::cout << "Loading image from " << imagePath << "..." << std::endl;
    auto rawImage = readImage<3>(imagePath.c_str(), resolution, resolution);
    auto inputData = preprocess(rawImage, resolution, resolution);

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
    Tensor output = eval_efficientnet(inputData, resolution, resolution, version, weights.get(), 1);
    std::cout << "eval_efficientnet returned." << std::endl;
    auto logits = downloadTensor(output);
    std::cout << "downloadTensor returned." << std::endl;

    if (logits.empty())
    {
        std::cout << "No output data returned.\n";
    }
    else
    {
        std::string labelsPath = std::string(PROJECT_CURRENT_DIR) + "/assets/imagenet_classes.txt";
        auto labels = loadLabels(labelsPath);
        
        std::vector<std::pair<float, int>> scores;
        for (size_t i = 0; i < logits.size(); ++i)
            scores.push_back({logits[i], (int)i});
            
        std::sort(scores.rbegin(), scores.rend());
        
        std::cout << "Top 5 predictions:" << std::endl;
        for (int i = 0; i < 5 && i < scores.size(); ++i) {
            std::string label = (scores[i].second < labels.size()) ? labels[scores[i].second] : "Unknown";
            std::cout << i + 1 << ": " << label << " (" << scores[i].first << ")" << std::endl;
        }
    }

    std::cout << "Done." << std::endl;
}

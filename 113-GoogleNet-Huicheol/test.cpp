#include "googleNet.h"
#include "neuralNodes.h"
#include "jsonParser.h"
#include "safeTensorsParser.h"
#include <stb/stb_image.h>
#include <vector>
#include <iostream>
#include <memory>
#include <tuple>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <string>

// Helper to read image (simplified from 11-mnist-refactor)
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
        std::cerr << "Failed to load image: " << filename << std::endl;
        throw std::runtime_error("Image load failed");
    }

    return std::make_tuple(srcImage, (uint32_t)w, (uint32_t)h);
}

std::vector<std::string> loadLabels(const char* path, size_t expect)
{
    std::ifstream ifs(path);
    if (!ifs.is_open())
        return {};

    std::vector<std::string> labels;
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back(); // handle CRLF
        if (line.empty()) continue; // skip blank lines
        labels.push_back(line);
    }

    // Some label files include a background class; drop it if count == expect+1
    if (expect && labels.size() == expect + 1) {
        if (labels.front() == "background" || labels.front() == "BACKGROUND")
            labels.erase(labels.begin());
    }

    if (!expect || labels.size() == expect)
        return labels;

    std::cerr << "Label count mismatch (" << labels.size() << " vs " << expect << "), ignoring labels.\n";
    return {};
}

void test()
{
    auto runStage = [](const char* label, auto&& fn)
    {
        try { fn(); }
        catch (VkResult vr) { std::cerr << "[" << label << "] VkResult: " << vkResult2String(vr) << " (0x" << std::hex << vr << std::dec << ")" << std::endl; throw; }
        catch (const std::exception& e) { std::cerr << "[" << label << "] Exception: " << e.what() << std::endl; throw; }
    };

    // Initialize Vulkan shaders
    void loadShaders();
    runStage("loadShaders", [&]{ loadShaders(); });

    std::cout << "Running GoogleNet test..." << std::endl;

    // Load image
    const uint32_t channels = 3;
    uint32_t H = 32, W = 32;    // fallback
    std::vector<uint8_t> srcImage;
    try
    {
        auto loaded = readImage<channels>(PROJECT_ROOT_DIR"/113-GoogleNet-Huicheol/data/cat.jpg");
        srcImage = std::get<0>(loaded);
        W = std::get<1>(loaded);
        H = std::get<2>(loaded);
        std::cout << "Loaded cat.jpg (" << W << "x" << H << ")" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cout << "cat.jpg not found, using zero image: " << e.what() << std::endl;
        srcImage.assign(H * W * channels, 0);
    }

    Tensor inputTensor(H, W, 3);
    std::vector<float> inputData(H * W * 3);
    for (size_t i = 0; i < inputData.size(); ++i)
        inputData[i] = (float)srcImage[i % srcImage.size()] / 255.0f;

    inputTensor.set(std::move(inputData));

    // Run inference
    std::unique_ptr<SafeTensorsParser> st;
    try
    {
        st = std::make_unique<SafeTensorsParser>(PROJECT_CURRENT_DIR"/weights.safetensors");
        std::cout << "Loaded weights.safetensors" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cout << "weights.safetensors not loaded: " << e.what() << std::endl;
    }

    std::unique_ptr<JsonParser> weights;
    try 
    {
        weights = std::make_unique<JsonParser>(PROJECT_CURRENT_DIR"/weights.json");
        std::cout << "Loaded weights.json" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cout << "weights.json not loaded, using zero-initialized weights: " << e.what() << std::endl;
    }

    GoogleNet googleNet(netGlobalDevice);
    std::cout << "GoogleNet constructed" << std::endl;
    runStage("loadWeights", [&]{ googleNet.loadWeights(weights.get(), st.get()); });
    std::cout << "Weights loaded (or zero-initialized)" << std::endl;
    
    // We don't have weights, so we just run the graph to verify structure
    std::cout << "Graph constructed. Running inference..." << std::endl;
    
    // Optional label map (one label per line)
    auto labels = loadLabels(PROJECT_ROOT_DIR"/113-GoogleNet-Huicheol/imagenet_labels.txt", 1000);

    runStage("inference", [&]{
        auto results = googleNet(inputTensor);
        Tensor& result = results[0];
        
        // Print output shape
        auto shape = result.shape();
        std::cout << "Output shape: [";
        for(auto s : shape) std::cout << s << " ";
        std::cout << "]" << std::endl;
        
        // Print first few values
        const size_t byteSize = result.numElements() * sizeof(float);
        vk::Buffer hostBuffer = netGlobalDevice.createBuffer({
            .size = byteSize,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        });

        netGlobalDevice.newCommandBuffer(queue_compute)
            .begin()
            .copyBuffer(hostBuffer, result.buffer())
            .end()
            .submit()
            .wait();

        const float* data = reinterpret_cast<const float*>(hostBuffer.map());
        const size_t N = result.numElements();

        std::cout << "First 10 output values: ";
        for (size_t i = 0; i < 10 && i < N; ++i) std::cout << data[i] << " ";
        std::cout << std::endl;

        // Softmax + Top-5 (CPU-side)
        std::vector<float> logits(data, data + N);
        float maxLogit = *std::max_element(logits.begin(), logits.end());
        float denom = 0.0f;
        for (float& v : logits) { v = std::exp(v - maxLogit); denom += v; }
        for (float& v : logits) v /= denom;

        std::vector<size_t> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        const size_t k = std::min<size_t>(5, N);
        std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
            [&](size_t a, size_t b){ return logits[a] > logits[b]; });

        std::cout << "Top-" << k << " predictions (idx: prob";
        if (!labels.empty()) std::cout << " / label";
        std::cout << "): ";
        for (size_t i = 0; i < k; ++i) {
            size_t idx = indices[i];
            std::cout << idx << ":" << logits[idx];
            if (!labels.empty()) std::cout << " (" << labels[idx] << ")";
            std::cout << " ";
        }
        std::cout << std::endl;
    });

    std::cout << "Test finished." << std::endl;
}

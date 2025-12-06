#include "googleNet.h"
#include "neuralNodes.h"
#include "jsonParser.h"
#include "safeTensorsParser.h"
#include <stb/stb_image.h>
#include <vector>
#include <iostream>
#include <memory>
#include <tuple>

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
    uint32_t H = 32, W = 32;    // original: 224 x 224
    std::vector<uint8_t> srcImage;
    try
    {
        auto loaded = readImage<channels>(PROJECT_ROOT_DIR"/113-GoogleNet-Huicheol/data/cat.jpg");
        srcImage = std::get<0>(loaded);
        std::cout << "Loaded cat.jpg (" << std::get<1>(loaded) << "x" << std::get<2>(loaded) << ")" << std::endl;
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
        std::cout << "First 10 output values: ";
        for(int i=0; i<10 && i<result.numElements(); ++i) std::cout << data[i] << " ";
        std::cout << std::endl;
    });

    std::cout << "Test finished." << std::endl;
}

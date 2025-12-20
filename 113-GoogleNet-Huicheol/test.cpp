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
        auto loaded = readImage<channels>(PROJECT_ROOT_DIR"/113-GoogleNet-Huicheol/data/cat_720p.jpg");
        srcImage = std::get<0>(loaded);
        W = std::get<1>(loaded);
        H = std::get<2>(loaded);
        std::cout << "Loaded cat_720p.jpg (" << W << "x" << H << ")" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cout << "cat_720p.jpg not found, using zero image: " << e.what() << std::endl;
        srcImage.assign(H * W * channels, 0);
    }

    // Resize -> center crop to match torchvision eval: resize so min side=256, then 224x224 center crop
    auto resizeBilinear = [](const std::vector<uint8_t>& src, uint32_t w, uint32_t h, uint32_t c, uint32_t newW, uint32_t newH)
    {
        std::vector<uint8_t> dst(newW * newH * c);
        for (uint32_t y = 0; y < newH; ++y) {
            float sy = (float(y) + 0.5f) * float(h) / float(newH) - 0.5f;
            uint32_t y0 = (uint32_t)std::max(0, std::min<int>(h - 1, int(floor(sy))));
            uint32_t y1 = std::min(h - 1, y0 + 1);
            float wy = sy - float(y0);
            for (uint32_t x = 0; x < newW; ++x) {
                float sx = (float(x) + 0.5f) * float(w) / float(newW) - 0.5f;
                uint32_t x0 = (uint32_t)std::max(0, std::min<int>(w - 1, int(floor(sx))));
                uint32_t x1 = std::min(w - 1, x0 + 1);
                float wx = sx - float(x0);
                for (uint32_t ch = 0; ch < c; ++ch) {
                    auto idx = [&](uint32_t X, uint32_t Y) { return (Y * w + X) * c + ch; };
                    float v00 = src[idx(x0, y0)];
                    float v01 = src[idx(x1, y0)];
                    float v10 = src[idx(x0, y1)];
                    float v11 = src[idx(x1, y1)];
                    float v0 = v00 + (v01 - v00) * wx;
                    float v1 = v10 + (v11 - v10) * wx;
                    float v = v0 + (v1 - v0) * wy;
                    dst[(y * newW + x) * c + ch] = (uint8_t)std::clamp(int(v + 0.5f), 0, 255);
                }
            }
        }
        return dst;
    };

    auto centerCrop = [](const std::vector<uint8_t>& src, uint32_t w, uint32_t h, uint32_t c, uint32_t cw, uint32_t ch)
    {
        std::vector<uint8_t> dst(cw * ch * c);
        uint32_t x0 = (w > cw) ? (w - cw) / 2 : 0;
        uint32_t y0 = (h > ch) ? (h - ch) / 2 : 0;
        for (uint32_t y = 0; y < ch; ++y) {
            for (uint32_t x = 0; x < cw; ++x) {
                for (uint32_t chn = 0; chn < c; ++chn) {
                    dst[(y * cw + x) * c + chn] = src[((y + y0) * w + (x + x0)) * c + chn];
                }
            }
        }
        return dst;
    };

    if (!srcImage.empty()) {
        // If already 224x224, keep as-is (avoids double scaling on pre-resized images).
        if (!(W == 224 && H == 224)) {
            uint32_t minSide = std::min(W, H);
            float scale = 256.0f / float(minSide);
            uint32_t newW = uint32_t(std::round(W * scale));
            uint32_t newH = uint32_t(std::round(H * scale));
            auto resized = resizeBilinear(srcImage, W, H, channels, newW, newH);
            srcImage = centerCrop(resized, newW, newH, channels, 224, 224);
            W = H = 224;
        }
    } else {
        srcImage.assign(224 * 224 * channels, 0);
        W = H = 224;
    }

    // Preprocess to match torchvision GoogLeNet: scale to [0,1] then normalize with ImageNet mean/std
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stdv[3] = {0.229f, 0.224f, 0.225f};

    Tensor inputTensor(H, W, 3);
    std::vector<float> inputData(H * W * 3);
    for (size_t i = 0; i < inputData.size(); ++i) {
        float v = static_cast<float>(srcImage[i]) / 255.0f;
        int c = static_cast<int>(i % 3);
        inputData[i] = (v - mean[c]) / stdv[c];
    }

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
    googleNet.setRetainTensors(true); // keep intermediates for debug stats
    std::cout << "GoogleNet constructed" << std::endl;
    runStage("loadWeights", [&]{ googleNet.loadWeights(weights.get(), st.get()); });
    std::cout << "Weights loaded (or zero-initialized)" << std::endl;

    // Quick sanity: check that loaded weights are non-zero to catch parsing/mapping issues.
    auto tensorStats = [](const Tensor& t)
    {
        const float* p = t.hostData();
        if (!p) return std::tuple<float, float, float>{0.f, 0.f, 0.f};
        size_t n = t.numElements();
        float minv = p[0], maxv = p[0], sum = 0.f;
        for (size_t i = 0; i < n; ++i) {
            float v = p[i];
            minv = std::min(minv, v);
            maxv = std::max(maxv, v);
            sum += v;
        }
        return std::tuple<float, float, float>{minv, maxv, sum / float(n)};
    };

    auto [wmin, wmax, wmean] = tensorStats(googleNet["conv1.weight"]);
    auto [bmin, bmax, bmean] = tensorStats(googleNet["conv1.bias"]);
    auto [fcmin, fcmax, fcmean] = tensorStats(googleNet["fc.weight"]);
    auto [fbmin, fbmax, fbmean] = tensorStats(googleNet["fc.bias"]);
    std::cout << "conv1.weight stats (min/max/mean): " << wmin << " / " << wmax << " / " << wmean << std::endl;
    std::cout << "conv1.bias   stats (min/max/mean): " << bmin << " / " << bmax << " / " << bmean << std::endl;
    std::cout << "fc.weight    stats (min/max/mean): " << fcmin << " / " << fcmax << " / " << fcmean << std::endl;
    std::cout << "fc.bias      stats (min/max/mean): " << fbmin << " / " << fbmax << " / " << fbmean << std::endl;
    
    // We don't have weights, so we just run the graph to verify structure
    std::cout << "Graph constructed. Running inference..." << std::endl;
    
    // Optional label map (one label per line)
    auto labels = loadLabels(PROJECT_ROOT_DIR"/113-GoogleNet-Huicheol/imagenet_labels.txt", 1000);

    runStage("inference", [&]{
        auto results = googleNet(inputTensor);
        Tensor& result = results[0];

        auto debugCapture = [&](const char* label, Tensor& t)
        {
            if (!t.hasDeviceData()) {
                std::cout << "[dbg] " << label << ": no device data" << std::endl;
                return;
            }
            const size_t byteSize = t.numElements() * sizeof(float);
            vk::Buffer hostBuffer = netGlobalDevice.createBuffer({
                .size = byteSize,
                .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            });

            netGlobalDevice.newCommandBuffer(queue_compute)
                .begin()
                .copyBuffer(hostBuffer, t.buffer())
                .end()
                .submit()
                .wait();

            const float* dataPtr = reinterpret_cast<const float*>(hostBuffer.map());
            size_t N = t.numElements();
            float mn = dataPtr[0], mx = dataPtr[0], sum = 0.f, sum2 = 0.f;
            for (size_t i = 0; i < N; ++i) {
                float v = dataPtr[i];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
                sum += v;
                sum2 += v * v;
            }
            float mean = sum / N;
            float var = sum2 / N - mean * mean;
            float stddev = var > 0.f ? std::sqrt(var) : 0.f;
            std::cout << "[dbg] " << label << " stats (min/max/mean/std): "
                      << mn << " / " << mx << " / " << mean << " / " << stddev
                      << " shape=";
            for (auto s : t.shape()) std::cout << s << " ";
            std::cout << std::endl;
        };

        debugCapture("conv1.out", googleNet.debugTensor("conv1.out"));
        debugCapture("pool1.out", googleNet.debugTensor("pool1.out"));
        debugCapture("conv2.out", googleNet.debugTensor("conv2.out"));
        debugCapture("pool2.out", googleNet.debugTensor("pool2.out"));
        debugCapture("inception3a.out", googleNet.debugTensor("inception3a.out"));
        debugCapture("inception3b.out", googleNet.debugTensor("inception3b.out"));
        debugCapture("pool3.out", googleNet.debugTensor("pool3.out"));
        debugCapture("inception4a.out", googleNet.debugTensor("inception4a.out"));
        debugCapture("inception4b.out", googleNet.debugTensor("inception4b.out"));
        debugCapture("inception4c.out", googleNet.debugTensor("inception4c.out"));
        debugCapture("inception4d.out", googleNet.debugTensor("inception4d.out"));
        debugCapture("inception4e.out", googleNet.debugTensor("inception4e.out"));
        debugCapture("pool4.out", googleNet.debugTensor("pool4.out"));
        debugCapture("inception5a.out", googleNet.debugTensor("inception5a.out"));
        debugCapture("inception5b.out", googleNet.debugTensor("inception5b.out"));
        debugCapture("avgpool.out", googleNet.debugTensor("avgpool.out"));
        debugCapture("flatten.out", googleNet.debugTensor("flatten.out"));
        
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
        // Logit stats to see distribution
        {
            float mn = logits[0], mx = logits[0], sum = 0.f, sum2 = 0.f;
            for (float v : logits) { mn = std::min(mn, v); mx = std::max(mx, v); sum += v; sum2 += v * v; }
            float mean = sum / N;
            float var = sum2 / N - mean * mean;
            float stddev = var > 0.f ? std::sqrt(var) : 0.f;
            std::cout << "logits stats (min/max/mean/std): " << mn << " / " << mx << " / " << mean << " / " << stddev << std::endl;
        }
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

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

void verifyConv()
{
    std::cout << "Running verifyConv (Rigorous Spatial/Channel Test)..." << std::endl;
    // Test 1: Spatial Mapping
    // Input: 1 channel, 4x4.
    // Kernel: 2x2, S=1, P=0.
    // Output: 3x3.
    
    uint32_t C = 1;
    uint32_t F = 1;
    uint32_t K = 3;
    uint32_t H = 4;
    uint32_t W = 4;
    uint32_t S = 1;
    uint32_t P = 0;

    ConvolutionNode conv(C, F, K, S, P);
    conv["in0"] = Tensor(C, H, W);
    conv["weight"] = Tensor(C*K*K, F); 
    conv["bias"] = Tensor(F);

    // Fill input with distinct values: 0, 1, 2, ... 15
    std::vector<float> inData(C*H*W);
    for(int i=0; i<inData.size(); ++i) inData[i] = (float)i;
    conv["in0"].set(std::move(inData));

    // Weight: Select bottom-right pixel of kernel (ky=1, kx=1). Index k = 1*2 + 1 = 3.
    // Expected output at (0,0) should be input at (1,1) = 5.
    std::vector<float> wData(C*K*K*F, 0.0f);
    wData[3] = 1.0f; 
    conv["weight"].set(std::move(wData));

    std::vector<float> bData = {0.0f};
    conv["bias"].set(std::move(bData));

    conv.prepare();

    // Helper to allocate and upload (same as before)
    auto allocateAndUpload = [&](Tensor& t, bool upload) {
        size_t size = t.numElements() * sizeof(float);
        auto buf = netGlobalDevice.createBuffer({
            .size = size,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .reqMemProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        });
        t.bindBuffer(buf);

        if (upload && t.hasHostData()) {
            auto staging = netGlobalDevice.createBuffer({
                .size = size,
                .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            });
            memcpy(staging.map(), t.hostData(), size);
            staging.unmap();
            
            netGlobalDevice.newCommandBuffer(queue_transfer)
                .begin()
                .copyBuffer(buf, staging)
                .end()
                .submit()
                .wait();
        }
    };

    allocateAndUpload(conv["in0"], true);
    allocateAndUpload(conv["weight"], true);
    allocateAndUpload(conv["bias"], true);
    allocateAndUpload(conv["im2colOut"], false);
    allocateAndUpload(conv["out0"], false);

    auto cmd = netGlobalDevice.newCommandBuffer(queue_compute).begin();
    conv.run(cmd);
    cmd.end().submit().wait();

    // Check im2colOut directly
    Tensor& im2col = conv["im2colOut"];
    size_t im2colSize = im2col.numElements();
    vk::Buffer im2colHost = netGlobalDevice.createBuffer({
        .size = im2colSize * sizeof(float),
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });

    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(im2colHost, im2col.buffer())
        .end()
        .submit()
        .wait();

    const float* im2colData = reinterpret_cast<const float*>(im2colHost.map());
    
    // im2colOut shape: (H_*W_, C*K*K) = (9, 9)
    std::cout << "im2colOut column j=3 (first 9 rows): ";
    for(int i=0; i<9; ++i) std::cout << im2colData[i * 9 + 3] << " ";
    std::cout << std::endl;

    std::cout << "im2colOut column j=4 (first 9 rows): ";
    for(int i=0; i<9; ++i) std::cout << im2colData[i * 9 + 4] << " ";
    std::cout << std::endl;

    // Check Weights on GPU
    Tensor& wT = conv["weight"];
    size_t wSize = wT.numElements();
    vk::Buffer wHost = netGlobalDevice.createBuffer({
        .size = wSize * sizeof(float),
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });
    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(wHost, wT.buffer())
        .end()
        .submit()
        .wait();
    const float* wGpu = reinterpret_cast<const float*>(wHost.map());
    std::cout << "Weights on GPU: ";
    for(int i=0; i<wSize; ++i) std::cout << wGpu[i] << " ";
    std::cout << std::endl;

    Tensor& out = conv["out0"];
    size_t outSize = out.numElements();
    vk::Buffer hostBuffer = netGlobalDevice.createBuffer({
        .size = outSize * sizeof(float),
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });

    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(hostBuffer, out.buffer())
        .end()
        .submit()
        .wait();

    const float* mapped = reinterpret_cast<const float*>(hostBuffer.map());
    
    std::cout << "verifyConv output (first 9): ";
    for(size_t i=0; i<std::min(outSize, (size_t)9); ++i) {
        std::cout << mapped[i] << " ";
    }
    std::cout << std::endl;
    
    // Expected at (0,0): Input(1,1) = 5.
    // Expected at (0,1): Input(1,2) = 6.
    // Expected at (1,0): Input(2,1) = 9.
    float val00 = mapped[0];
    // With wData[3]=1.0 (ky=1, kx=0), we expect input at (1,0) which is 4.
    std::cout << "Expected(0,0)=4, Got=" << val00 << std::endl;
    
    if (std::abs(val00 - 4.0f) < 1e-3f) std::cout << "verifyConv PASSED" << std::endl;
    else std::cout << "verifyConv FAILED" << std::endl;
}

void verifyMaxPool()
{
    std::cout << "Running verifyMaxPool (2x2, S=2, CeilMode)..." << std::endl;
    // Input: 1 channel, 3x3.
    // Pool: 2x2, S=2.
    // Output: 2x2 (due to ceil_mode).
    // Input:
    // 1 2 3
    // 4 5 6
    // 7 8 9
    
    // Top-Left (0,0): Max(1,2,4,5) = 5.
    // Top-Right (0,1): Max(3, 6) = 6. (Padding implicit/ignore)
    // Bottom-Left (1,0): Max(7, 8) = 8.
    // Bottom-Right (1,1): Max(9) = 9.
    
    uint32_t C = 1;
    uint32_t H = 3;
    uint32_t W = 3;
    uint32_t P = 2; // Pool size
    uint32_t S = 2; // Stride
    uint32_t Pad = 0;

    MaxPoolingNode pool(P, S, Pad);
    pool["in0"] = Tensor(C, H, W);
    
    std::vector<float> inData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    pool["in0"].set(std::move(inData));
    
    pool.prepare();
    
    // Allocate and upload
    auto allocateAndUpload = [&](Tensor& t, bool upload) {
        size_t size = t.numElements() * sizeof(float);
        auto buf = netGlobalDevice.createBuffer({
            .size = size,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .reqMemProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        });
        t.bindBuffer(buf);

        if (upload && t.hasHostData()) {
            auto staging = netGlobalDevice.createBuffer({
                .size = size,
                .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            });
            memcpy(staging.map(), t.hostData(), size);
            staging.unmap();
            
            netGlobalDevice.newCommandBuffer(queue_transfer)
                .begin()
                .copyBuffer(buf, staging)
                .end()
                .submit()
                .wait();
        }
    };
    
    allocateAndUpload(pool["in0"], true);
    allocateAndUpload(pool["out0"], false);
    
    auto cmd = netGlobalDevice.newCommandBuffer(queue_compute).begin();
    pool.run(cmd);
    cmd.end().submit().wait();
    
    Tensor& out = pool["out0"];
    size_t outSize = out.numElements();
    vk::Buffer hostBuffer = netGlobalDevice.createBuffer({
        .size = outSize * sizeof(float),
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });

    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(hostBuffer, out.buffer())
        .end()
        .submit()
        .wait();

    const float* mapped = reinterpret_cast<const float*>(hostBuffer.map());
    std::cout << "verifyMaxPool output: ";
    for(size_t i=0; i<outSize; ++i) std::cout << mapped[i] << " ";
    std::cout << std::endl;
    
    if (outSize != 4) {
        std::cout << "verifyMaxPool FAILED: Wrong output size " << outSize << std::endl;
        return;
    }
    
    bool pass = true;
    if (mapped[0] != 5.0f) pass = false;
    if (mapped[1] != 6.0f) pass = false;
    if (mapped[2] != 8.0f) pass = false;
    if (mapped[3] != 9.0f) pass = false;
    
    if (pass) std::cout << "verifyMaxPool PASSED" << std::endl;
    else std::cout << "verifyMaxPool FAILED" << std::endl;
}

void verifyConvRigorous()
{
    std::cout << "Running verifyConvRigorous (K=7, S=2, P=3, C=3, F=1)..." << std::endl;
    
    uint32_t C = 3;
    uint32_t H = 224;
    uint32_t W = 224;
    uint32_t F = 64;
    uint32_t K = 7;
    uint32_t S = 2;
    uint32_t P = 3;
    
    ConvolutionNode conv(C, F, K, S, P);
    
    // Random input
    std::vector<float> inData(C * H * W);
    for(size_t i=0; i<inData.size(); ++i) inData[i] = (float(rand()) / RAND_MAX) * 2.0f - 1.0f;
    
    conv["in0"] = Tensor(C, H, W);
    conv["in0"].set(inData);
    
    // Random weights
    std::vector<float> wData(F * C * K * K);
    for(size_t i=0; i<wData.size(); ++i) wData[i] = (float(rand()) / RAND_MAX) * 0.1f;
    
    // Initialize weight tensor with correct shape (In*K*K, Out) for GEMM
    conv["weight"] = Tensor(C * K * K, F); 
    conv["weight"].set(wData);
    
    // Zero bias
    std::vector<float> bData(F, 0.0f);
    conv["bias"] = Tensor(F);
    conv["bias"].set(bData);
    
    conv.prepare();
    
    // Run GPU
    auto allocateAndUpload = [&](Tensor& t, bool upload) {
        size_t size = t.numElements() * sizeof(float);
        auto buf = netGlobalDevice.createBuffer({
            .size = size,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .reqMemProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        });
        t.bindBuffer(buf);

        if (upload && t.hasHostData()) {
            auto staging = netGlobalDevice.createBuffer({
                .size = size,
                .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            });
            memcpy(staging.map(), t.hostData(), size);
            staging.unmap();
            
            netGlobalDevice.newCommandBuffer(queue_transfer)
                .begin()
                .copyBuffer(buf, staging)
                .end()
                .submit()
                .wait();
        }
    };
    
    allocateAndUpload(conv["in0"], true);
    allocateAndUpload(conv["weight"], true);
    allocateAndUpload(conv["bias"], true);
    allocateAndUpload(conv["out0"], false);
    allocateAndUpload(conv["im2colOut"], false); // Allocate buffer for internal tensor
    
    auto cmd = netGlobalDevice.newCommandBuffer(queue_compute).begin();
    conv.run(cmd);
    cmd.end().submit().wait();
    
    // Download GPU result
    Tensor& out = conv["out0"];
    size_t outSize = out.numElements();
    vk::Buffer hostBuffer = netGlobalDevice.createBuffer({
        .size = outSize * sizeof(float),
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });

    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(hostBuffer, out.buffer())
        .end()
        .submit()
        .wait();

    const float* gpuOut = reinterpret_cast<const float*>(hostBuffer.map());
    
    // Run CPU Reference
    auto shape = out.shape();
    uint32_t outH = shape[1];
    uint32_t outW = shape[2];
    std::cout << "Output shape: " << outH << "x" << outW << std::endl;
    
    float maxDiff = 0.0f;
    for (uint32_t f = 0; f < F; ++f) {
        for (uint32_t oy = 0; oy < outH; ++oy) {
            for (uint32_t ox = 0; ox < outW; ++ox) {
                float acc = 0.0f;
                for (uint32_t c = 0; c < C; ++c) {
                    for (uint32_t ky = 0; ky < K; ++ky) {
                        for (uint32_t kx = 0; kx < K; ++kx) {
                            int iy = oy * S + ky - P;
                            int ix = ox * S + kx - P;
                            
                            float val = 0.0f;
                            if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                                val = inData[(c * H + iy) * W + ix];
                            }
                            
                            // Weight index: (f, c, ky, kx) -> but loaded as (c, ky, kx, f) in C++?
                            // Wait, loadConvWeight permutes to (In, K, K, Out).
                            // But here we set 'wData' directly.
                            // ConvolutionNode expects weights in (In, K, K, Out) format for GEMM?
                            // Let's check loadConvWeight again.
                            // It returns (In * K * K, Out).
                            // So we should prepare wData in (In, K, K, Out) format.
                            // Index: ((c * K + ky) * K + kx) * F + f
                            
                            float w = wData[((c * K + ky) * K + kx) * F + f];
                            acc += val * w;
                        }
                    }
                }
                
                int outIdx = (f * outH + oy) * outW + ox;
                float gpuVal = gpuOut[outIdx];
                float diff = std::abs(acc - gpuVal);
                maxDiff = std::max(maxDiff, diff);
                
                if (diff > 1e-3f && maxDiff == diff) {
                     std::cout << "Mismatch at (" << f << "," << oy << "," << ox << "): CPU=" << acc << " GPU=" << gpuVal << " Diff=" << diff << std::endl;
                }
            }
        }
    }
    
    std::cout << "verifyConvRigorous Max Diff: " << maxDiff << std::endl;
    if (maxDiff < 1e-3f) std::cout << "verifyConvRigorous PASSED" << std::endl;
    else std::cout << "verifyConvRigorous FAILED" << std::endl;
}

void verifyPermute()
{
    std::cout << "Running verifyPermute..." << std::endl;
    // Tensor (2, 2, 2)
    // 0 1
    // 2 3
    // ---
    // 4 5
    // 6 7
    
    // Permute (1, 2, 0) -> (2, 2, 2)
    // Old indices: (i, j, k)
    // New indices: (j, k, i)
    // Value at new(j, k, i) = old(i, j, k)
    //
    // New(0,0,0) <= Old(0,0,0) = 0
    // New(0,0,1) <= Old(1,0,0) = 4
    // New(0,1,0) <= Old(0,0,1) = 1
    // New(0,1,1) <= Old(1,0,1) = 5
    // ...
    
    Tensor t(2, 2, 2);
    std::vector<float> data(8);
    std::iota(data.begin(), data.end(), 0.0f);
    t.set(std::move(data));
    
    t.permute(1, 2, 0);
    
    const float* d = t.hostData();
    // Expected: 0, 4, 1, 5, 2, 6, 3, 7
    float expected[] = {0, 4, 1, 5, 2, 6, 3, 7};
    
    bool pass = true;
    std::cout << "verifyPermute output: ";
    for(int i=0; i<8; ++i) {
        std::cout << d[i] << " ";
        if (std::abs(d[i] - expected[i]) > 1e-3f) pass = false;
    }
    std::cout << std::endl;
    
    if (pass) std::cout << "verifyPermute PASSED" << std::endl;
    else std::cout << "verifyPermute FAILED" << std::endl;
}

void test(const char* imagePath = nullptr)
{
    verifyPermute();
    verifyConv();
    verifyMaxPool();
    verifyConvRigorous();
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
    const char* imgPath = imagePath ? imagePath : PROJECT_ROOT_DIR"/113-GoogleNet-Huicheol/data/dog.jpg";
    try
    {
        auto loaded = readImage<channels>(imgPath);
        srcImage = std::get<0>(loaded);
        W = std::get<1>(loaded);
        H = std::get<2>(loaded);
        std::cout << "Loaded " << imgPath << " (" << W << "x" << H << ")" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cout << imgPath << " not found, using zero image: " << e.what() << std::endl;
        srcImage.assign(H * W * channels, 0);
    }

    // Resize -> center crop to match torchvision eval: resize so min side=256, then 224x224 center crop
    // Match torchvision Resize (bilinear, align_corners=False) semantics; keep float to avoid extra quantization
    auto resizeBilinear = [](const std::vector<uint8_t>& src, uint32_t w, uint32_t h, uint32_t c, uint32_t newW, uint32_t newH)
    {
        std::vector<float> dst(newW * newH * c);
        for (uint32_t y = 0; y < newH; ++y) {
            float sy = (float(y) + 0.5f) * (float(h) / float(newH)) - 0.5f;
            int y0 = int(floor(sy));
            int y1 = y0 + 1;
            float wy = sy - float(y0);
            y0 = std::clamp(y0, 0, int(h) - 1);
            y1 = std::clamp(y1, 0, int(h) - 1);
            for (uint32_t x = 0; x < newW; ++x) {
                float sx = (float(x) + 0.5f) * (float(w) / float(newW)) - 0.5f;
                int x0 = int(floor(sx));
                int x1 = x0 + 1;
                float wx = sx - float(x0);
                x0 = std::clamp(x0, 0, int(w) - 1);
                x1 = std::clamp(x1, 0, int(w) - 1);
                for (uint32_t ch = 0; ch < c; ++ch) {
                    auto idx = [&](uint32_t X, uint32_t Y) { return (Y * w + X) * c + ch; };
                    float v00 = float(src[idx(x0, y0)]);
                    float v01 = float(src[idx(x1, y0)]);
                    float v10 = float(src[idx(x0, y1)]);
                    float v11 = float(src[idx(x1, y1)]);
                    float v0 = v00 + (v01 - v00) * wx;
                    float v1 = v10 + (v11 - v10) * wx;
                    float v = v0 + (v1 - v0) * wy;
                    dst[(y * newW + x) * c + ch] = v;
                }
            }
        }
        return dst;
    };

    auto centerCrop = [](const std::vector<float>& src, uint32_t w, uint32_t h, uint32_t c, uint32_t cw, uint32_t ch)
    {
        std::vector<float> dst(cw * ch * c);
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

    std::vector<float> imgFloat;
    if (!srcImage.empty()) {
        // If image is already 224x224, use it directly to avoid unnecessary scaling/cropping
        if (W == 224 && H == 224) {
            std::cout << "Image is already 224x224. Skipping resize/crop." << std::endl;
            imgFloat.resize(224 * 224 * channels);
            for(size_t i=0; i<imgFloat.size(); ++i) imgFloat[i] = float(srcImage[i]);
        }
        else {
            // Otherwise follow torchvision eval: resize shortest side to 256, then 224x224 center crop
            uint32_t minSide = std::min(W, H);
            float scale = 256.0f / float(minSide);
            uint32_t newW = uint32_t(std::round(W * scale));
            uint32_t newH = uint32_t(std::round(H * scale));
            auto resized = resizeBilinear(srcImage, W, H, channels, newW, newH); // float output
            imgFloat = centerCrop(resized, newW, newH, channels, 224, 224);
            W = H = 224;
        }
    } else {
        imgFloat.assign(224 * 224 * channels, 0.0f);
        W = H = 224;
    }

    // GoogleNet (IMAGENET1K_V1) uses transform_input=True, which maps [0,1] to [-1,1].
    const float mean[3] = {0.5f, 0.5f, 0.5f};
    const float stdv[3] = {0.5f, 0.5f, 0.5f};

    Tensor inputTensor(3, H, W); // NCHW
    std::vector<float> inputData(3 * H * W);
    for (uint32_t h = 0; h < H; ++h) {
        for (uint32_t w = 0; w < W; ++w) {
            for (uint32_t c = 0; c < channels; ++c) {
                size_t srcIdx = (h * W + w) * channels + c;
                size_t dstIdx = (c * H + h) * W + w;
                float v = imgFloat[srcIdx] / 255.0f;
                inputData[dstIdx] = (v - mean[c]) / stdv[c];
            }
        }
    }

    // Quick check that different images produce different inputs
    {
        float mn = inputData[0], mx = inputData[0], sum = 0.f;
        for (float v : inputData) { mn = std::min(mn, v); mx = std::max(mx, v); sum += v; }
        std::cout << "input stats (min/max/mean): " << mn << " / " << mx << " / " << (sum / inputData.size()) << std::endl;
        std::cout << "input first 6 vals: ";
        for (size_t i = 0; i < 6 && i < inputData.size(); ++i) std::cout << inputData[i] << " ";
        std::cout << std::endl;
    }

    inputTensor.set(std::move(inputData));

    // Run inference
    std::unique_ptr<SafeTensorsParser> st;
    try
    {
        st = std::make_unique<SafeTensorsParser>(PROJECT_CURRENT_DIR"/weights_fixed.safetensors");
        std::cout << "Loaded weights_fixed.safetensors" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cout << "weights_fixed.safetensors not loaded: " << e.what() << std::endl;
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
    runStage("loadWeights", [&]{ 
        if (st) {
            std::cerr << "SafeTensors Keys (stderr):" << std::endl;
            auto names = st->getTensorNames();
            for (const auto& name : names) {
                std::cerr << "  " << name << std::endl;
            }
        }
        googleNet.loadWeights(weights.get(), st.get()); 
    });
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
        debugCapture("conv2_reduce.out", googleNet.debugTensor("conv2_reduce.out"));
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

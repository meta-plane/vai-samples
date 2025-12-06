#include "neuralNodes.h"
#include "mobileNetNodes.h"
#include "jsonParser.h"
#include "timeChecker.hpp"
#include <stb/stb_image.h>
#include <cstring>
#include <cstdio>
#include <random>
#include <chrono>
#include <fstream>
#include <string>
#include "safeTensorsParser.h"

// ============================================================================
// Helper functions
// ============================================================================

const char* getImageNetClassName(int idx) {
    switch (idx) {
    case 281: return "tabby cat";
    case 282: return "tiger cat";
    case 283: return "Persian cat";
    case 284: return "Siamese cat";
    case 285: return "Egyptian cat";
    case 207: return "golden retriever";
    case 208: return "Labrador retriever";
    case 209: return "flat-coated retriever";
    case 230: return "collie";
    case 231: return "Border collie";
    case 232: return "Bouvier des Flandres";
    case 233: return "Rottweiler";
    case 234: return "German shepherd";
    case 235: return "Doberman";
    case 259: return "Samoyed";
    default: {
        static char buf[32];
        snprintf(buf, sizeof(buf), "class_%d", idx);
        return buf;
    }
    }
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


Tensor makeRandomTensor(const std::vector<uint32_t>& shape, std::mt19937& gen, float stddev = 0.02f)
{
    std::normal_distribution<float> dist(0.0f, stddev);
    Tensor t(shape);
    size_t n = t.numElements();
    std::vector<float> data(n);
    for (auto& v : data) v = dist(gen);
    t.set(std::move(data));
    return t;
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

// ============================================================================
// Individual Node Tests
// ============================================================================

void testRelu6()
{
    printf("\n=== Testing ReLU6 Node ===\n");

    Relu6Node relu6;

    std::vector<float> inputData = { -2.0f, -1.0f, 0.0f, 1.0f, 3.0f, 5.0f, 6.0f, 7.0f, 10.0f };
    Tensor input(3, 3, 1);
    input.set(inputData);

    NeuralNet net(gDevice, 1, 1);
    net.input(0) - relu6 - net.output(0);

    auto results = net(std::move(input));

    Buffer outBuffer = gDevice.createBuffer({
        9 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        });

    gDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuffer, results[0].buffer())
        .end()
        .submit()
        .wait();

    float* data = (float*)outBuffer.map();
    printf("Input:    ");
    for (int i = 0; i < 9; ++i) printf("%5.1f ", inputData[i]);
    printf("\nOutput:   ");
    for (int i = 0; i < 9; ++i) printf("%5.1f ", data[i]);
    printf("\nExpected: ");
    float expected[] = { 0.0f, 0.0f, 0.0f, 1.0f, 3.0f, 5.0f, 6.0f, 6.0f, 6.0f };
    for (int i = 0; i < 9; ++i) printf("%5.1f ", expected[i]);
    printf("\n");

    bool pass = true;
    for (int i = 0; i < 9; ++i) {
        if (std::abs(data[i] - expected[i]) > 0.001f) pass = false;
    }
    printf("Result: %s\n", pass ? "PASS" : "FAIL");
}


void testDepthwiseConv()
{
    printf("\n=== Testing Depthwise Conv Node ===\n");

    const uint32_t H = 4, W = 4, C = 2;
    const uint32_t K = 3;

    DepthwiseConvNode dwconv(C, K, 1, 1);  // stride=1, pad=1

    std::vector<float> inputData(H * W * C, 1.0f);
    Tensor input(H, W, C);
    input.set(inputData);

    std::vector<float> weightData(K * K * C, 1.0f);
    Tensor weight(K * K, C);
    weight.set(weightData);

    dwconv["weight"] = weight;

    NeuralNet net(gDevice, 1, 1);
    net.input(0) - dwconv - net.output(0);

    auto results = net(std::move(input));

    const auto& outShape = results[0].shape();
    printf("Input shape: %d x %d x %d\n", H, W, C);
    printf("Output shape: %d x %d x %d\n", outShape[0], outShape[1], outShape[2]);

    size_t outSize = outShape[0] * outShape[1] * outShape[2];
    Buffer outBuffer = gDevice.createBuffer({
        outSize * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        });

    gDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuffer, results[0].buffer())
        .end()
        .submit()
        .wait();

    float* data = (float*)outBuffer.map();

    // Center pixel should be 9 (3x3 kernel, all 1s, input all 1s)
    int centerIdx = (1 * outShape[1] + 1) * outShape[2] + 0;
    printf("Center pixel (should be 9.0): %.1f\n", data[centerIdx]);
    printf("Result: %s\n", std::abs(data[centerIdx] - 9.0f) < 0.001f ? "PASS" : "FAIL");
}


void testPointwiseConv()
{
    printf("\n=== Testing Pointwise Conv Node ===\n");

    const uint32_t H = 2, W = 2, C_in = 3, C_out = 4;

    PointwiseConvNode pwconv(C_in, C_out);

    std::vector<float> inputData(H * W * C_in, 1.0f);
    Tensor input(H, W, C_in);
    input.set(inputData);

    std::vector<float> weightData(C_in * C_out, 1.0f);
    Tensor weight(C_in, C_out);
    weight.set(weightData);

    pwconv["weight"] = weight;

    NeuralNet net(gDevice, 1, 1);
    net.input(0) - pwconv - net.output(0);

    auto results = net(std::move(input));

    const auto& outShape = results[0].shape();
    printf("Input shape: %d x %d x %d\n", H, W, C_in);
    printf("Output shape: %d x %d x %d\n", outShape[0], outShape[1], outShape[2]);

    size_t outSize = outShape[0] * outShape[1] * outShape[2];
    Buffer outBuffer = gDevice.createBuffer({
        outSize * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        });

    gDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuffer, results[0].buffer())
        .end()
        .submit()
        .wait();

    float* data = (float*)outBuffer.map();
    printf("Output (all should be %.1f): ", (float)C_in);
    for (int i = 0; i < 4; ++i) printf("%.1f ", data[i]);

    bool pass = true;
    for (size_t i = 0; i < outSize; ++i) {
        if (std::abs(data[i] - (float)C_in) > 0.001f) pass = false;
    }
    printf("\nResult: %s\n", pass ? "PASS" : "FAIL");
}


void testGlobalAvgPool()
{
    printf("\n=== Testing Global Average Pool Node ===\n");

    const uint32_t H = 2, W = 2, C = 3;

    GlobalAvgPoolNode gap;

    std::vector<float> inputData(H * W * C);
    for (uint32_t h = 0; h < H; ++h) {
        for (uint32_t w = 0; w < W; ++w) {
            for (uint32_t c = 0; c < C; ++c) {
                inputData[(h * W + w) * C + c] = (float)(c + 1);
            }
        }
    }

    Tensor input(H, W, C);
    input.set(inputData);

    NeuralNet net(gDevice, 1, 1);
    net.input(0) - gap - net.output(0);

    auto results = net(std::move(input));

    const auto& outShape = results[0].shape();
    printf("Input shape: %d x %d x %d\n", H, W, C);
    printf("Output shape: %d\n", outShape[0]);

    Buffer outBuffer = gDevice.createBuffer({
        C * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        });

    gDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuffer, results[0].buffer())
        .end()
        .submit()
        .wait();

    float* data = (float*)outBuffer.map();
    printf("Output (expected: 1.0 2.0 3.0): ");
    for (uint32_t i = 0; i < C; ++i) printf("%.1f ", data[i]);

    bool pass = true;
    for (uint32_t i = 0; i < C; ++i) {
        if (std::abs(data[i] - (float)(i + 1)) > 0.001f) pass = false;
    }
    printf("\nResult: %s\n", pass ? "PASS" : "FAIL");
}


void testConvBnRelu6()
{
    printf("\n=== Testing Conv+BN+ReLU6 Combined Node ===\n");

    const uint32_t H = 8, W = 8, C_in = 3, C_out = 16;
    const uint32_t K = 3, stride = 2, pad = 1;

    ConvBnRelu6Node convBnRelu6(C_in, C_out, K, stride, pad);

    std::mt19937 gen(42);

    std::vector<float> inputData(H * W * C_in, 0.5f);
    Tensor input(H, W, C_in);
    input.set(inputData);

    convBnRelu6["conv_weight"] = makeRandomTensor({ C_out * C_in * K * K }, gen, 0.1f);
    convBnRelu6["conv_bias"] = makeConstTensor(C_out, 0.0f);
    convBnRelu6["bn_gamma"] = makeConstTensor(C_out, 1.0f);
    convBnRelu6["bn_beta"] = makeConstTensor(C_out, 0.0f);
    convBnRelu6["bn_mean"] = makeConstTensor(C_out, 0.0f);
    convBnRelu6["bn_var"] = makeConstTensor(C_out, 1.0f);

    NeuralNet net(gDevice, 1, 1);
    net.input(0) - convBnRelu6 - net.output(0);

    auto results = net(std::move(input));

    const auto& outShape = results[0].shape();
    uint32_t expectedH = (H + 2 * pad - K) / stride + 1;
    uint32_t expectedW = (W + 2 * pad - K) / stride + 1;

    printf("Input shape: %d x %d x %d\n", H, W, C_in);
    printf("Output shape: %d x %d x %d\n", outShape[0], outShape[1], outShape[2]);
    printf("Expected shape: %d x %d x %d\n", expectedH, expectedW, C_out);

    size_t outSize = outShape[0] * outShape[1] * outShape[2];
    Buffer outBuffer = gDevice.createBuffer({
        outSize * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        });

    gDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuffer, results[0].buffer())
        .end()
        .submit()
        .wait();

    float* data = (float*)outBuffer.map();

    bool allInRange = true;
    float minVal = data[0], maxVal = data[0];
    for (size_t i = 0; i < outSize; ++i) {
        if (data[i] < 0.0f || data[i] > 6.0f) allInRange = false;
        minVal = std::min(minVal, data[i]);
        maxVal = std::max(maxVal, data[i]);
    }
    printf("Output range: [%.3f, %.3f]\n", minVal, maxVal);
    printf("All values in [0, 6]: %s\n", allInRange ? "YES" : "NO");

    bool shapePass = (outShape[0] == expectedH && outShape[1] == expectedW && outShape[2] == C_out);
    printf("Result: %s\n", (shapePass && allInRange) ? "PASS" : "FAIL");
}


void testInvertedResidualBlock()
{
    printf("\n=== Testing Inverted Residual Block ===\n");

    const uint32_t H = 14, W = 14, C_in = 32, C_out = 32;
    const uint32_t expandRatio = 6;
    const uint32_t stride = 1;

    InvertedResidualBlock irb(C_in, C_out, expandRatio, stride);

    uint32_t hiddenDim = C_in * expandRatio;

    std::mt19937 gen(42);

    std::vector<float> inputData(H * W * C_in, 0.1f);
    Tensor input(H, W, C_in);
    input.set(inputData);

    // Expansion layer weights
    irb["expand_pw_weight"] = makeRandomTensor({ C_in, hiddenDim }, gen);
    irb["expand_bn_gamma"] = makeConstTensor(hiddenDim, 1.0f);
    irb["expand_bn_beta"] = makeConstTensor(hiddenDim, 0.0f);
    irb["expand_bn_mean"] = makeConstTensor(hiddenDim, 0.0f);
    irb["expand_bn_var"] = makeConstTensor(hiddenDim, 1.0f);

    // Depthwise layer weights
    irb["dw_weight"] = makeRandomTensor({ 9, hiddenDim }, gen);
    irb["dw_bn_gamma"] = makeConstTensor(hiddenDim, 1.0f);
    irb["dw_bn_beta"] = makeConstTensor(hiddenDim, 0.0f);
    irb["dw_bn_mean"] = makeConstTensor(hiddenDim, 0.0f);
    irb["dw_bn_var"] = makeConstTensor(hiddenDim, 1.0f);

    // Projection layer weights
    irb["proj_pw_weight"] = makeRandomTensor({ hiddenDim, C_out }, gen);
    irb["proj_bn_gamma"] = makeConstTensor(C_out, 1.0f);
    irb["proj_bn_beta"] = makeConstTensor(C_out, 0.0f);
    irb["proj_bn_mean"] = makeConstTensor(C_out, 0.0f);
    irb["proj_bn_var"] = makeConstTensor(C_out, 1.0f);

    NeuralNet net(gDevice, 1, 1);
    net.input(0) - irb - net.output(0);

    printf("Running forward pass...\n");
    auto start = std::chrono::high_resolution_clock::now();

    auto results = net(std::move(input));

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    const auto& outShape = results[0].shape();
    printf("Input shape: %d x %d x %d\n", H, W, C_in);
    printf("Hidden dim: %d\n", hiddenDim);
    printf("Output shape: %d x %d x %d\n", outShape[0], outShape[1], outShape[2]);
    printf("Time: %lld us\n", (long long)duration.count());
    printf("Residual connection used: %s\n",
        (stride == 1 && C_in == C_out) ? "YES" : "NO");

    size_t outSize = outShape[0] * outShape[1] * outShape[2];
    Buffer outBuffer = gDevice.createBuffer({
        outSize * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        });

    gDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuffer, results[0].buffer())
        .end()
        .submit()
        .wait();

    float* data = (float*)outBuffer.map();
    float minVal = data[0], maxVal = data[0], sum = 0;
    for (size_t i = 0; i < outSize; ++i) {
        minVal = std::min(minVal, data[i]);
        maxVal = std::max(maxVal, data[i]);
        sum += data[i];
    }
    printf("Output stats: min=%.4f, max=%.4f, mean=%.4f\n",
        minVal, maxVal, sum / outSize);

    bool shapePass = (outShape[0] == H && outShape[1] == W && outShape[2] == C_out);
    printf("Result: %s\n", shapePass ? "PASS" : "FAIL");
}


void testSimplePipeline()
{
    printf("\n=== Testing Simple Pipeline (Conv -> IRB -> GAP -> FC) ===\n");

    const uint32_t inputH = 56, inputW = 56, inputC = 3;
    const uint32_t conv1OutC = 32;
    const uint32_t irb1OutC = 16;
    const uint32_t numClasses = 10;

    // Create nodes
    ConvBnRelu6Node conv1(inputC, conv1OutC, 3, 2, 1);
    InvertedResidualBlock irb1(conv1OutC, irb1OutC, 1, 1);  // t=1
    GlobalAvgPoolNode gap;
    FCMobileNetNode fc(irb1OutC, numClasses);

    std::mt19937 gen(42);

    // Conv1 weights
    conv1["conv_weight"] = makeRandomTensor({ conv1OutC * inputC * 3 * 3 }, gen);
    conv1["conv_bias"] = makeConstTensor(conv1OutC, 0.0f);
    conv1["bn_gamma"] = makeConstTensor(conv1OutC, 1.0f);
    conv1["bn_beta"] = makeConstTensor(conv1OutC, 0.0f);
    conv1["bn_mean"] = makeConstTensor(conv1OutC, 0.0f);
    conv1["bn_var"] = makeConstTensor(conv1OutC, 1.0f);

    // IRB1 weights (t=1, no expansion)
    uint32_t hiddenDim = conv1OutC;  // t=1
    irb1["dw_weight"] = makeRandomTensor({ 9, hiddenDim }, gen);
    irb1["dw_bn_gamma"] = makeConstTensor(hiddenDim, 1.0f);
    irb1["dw_bn_beta"] = makeConstTensor(hiddenDim, 0.0f);
    irb1["dw_bn_mean"] = makeConstTensor(hiddenDim, 0.0f);
    irb1["dw_bn_var"] = makeConstTensor(hiddenDim, 1.0f);
    irb1["proj_pw_weight"] = makeRandomTensor({ hiddenDim, irb1OutC }, gen);
    irb1["proj_bn_gamma"] = makeConstTensor(irb1OutC, 1.0f);
    irb1["proj_bn_beta"] = makeConstTensor(irb1OutC, 0.0f);
    irb1["proj_bn_mean"] = makeConstTensor(irb1OutC, 0.0f);
    irb1["proj_bn_var"] = makeConstTensor(irb1OutC, 1.0f);

    // FC weights
    fc["weight"] = makeRandomTensor({ irb1OutC, numClasses }, gen);
    fc["bias"] = makeConstTensor(numClasses, 0.0f);

    // Input
    std::normal_distribution<float> dist(0.5f, 0.1f);
    std::vector<float> inputData(inputH * inputW * inputC);
    for (auto& v : inputData) v = dist(gen);
    Tensor input(inputH, inputW, inputC);
    input.set(inputData);

    // Build network
    NeuralNet net(gDevice, 1, 1);
    net.input(0) - conv1 - irb1 - gap - fc - net.output(0);

    printf("Network structure:\n");
    printf("  Input: %d x %d x %d\n", inputH, inputW, inputC);
    printf("  Conv1: -> 28 x 28 x %d\n", conv1OutC);
    printf("  IRB1:  -> 28 x 28 x %d\n", irb1OutC);
    printf("  GAP:   -> %d\n", irb1OutC);
    printf("  FC:    -> %d\n", numClasses);

    printf("\nRunning forward pass...\n");
    auto start = std::chrono::high_resolution_clock::now();

    auto results = net(std::move(input));

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    printf("Forward pass completed in %lld us\n", (long long)duration.count());

    // Read output
    Buffer outBuffer = gDevice.createBuffer({
        numClasses * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        });

    gDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuffer, results[0].buffer())
        .end()
        .submit()
        .wait();

    float* logits = (float*)outBuffer.map();

    printf("\nOutput logits:\n");
    for (uint32_t i = 0; i < numClasses; ++i) {
        printf("  Class %d: %8.4f\n", i, logits[i]);
    }

    int maxIdx = 0;
    float maxVal = logits[0];
    for (uint32_t i = 1; i < numClasses; ++i) {
        if (logits[i] > maxVal) {
            maxVal = logits[i];
            maxIdx = i;
        }
    }
    printf("\nPredicted class: %d (logit: %.4f)\n", maxIdx, maxVal);
    printf("Result: PASS (pipeline executed successfully)\n");
}

void testLoadWeights()
{
    printf("\n=== Testing SafeTensors Weight Loading ===\n");
    fflush(stdout);

    const char* weightsPath = PROJECT_CURRENT_DIR "/mobilenetv2_weights.safetensors";
    printf("Looking for: %s\n", weightsPath);
    fflush(stdout);

    try {
        SafeTensorsParser weights(weightsPath);

        auto tensorNames = weights.getTensorNames();
        printf("Loaded %zu tensors\n", tensorNames.size());

        // 첫 번째 Conv 가중치 확인
        auto shape = weights["features.0.0.weight"].getShape();
        printf("features.0.0.weight: [%u, %u, %u, %u]\n",
            shape[0], shape[1], shape[2], shape[3]);

        printf("Result: PASS\n");
    }
    catch (const std::exception& e) {
        printf("Error: %s\n", e.what());
    }
}

void testImageClassification()
{
    printf("\n=== Testing Image Classification ===\n");

    const char* weightsPath = PROJECT_CURRENT_DIR "/mobilenetv2_weights.safetensors";
    const char* imagePath = PROJECT_CURRENT_DIR "/images/cat.jpg";  // 하나만 테스트

    // 1. 이미지 로드
    int w, h, c;
    uint8_t* data = stbi_load(imagePath, &w, &h, &c, 3);
    if (!data) {
        printf("Failed to load image: %s\n", imagePath);
        printf("Please put an image in the images folder!\n");
        return;
    }
    printf("Loaded image: %s (%d x %d)\n", imagePath, w, h);

    // 2. 224x224로 리사이즈 + 정규화
    const int size = 224;
    std::vector<float> input(size * size * 3);

    int cropSize = std::min(w, h);
    int offsetX = (w - cropSize) / 2;
    int offsetY = (h - cropSize) / 2;

    const float mean[3] = { 0.485f, 0.456f, 0.406f };
    const float std_[3] = { 0.229f, 0.224f, 0.225f };

    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            int srcX = offsetX + (x * cropSize) / size;
            int srcY = offsetY + (y * cropSize) / size;
            srcX = std::max(0, std::min(w - 1, srcX));
            srcY = std::max(0, std::min(h - 1, srcY));

            int srcIdx = (srcY * w + srcX) * 3;
            int dstIdx = (y * size + x) * 3;

            for (int ch = 0; ch < 3; ++ch) {
                float pixel = data[srcIdx + ch] / 255.0f;
                input[dstIdx + ch] = (pixel - mean[ch]) / std_[ch];
            }
        }
    }
    stbi_image_free(data);
    printf("Preprocessed to 224x224\n");

    // 3. 가중치 로드
    SafeTensorsParser weights(weightsPath);
    printf("Loaded weights\n");

    // 4. 첫 번째 Conv + BN + ReLU6 레이어만 테스트
    ConvBnRelu6Node conv1(3, 32, 3, 2, 1);

    auto conv1_w = weights["features.0.0.weight"].parseNDArray();
    Tensor conv1_weight(32 * 3 * 3 * 3);
    conv1_weight.set(conv1_w);

    Tensor conv1_bias(32);
    conv1_bias.set(std::vector<float>(32, 0.0f));

    auto bn1_gamma = weights["features.0.1.weight"].parseNDArray();
    auto bn1_beta = weights["features.0.1.bias"].parseNDArray();
    auto bn1_mean = weights["features.0.1.running_mean"].parseNDArray();
    auto bn1_var = weights["features.0.1.running_var"].parseNDArray();

    Tensor conv1_bn_gamma(32); conv1_bn_gamma.set(bn1_gamma);
    Tensor conv1_bn_beta(32); conv1_bn_beta.set(bn1_beta);
    Tensor conv1_bn_mean(32); conv1_bn_mean.set(bn1_mean);
    Tensor conv1_bn_var(32); conv1_bn_var.set(bn1_var);

    conv1["conv_weight"] = conv1_weight;
    conv1["conv_bias"] = conv1_bias;
    conv1["bn_gamma"] = conv1_bn_gamma;
    conv1["bn_beta"] = conv1_bn_beta;
    conv1["bn_mean"] = conv1_bn_mean;
    conv1["bn_var"] = conv1_bn_var;

    // 5. 입력 텐서 생성
    Tensor inputTensor(224, 224, 3);
    inputTensor.set(input);

    // 6. 네트워크 실행
    GlobalAvgPoolNode gap;

    NeuralNet net(gDevice, 1, 1);
    net.input(0) - conv1 - gap - net.output(0);

    printf("Running first layer...\n");
    auto results = net(std::move(inputTensor));

    // 7. 결과 출력
    const auto& outShape = results[0].shape();
    size_t outSize = 1;
    for (auto d : outShape) outSize *= d;

    Buffer outBuffer = gDevice.createBuffer({
        outSize * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        });

    gDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuffer, results[0].buffer())
        .end()
        .submit()
        .wait();

    float* output = (float*)outBuffer.map();

    printf("Output (32 channels after first Conv+BN+ReLU6 + GAP):\n");
    for (size_t i = 0; i < std::min((size_t)10, outSize); ++i) {
        printf("  Channel %zu: %.4f\n", i, output[i]);
    }

    printf("\nResult: PASS - Image processed with real weights!\n");
}

// ============================================================================
// MobileNetV2 Full Implementation
// 전체 17개 Inverted Residual Block 포함
// test_mobilenet.cpp의 testFullClassification() 함수를 이 코드로 교체하세요
// ============================================================================

// 헬퍼: IRB 가중치 로드 (expansion 있는 경우)
void loadIRBWeightsWithExpansion(
    InvertedResidualBlock& irb,
    SafeTensorsParser& weights,
    int featureIdx,
    int inC, int outC, int expandRatio)
{
    int hiddenDim = inC * expandRatio;
    char keyBuf[128];

    // Expansion PW Conv
    snprintf(keyBuf, sizeof(keyBuf), "features.%d.conv.0.0.weight", featureIdx);
    auto exp_w = weights[keyBuf].parseNDArray();
    std::vector<float> exp_w_t(inC * hiddenDim);
    for (int o = 0; o < hiddenDim; ++o)
        for (int i = 0; i < inC; ++i)
            exp_w_t[i * hiddenDim + o] = exp_w[o * inC + i];
    Tensor t_exp_w(inC, hiddenDim); t_exp_w.set(exp_w_t);
    irb["expand_pw_weight"] = t_exp_w;

    // Expansion BN
    snprintf(keyBuf, sizeof(keyBuf), "features.%d.conv.0.1.weight", featureIdx);
    auto exp_bn_g = weights[keyBuf].parseNDArray();
    snprintf(keyBuf, sizeof(keyBuf), "features.%d.conv.0.1.bias", featureIdx);
    auto exp_bn_b = weights[keyBuf].parseNDArray();
    snprintf(keyBuf, sizeof(keyBuf), "features.%d.conv.0.1.running_mean", featureIdx);
    auto exp_bn_m = weights[keyBuf].parseNDArray();
    snprintf(keyBuf, sizeof(keyBuf), "features.%d.conv.0.1.running_var", featureIdx);
    auto exp_bn_v = weights[keyBuf].parseNDArray();
    Tensor teg(hiddenDim); teg.set(exp_bn_g); irb["expand_bn_gamma"] = teg;
    Tensor teb(hiddenDim); teb.set(exp_bn_b); irb["expand_bn_beta"] = teb;
    Tensor tem(hiddenDim); tem.set(exp_bn_m); irb["expand_bn_mean"] = tem;
    Tensor tev(hiddenDim); tev.set(exp_bn_v); irb["expand_bn_var"] = tev;

    // DW Conv
    snprintf(keyBuf, sizeof(keyBuf), "features.%d.conv.1.0.weight", featureIdx);
    auto dw_w = weights[keyBuf].parseNDArray();
    // K=3, KK=9
    std::vector<float> dw_w_t(hiddenDim * 9); // size = C * KK
    // Rearrange to (k * C + c) order to match shader weight layout [K][K][C]
    for (int k = 0; k < 9; ++k) {
        for (int c = 0; c < hiddenDim; ++c) {
            dw_w_t[k * hiddenDim + c] = dw_w[c * 9 + k];
        }
    }
    Tensor t_dw_w(hiddenDim * 9);
    t_dw_w.set(dw_w_t);
    irb["dw_weight"] = t_dw_w;

    // DW BN
    snprintf(keyBuf, sizeof(keyBuf), "features.%d.conv.1.1.weight", featureIdx);
    auto dw_bn_g = weights[keyBuf].parseNDArray();
    snprintf(keyBuf, sizeof(keyBuf), "features.%d.conv.1.1.bias", featureIdx);
    auto dw_bn_b = weights[keyBuf].parseNDArray();
    snprintf(keyBuf, sizeof(keyBuf), "features.%d.conv.1.1.running_mean", featureIdx);
    auto dw_bn_m = weights[keyBuf].parseNDArray();
    snprintf(keyBuf, sizeof(keyBuf), "features.%d.conv.1.1.running_var", featureIdx);
    auto dw_bn_v = weights[keyBuf].parseNDArray();
    Tensor tdg(hiddenDim); tdg.set(dw_bn_g); irb["dw_bn_gamma"] = tdg;
    Tensor tdb(hiddenDim); tdb.set(dw_bn_b); irb["dw_bn_beta"] = tdb;
    Tensor tdm(hiddenDim); tdm.set(dw_bn_m); irb["dw_bn_mean"] = tdm;
    Tensor tdv(hiddenDim); tdv.set(dw_bn_v); irb["dw_bn_var"] = tdv;

    // Proj PW Conv
    snprintf(keyBuf, sizeof(keyBuf), "features.%d.conv.2.weight", featureIdx);
    auto proj_w = weights[keyBuf].parseNDArray();
    std::vector<float> proj_w_t(hiddenDim * outC);
    for (int o = 0; o < outC; ++o)
        for (int i = 0; i < hiddenDim; ++i)
            proj_w_t[i * outC + o] = proj_w[o * hiddenDim + i];
    Tensor t_proj_w(hiddenDim, outC); t_proj_w.set(proj_w_t);
    irb["proj_pw_weight"] = t_proj_w;

    // Proj BN
    snprintf(keyBuf, sizeof(keyBuf), "features.%d.conv.3.weight", featureIdx);
    auto proj_bn_g = weights[keyBuf].parseNDArray();
    snprintf(keyBuf, sizeof(keyBuf), "features.%d.conv.3.bias", featureIdx);
    auto proj_bn_b = weights[keyBuf].parseNDArray();
    snprintf(keyBuf, sizeof(keyBuf), "features.%d.conv.3.running_mean", featureIdx);
    auto proj_bn_m = weights[keyBuf].parseNDArray();
    snprintf(keyBuf, sizeof(keyBuf), "features.%d.conv.3.running_var", featureIdx);
    auto proj_bn_v = weights[keyBuf].parseNDArray();
    Tensor tpg(outC); tpg.set(proj_bn_g); irb["proj_bn_gamma"] = tpg;
    Tensor tpb(outC); tpb.set(proj_bn_b); irb["proj_bn_beta"] = tpb;
    Tensor tpm(outC); tpm.set(proj_bn_m); irb["proj_bn_mean"] = tpm;
    Tensor tpv(outC); tpv.set(proj_bn_v); irb["proj_bn_var"] = tpv;
}

// 헬퍼: IRB 가중치 로드 (expansion 없는 경우, t=1)
void loadIRBWeightsNoExpansion(
    InvertedResidualBlock& irb,
    SafeTensorsParser& weights,
    int featureIdx,
    int inC, int outC)
{
    char keyBuf[128];

    // DW Conv (index 0)
    snprintf(keyBuf, sizeof(keyBuf), "features.%d.conv.0.0.weight", featureIdx);
    auto dw_w = weights[keyBuf].parseNDArray();
    std::vector<float> dw_w_t(9 * inC);
    for (int c = 0; c < inC; ++c)
        for (int k = 0; k < 9; ++k)
            dw_w_t[k * inC + c] = dw_w[c * 9 + k];
    Tensor t_dw_w(9, inC); t_dw_w.set(dw_w_t);
    irb["dw_weight"] = t_dw_w;

    // DW BN
    snprintf(keyBuf, sizeof(keyBuf), "features.%d.conv.0.1.weight", featureIdx);
    auto dw_bn_g = weights[keyBuf].parseNDArray();
    snprintf(keyBuf, sizeof(keyBuf), "features.%d.conv.0.1.bias", featureIdx);
    auto dw_bn_b = weights[keyBuf].parseNDArray();
    snprintf(keyBuf, sizeof(keyBuf), "features.%d.conv.0.1.running_mean", featureIdx);
    auto dw_bn_m = weights[keyBuf].parseNDArray();
    snprintf(keyBuf, sizeof(keyBuf), "features.%d.conv.0.1.running_var", featureIdx);
    auto dw_bn_v = weights[keyBuf].parseNDArray();
    Tensor tdg(inC); tdg.set(dw_bn_g); irb["dw_bn_gamma"] = tdg;
    Tensor tdb(inC); tdb.set(dw_bn_b); irb["dw_bn_beta"] = tdb;
    Tensor tdm(inC); tdm.set(dw_bn_m); irb["dw_bn_mean"] = tdm;
    Tensor tdv(inC); tdv.set(dw_bn_v); irb["dw_bn_var"] = tdv;

    // Proj PW Conv (index 1)
    snprintf(keyBuf, sizeof(keyBuf), "features.%d.conv.1.weight", featureIdx);
    auto proj_w = weights[keyBuf].parseNDArray();
    std::vector<float> proj_w_t(inC * outC);
    for (int o = 0; o < outC; ++o)
        for (int i = 0; i < inC; ++i)
            proj_w_t[i * outC + o] = proj_w[o * inC + i];
    Tensor t_proj_w(inC, outC); t_proj_w.set(proj_w_t);
    irb["proj_pw_weight"] = t_proj_w;

    // Proj BN (index 2)
    snprintf(keyBuf, sizeof(keyBuf), "features.%d.conv.2.weight", featureIdx);
    auto proj_bn_g = weights[keyBuf].parseNDArray();
    snprintf(keyBuf, sizeof(keyBuf), "features.%d.conv.2.bias", featureIdx);
    auto proj_bn_b = weights[keyBuf].parseNDArray();
    snprintf(keyBuf, sizeof(keyBuf), "features.%d.conv.2.running_mean", featureIdx);
    auto proj_bn_m = weights[keyBuf].parseNDArray();
    snprintf(keyBuf, sizeof(keyBuf), "features.%d.conv.2.running_var", featureIdx);
    auto proj_bn_v = weights[keyBuf].parseNDArray();
    Tensor tpg(outC); tpg.set(proj_bn_g); irb["proj_bn_gamma"] = tpg;
    Tensor tpb(outC); tpb.set(proj_bn_b); irb["proj_bn_beta"] = tpb;
    Tensor tpm(outC); tpm.set(proj_bn_m); irb["proj_bn_mean"] = tpm;
    Tensor tpv(outC); tpv.set(proj_bn_v); irb["proj_bn_var"] = tpv;
}

void printTensorStats(const std::vector<float>& t, const std::string& name)
{
    float mn = 1e9, mx = -1e9, sum = 0.0f;
    for (float v : t) {
        mn = std::min(mn, v);
        mx = std::max(mx, v);
        sum += v;
    }
    float mean = sum / t.size();
    printf("=== %s Stats ===\n", name.c_str());
    printf("min=%.6f, max=%.6f, mean=%.6f, size=%zu\n\n", mn, mx, mean, t.size());
}
void debugIRB1Output(SafeTensorsParser& weights,
    const std::vector<float>& input224)
{
    printf("\n[Debug] Running Conv0 + IRB1 only...\n");

    // 1) Conv0 노드 새로 생성 + 가중치 로딩 (full 모델이랑 동일하게)
    ConvBnRelu6Node dbgConv0(3, 32, 3, 2, 1);

    {
        auto w = weights["features.0.0.weight"].parseNDArray();
        Tensor tw(32 * 3 * 3 * 3); tw.set(w);
        dbgConv0["conv_weight"] = tw;
        dbgConv0["conv_bias"] = makeConstTensor(32, 0.0f);

        auto g = weights["features.0.1.weight"].parseNDArray();
        auto b = weights["features.0.1.bias"].parseNDArray();
        auto m = weights["features.0.1.running_mean"].parseNDArray();
        auto v = weights["features.0.1.running_var"].parseNDArray();
        Tensor tg(32); tg.set(g); dbgConv0["bn_gamma"] = tg;
        Tensor tb(32); tb.set(b); dbgConv0["bn_beta"] = tb;
        Tensor tm(32); tm.set(m); dbgConv0["bn_mean"] = tm;
        Tensor tv(32); tv.set(v); dbgConv0["bn_var"] = tv;
    }

    // 2) IRB1 노드 새로 생성 + 가중치 로딩
    InvertedResidualBlock dbgIRB1(32, 16, 1, 1);
    loadIRBWeightsNoExpansion(dbgIRB1, weights, 1, 32, 16);

    // 3) 작은 네트워크 구성: input -> conv0 -> irb1 -> output
    NeuralNet netDbg(gDevice, 1, 1);
    netDbg.input(0) - dbgConv0 - dbgIRB1 - netDbg.output(0);

    // 4) 입력 텐서 준비 (224x224x3, 이미 전처리된 input 벡터 사용)
    Tensor inTensor(224, 224, 3);
    inTensor.set(input224);

    auto res = netDbg(std::move(inTensor));

    const auto& s = res[0].shape();   // [H, W, C]
    uint32_t H = s[0], W = s[1], C = s[2];
    size_t irbSize = (size_t)H * W * C;

    Buffer buf = gDevice.createBuffer({
        irbSize * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        });

    gDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(buf, res[0].buffer())
        .end()
        .submit()
        .wait();

    float* p = (float*)buf.map();
    std::vector<float> IRB1_output(p, p + irbSize);

    printf("IRB1 output shape: %u x %u x %u (size=%zu)\n", H, W, C, irbSize);
    printTensorStats(IRB1_output, "IRB1_output");
}


// ============================================================================
// 전체 MobileNetV2 분류
// ============================================================================
void testFullClassification()
{
    printf("\n");
    printf("============================================================\n");
    printf("    MobileNetV2 Full Classification (17 IRB blocks)\n");
    printf("============================================================\n");

    const char* weightsPath = PROJECT_CURRENT_DIR "/mobilenetv2_weights.safetensors";
    const char* imagePath = PROJECT_CURRENT_DIR "/images/dog.jpg";

    // ========================================
    // 1. 이미지 로드 및 전처리
    // ========================================
    printf("\n[1/4] Loading image...\n");
    int w, h, c;
    uint8_t* imgData = stbi_load(imagePath, &w, &h, &c, 3);
    if (!imgData) {
        printf("Failed to load image: %s\n", imagePath);
        return;
    }
    printf("  Image: %s (%d x %d)\n", imagePath, w, h);

    const int size = 224;
    std::vector<float> input(size * size * 3);
    int cropSize = std::min(w, h);
    int offsetX = (w - cropSize) / 2;
    int offsetY = (h - cropSize) / 2;
    const float mean[3] = { 0.485f, 0.456f, 0.406f };
    const float std_[3] = { 0.229f, 0.224f, 0.225f };

    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            int srcX = offsetX + (x * cropSize) / size;
            int srcY = offsetY + (y * cropSize) / size;
            srcX = std::max(0, std::min(w - 1, srcX));
            srcY = std::max(0, std::min(h - 1, srcY));
            int srcIdx = (srcY * w + srcX) * 3;
            int dstIdx = (y * size + x) * 3;
            for (int ch = 0; ch < 3; ++ch) {
                float pixel = imgData[srcIdx + ch] / 255.0f;
                input[dstIdx + ch] = (pixel - mean[ch]) / std_[ch];
            }
        }
    }
    stbi_image_free(imgData);
    printf("  Preprocessed to 224x224\n");

    // ========================================
    // 2. 가중치 로드
    // ========================================
    printf("\n[2/4] Loading weights...\n");
    SafeTensorsParser weights(weightsPath);
    printf("  Loaded %zu tensors\n", weights.getTensorNames().size());


    /* debug test */
    debugIRB1Output(weights, input);
    /* debug test */
    // ========================================
    // 3. 전체 네트워크 구성
    // ========================================
    printf("\n[3/4] Building full MobileNetV2...\n");

    // MobileNetV2 구조:
    // features.0:  Conv2d(3, 32, 3, stride=2, padding=1) + BN + ReLU6
    // features.1:  InvertedResidual(32, 16, stride=1, expand_ratio=1)
    // features.2:  InvertedResidual(16, 24, stride=2, expand_ratio=6)
    // features.3:  InvertedResidual(24, 24, stride=1, expand_ratio=6)
    // features.4:  InvertedResidual(24, 32, stride=2, expand_ratio=6)
    // features.5:  InvertedResidual(32, 32, stride=1, expand_ratio=6)
    // features.6:  InvertedResidual(32, 32, stride=1, expand_ratio=6)
    // features.7:  InvertedResidual(32, 64, stride=2, expand_ratio=6)
    // features.8:  InvertedResidual(64, 64, stride=1, expand_ratio=6)
    // features.9:  InvertedResidual(64, 64, stride=1, expand_ratio=6)
    // features.10: InvertedResidual(64, 64, stride=1, expand_ratio=6)
    // features.11: InvertedResidual(64, 96, stride=1, expand_ratio=6)
    // features.12: InvertedResidual(96, 96, stride=1, expand_ratio=6)
    // features.13: InvertedResidual(96, 96, stride=1, expand_ratio=6)
    // features.14: InvertedResidual(96, 160, stride=2, expand_ratio=6)
    // features.15: InvertedResidual(160, 160, stride=1, expand_ratio=6)
    // features.16: InvertedResidual(160, 160, stride=1, expand_ratio=6)
    // features.17: InvertedResidual(160, 320, stride=1, expand_ratio=6)
    // features.18: Conv2d(320, 1280, 1) + BN + ReLU6
    // classifier:  Dropout + Linear(1280, 1000)

    // === features.0: First Conv ===
    ConvBnRelu6Node conv0(3, 32, 3, 2, 1);  // 224 -> 112
    {
        auto w = weights["features.0.0.weight"].parseNDArray();

        /* test */
        if (w.size() >= 4) {
            printf(">>> DEBUG: Loaded Weights (First 4): %.6f, %.6f, %.6f, %.6f\n",
                w[0], w[1], w[2], w[3]);
        }
        else {
            printf(">>> DEBUG: Weight size is too small!\n");
        }
        /* test */
        Tensor tw(32 * 3 * 3 * 3); tw.set(w);
        conv0["conv_weight"] = tw;
        conv0["conv_bias"] = makeConstTensor(32, 0.0f);

        auto g = weights["features.0.1.weight"].parseNDArray();
        auto b = weights["features.0.1.bias"].parseNDArray();
        auto m = weights["features.0.1.running_mean"].parseNDArray();
        auto v = weights["features.0.1.running_var"].parseNDArray();
        Tensor tg(32); tg.set(g); conv0["bn_gamma"] = tg;
        Tensor tb(32); tb.set(b); conv0["bn_beta"] = tb;
        Tensor tm(32); tm.set(m); conv0["bn_mean"] = tm;
        Tensor tv(32); tv.set(v); conv0["bn_var"] = tv;
    }
    printf("  features.0: Conv(3->32, s=2) 224->112\n");



    // === features.1: IRB(32->16, t=1, s=1) ===
    InvertedResidualBlock irb1(32, 16, 1, 1);
    loadIRBWeightsNoExpansion(irb1, weights, 1, 32, 16);
    printf("  features.1: IRB(32->16, t=1, s=1)\n");

    // === features.2: IRB(16->24, t=6, s=2) ===
    InvertedResidualBlock irb2(16, 24, 6, 2);  // 112 -> 56
    loadIRBWeightsWithExpansion(irb2, weights, 2, 16, 24, 6);
    printf("  features.2: IRB(16->24, t=6, s=2) 112->56\n");

    // === features.3: IRB(24->24, t=6, s=1) ===
    InvertedResidualBlock irb3(24, 24, 6, 1);
    loadIRBWeightsWithExpansion(irb3, weights, 3, 24, 24, 6);
    printf("  features.3: IRB(24->24, t=6, s=1)\n");

    // === features.4: IRB(24->32, t=6, s=2) ===
    InvertedResidualBlock irb4(24, 32, 6, 2);  // 56 -> 28
    loadIRBWeightsWithExpansion(irb4, weights, 4, 24, 32, 6);
    printf("  features.4: IRB(24->32, t=6, s=2) 56->28\n");

    // === features.5: IRB(32->32, t=6, s=1) ===
    InvertedResidualBlock irb5(32, 32, 6, 1);
    loadIRBWeightsWithExpansion(irb5, weights, 5, 32, 32, 6);
    printf("  features.5: IRB(32->32, t=6, s=1)\n");

    // === features.6: IRB(32->32, t=6, s=1) ===
    InvertedResidualBlock irb6(32, 32, 6, 1);
    loadIRBWeightsWithExpansion(irb6, weights, 6, 32, 32, 6);
    printf("  features.6: IRB(32->32, t=6, s=1)\n");

    // === features.7: IRB(32->64, t=6, s=2) ===
    InvertedResidualBlock irb7(32, 64, 6, 2);  // 28 -> 14
    loadIRBWeightsWithExpansion(irb7, weights, 7, 32, 64, 6);
    printf("  features.7: IRB(32->64, t=6, s=2) 28->14\n");

    // === features.8: IRB(64->64, t=6, s=1) ===
    InvertedResidualBlock irb8(64, 64, 6, 1);
    loadIRBWeightsWithExpansion(irb8, weights, 8, 64, 64, 6);
    printf("  features.8: IRB(64->64, t=6, s=1)\n");

    // === features.9: IRB(64->64, t=6, s=1) ===
    InvertedResidualBlock irb9(64, 64, 6, 1);
    loadIRBWeightsWithExpansion(irb9, weights, 9, 64, 64, 6);
    printf("  features.9: IRB(64->64, t=6, s=1)\n");

    // === features.10: IRB(64->64, t=6, s=1) ===
    InvertedResidualBlock irb10(64, 64, 6, 1);
    loadIRBWeightsWithExpansion(irb10, weights, 10, 64, 64, 6);
    printf("  features.10: IRB(64->64, t=6, s=1)\n");

    // === features.11: IRB(64->96, t=6, s=1) ===
    InvertedResidualBlock irb11(64, 96, 6, 1);
    loadIRBWeightsWithExpansion(irb11, weights, 11, 64, 96, 6);
    printf("  features.11: IRB(64->96, t=6, s=1)\n");

    // === features.12: IRB(96->96, t=6, s=1) ===
    InvertedResidualBlock irb12(96, 96, 6, 1);
    loadIRBWeightsWithExpansion(irb12, weights, 12, 96, 96, 6);
    printf("  features.12: IRB(96->96, t=6, s=1)\n");

    // === features.13: IRB(96->96, t=6, s=1) ===
    InvertedResidualBlock irb13(96, 96, 6, 1);
    loadIRBWeightsWithExpansion(irb13, weights, 13, 96, 96, 6);
    printf("  features.13: IRB(96->96, t=6, s=1)\n");

    // === features.14: IRB(96->160, t=6, s=2) ===
    InvertedResidualBlock irb14(96, 160, 6, 2);  // 14 -> 7
    loadIRBWeightsWithExpansion(irb14, weights, 14, 96, 160, 6);
    printf("  features.14: IRB(96->160, t=6, s=2) 14->7\n");

    // === features.15: IRB(160->160, t=6, s=1) ===
    InvertedResidualBlock irb15(160, 160, 6, 1);
    loadIRBWeightsWithExpansion(irb15, weights, 15, 160, 160, 6);
    printf("  features.15: IRB(160->160, t=6, s=1)\n");

    // === features.16: IRB(160->160, t=6, s=1) ===
    InvertedResidualBlock irb16(160, 160, 6, 1);
    loadIRBWeightsWithExpansion(irb16, weights, 16, 160, 160, 6);
    printf("  features.16: IRB(160->160, t=6, s=1)\n");

    // === features.17: IRB(160->320, t=6, s=1) ===
    InvertedResidualBlock irb17(160, 320, 6, 1);
    loadIRBWeightsWithExpansion(irb17, weights, 17, 160, 320, 6);
    printf("  features.17: IRB(160->320, t=6, s=1)\n");

    // === features.18: Last Conv (320->1280) ===
    PwConvBnRelu6Node lastConv(320, 1280);
    {
        auto w = weights["features.18.0.weight"].parseNDArray();
        // [1280, 320, 1, 1] -> [320, 1280]
        std::vector<float> w_t(320 * 1280);
        for (int o = 0; o < 1280; ++o)
            for (int i = 0; i < 320; ++i)
                w_t[i * 1280 + o] = w[o * 320 + i];
        Tensor tw(320, 1280); tw.set(w_t);
        lastConv["pw_weight"] = tw;

        auto g = weights["features.18.1.weight"].parseNDArray();
        auto b = weights["features.18.1.bias"].parseNDArray();
        auto m = weights["features.18.1.running_mean"].parseNDArray();
        auto v = weights["features.18.1.running_var"].parseNDArray();
        Tensor tg(1280); tg.set(g); lastConv["bn_gamma"] = tg;
        Tensor tb(1280); tb.set(b); lastConv["bn_beta"] = tb;
        Tensor tm(1280); tm.set(m); lastConv["bn_mean"] = tm;
        Tensor tv(1280); tv.set(v); lastConv["bn_var"] = tv;
    }
    printf("  features.18: Conv(320->1280, 1x1)\n");

    // === Global Average Pooling ===
    GlobalAvgPoolNode gap;
    printf("  GAP: 7x7x1280 -> 1280\n");

    // === Classifier ===
    FCMobileNetNode fc(1280, 1000);
    {
        auto w = weights["classifier.1.weight"].parseNDArray();
        auto b = weights["classifier.1.bias"].parseNDArray();
        // [1000, 1280] -> [1280, 1000]
        std::vector<float> w_t(1280 * 1000);
        for (int o = 0; o < 1000; ++o)
            for (int i = 0; i < 1280; ++i)
                w_t[i * 1000 + o] = w[o * 1280 + i];
        Tensor tw(1280, 1000); tw.set(w_t);
        Tensor tb(1000); tb.set(b);
        fc["weight"] = tw;
        fc["bias"] = tb;
    }
    printf("  classifier: FC(1280->1000)\n");

    printf("  Total: 1 Conv + 17 IRB + 1 Conv + GAP + FC\n");

    // ========================================
    // 4. 추론 실행
    // ========================================
    printf("\n[4/4] Running inference...\n");

    Tensor inputTensor(224, 224, 3);
    inputTensor.set(input);

    NeuralNet net(gDevice, 1, 1);
    net.input(0)
        - conv0
        - irb1 - irb2 - irb3 - irb4 - irb5 - irb6 - irb7
        - irb8 - irb9 - irb10 - irb11 - irb12 - irb13
        - irb14 - irb15 - irb16 - irb17
        - lastConv - gap - fc
        - net.output(0);

    auto start = std::chrono::high_resolution_clock::now();
    auto results = net(std::move(inputTensor));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("  Inference time: %lld ms\n", (long long)duration.count());

    // ========================================
    // 5. 결과 출력
    // ========================================
    Buffer outBuffer = gDevice.createBuffer({
        1000 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        });

    gDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuffer, results[0].buffer())
        .end()
        .submit()
        .wait();

    float* logits = (float*)outBuffer.map();
    std::vector<float> probs(logits, logits + 1000);
    softmax(probs.data(), 1000);
    auto top5 = getTopK(probs.data(), 1000, 5);

    printf("\n");
    printf("============================================================\n");
    printf("                    CLASSIFICATION RESULTS\n");
    printf("============================================================\n");
    printf("\n  Image: %s\n\n", imagePath);
    printf("  Top-5 Predictions:\n");
    printf("  ----------------------------------------------------------\n");
    for (int i = 0; i < 5; ++i) {
        int idx = top5[i];
        printf("  %d. %-25s (class %3d): %.2f%%\n",
            i + 1, getImageNetClassName(idx), idx, probs[idx] * 100.0f);
    }
    printf("  ----------------------------------------------------------\n");
    printf("\n  >>> Predicted: %s <<<\n", getImageNetClassName(top5[0]));
    printf("\n============================================================\n");
}


// ============================================================================
// Main
// ============================================================================

void test_mobilenet()
{
    printf("========================================\n");
    printf("MobileNetV2 Node Tests\n");
    printf("========================================\n");

    try {
        testRelu6();
        testDepthwiseConv();
        testPointwiseConv();
        testGlobalAvgPool();
        testConvBnRelu6();
        testInvertedResidualBlock();
        testSimplePipeline();
        testLoadWeights();
        testImageClassification();
        testFullClassification();
   

        printf("\n========================================\n");
        printf("All tests completed!\n");
        printf("========================================\n");
    }
    catch (const std::exception& e) {
        printf("\nError: %s\n", e.what());
    }
}

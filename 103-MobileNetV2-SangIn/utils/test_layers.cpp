#include "test_layers.h"
#include <stb/stb_image.h>
#include "../library/neuralNodes.h"
#include "../library/vulkanApp.h"
#include "../library/timeChecker.hpp"


Tensor makeRandomTensor(const std::vector<uint32_t>& shape, std::mt19937& gen, float stddev)
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

bool allClose(const float* data, const float* expected, int n, float eps)
{
    for (int i = 0; i < n; ++i)
    {
        if (std::abs(data[i] - expected[i]) > eps)
            return false;
    }

    return true;
}

void testRelu6()
{
    printf("\n=== Testing ReLU6 Node ===\n");

    Relu6Node relu6;

    std::vector<float> inputData = { -2.0f, -1.0f, 0.0f, 1.0f, 3.0f, 5.0f, 6.0f, 7.0f, 10.0f };
    Tensor input(3, 3, 1);
    input.set(inputData);

    NeuralNet net(netGlobalDevice, 1, 1);
    net.input(0) - relu6 - net.output(0);

    auto results = net(std::move(input));

    Buffer outBuffer = netGlobalDevice.createBuffer({
        9 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        });

    netGlobalDevice.newCommandBuffer(queue_compute)
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
    printf("\n\n");

    allClose(data, expected, 9) ? printf("Result: PASS\n") : printf("Result: FAIL\n");
}

void testDepthwiseConv()
{
    printf("\n=== Testing Depthwise Conv Node ===\n");

    const uint32_t H = 4, W = 4, C = 2;
    const uint32_t K = 3;

    DepthwiseConvNode dwconv(C, K, 1);  // stride=1, pad=1

    std::vector<float> inputData(H * W * C, 1.0f);
    Tensor input(H, W, C);
    input.set(inputData);

    std::vector<float> weightData(K * K * C, 1.0f);
    Tensor weight(K * K, C);
    weight.set(weightData);

    dwconv["weight"] = weight;

    NeuralNet net(netGlobalDevice, 1, 1);
    net.input(0) - dwconv - net.output(0);

    auto results = net(std::move(input));

    const auto& outShape = results[0].shape();
    printf("Input shape: %d x %d x %d\n", H, W, C);
    printf("Output shape: %d x %d x %d\n", outShape[0], outShape[1], outShape[2]);

    size_t outSize = outShape[0] * outShape[1] * outShape[2];
    Buffer outBuffer = netGlobalDevice.createBuffer({
        outSize * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        });

    netGlobalDevice.newCommandBuffer(queue_compute)
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

    NeuralNet net(netGlobalDevice, 1, 1);
    net.input(0) - pwconv - net.output(0);

    auto results = net(std::move(input));

    const auto& outShape = results[0].shape();
    printf("Input shape: %d x %d x %d\n", H, W, C_in);
    printf("Output shape: %d x %d x %d\n", outShape[0], outShape[1], outShape[2]);

    size_t outSize = outShape[0] * outShape[1] * outShape[2];
    Buffer outBuffer = netGlobalDevice.createBuffer({
        outSize * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        });

    netGlobalDevice.newCommandBuffer(queue_compute)
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

    NeuralNet net(netGlobalDevice, 1, 1);
    net.input(0) - gap - net.output(0);

    auto results = net(std::move(input));

    const auto& outShape = results[0].shape();
    printf("Input shape: %d x %d x %d\n", H, W, C);
    printf("Output shape: %d\n", outShape[0]);

    Buffer outBuffer = netGlobalDevice.createBuffer({
        C * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        });

    netGlobalDevice.newCommandBuffer(queue_compute)
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


void testConvBnReLU6()
{
    printf("\n=== Testing Conv+BN+ReLU6 Combined Node ===\n");

    const uint32_t H = 8, W = 8, C_in = 3, C_out = 16;
    const uint32_t K = 3, stride = 2, pad = 1;

    ConvBNReLU6 convBnRelu6(C_in, C_out, K, stride, pad);

    std::mt19937 gen(42);

    std::vector<float> inputData(H * W * C_in, 0.5f);
    Tensor input(H, W, C_in);
    input.set(inputData);

    convBnRelu6["conv.weight"] = makeRandomTensor({ C_in * K * K, C_out }, gen, 0.1f);
    convBnRelu6["conv.bias"] = makeConstTensor(C_out, 0.0f);
    convBnRelu6["bn.mean"] = makeConstTensor(C_out, 0.0f);
    convBnRelu6["bn.variance"] = makeConstTensor(C_out, 1.0f);
    convBnRelu6["bn.gamma"] = makeConstTensor(C_out, 1.0f);
    convBnRelu6["bn.beta"] = makeConstTensor(C_out, 0.0f);
    
    NeuralNet net(netGlobalDevice, 1, 1);
    net.input(0) - convBnRelu6 - net.output(0);

    auto results = net(std::move(input));

    const auto& outShape = results[0].shape();
    uint32_t expectedH = (H + 2 * pad - K) / stride + 1;
    uint32_t expectedW = (W + 2 * pad - K) / stride + 1;

    printf("Input shape: %d x %d x %d\n", H, W, C_in);
    printf("Output shape: %d x %d x %d\n", outShape[0], outShape[1], outShape[2]);
    printf("Expected shape: %d x %d x %d\n", expectedH, expectedW, C_out);

    size_t outSize = outShape[0] * outShape[1] * outShape[2];
    Buffer outBuffer = netGlobalDevice.createBuffer({
        outSize * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        });

    netGlobalDevice.newCommandBuffer(queue_compute)
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
    irb["pwConvBNReLU6.pointwiseConv.weight"] = makeRandomTensor({ C_in, hiddenDim }, gen);
    irb["pwConvBNReLU6.bn.gamma"] = makeConstTensor(hiddenDim, 1.0f);
    irb["pwConvBNReLU6.bn.beta"] = makeConstTensor(hiddenDim, 0.0f);
    irb["pwConvBNReLU6.bn.mean"] = makeConstTensor(hiddenDim, 0.0f);
    irb["pwConvBNReLU6.bn.variance"] = makeConstTensor(hiddenDim, 1.0f);

    // Depthwise layer weights
    irb["dwConvBNReLU6.depthwiseConv.weight"] = makeRandomTensor({ 9, hiddenDim }, gen);
    irb["dwConvBNReLU6.bn.gamma"] = makeConstTensor(hiddenDim, 1.0f);
    irb["dwConvBNReLU6.bn.beta"] = makeConstTensor(hiddenDim, 0.0f);
    irb["dwConvBNReLU6.bn.mean"] = makeConstTensor(hiddenDim, 0.0f);
    irb["dwConvBNReLU6.bn.variance"] = makeConstTensor(hiddenDim, 1.0f);

    // Projection layer weights
    irb["pwConvBN.pointwiseConv.weight"] = makeRandomTensor({ hiddenDim, C_out }, gen);
    irb["pwConvBN.bn.gamma"] = makeConstTensor(C_out, 1.0f);
    irb["pwConvBN.bn.beta"] = makeConstTensor(C_out, 0.0f);
    irb["pwConvBN.bn.mean"] = makeConstTensor(C_out, 0.0f);
    irb["pwConvBN.bn.variance"] = makeConstTensor(C_out, 1.0f);

    NeuralNet net(netGlobalDevice, 1, 1);
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
    Buffer outBuffer = netGlobalDevice.createBuffer({
        outSize * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        });

    netGlobalDevice.newCommandBuffer(queue_compute)
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
    ConvBNReLU6 conv1(inputC, conv1OutC, 3, 2, 1);
    InvertedResidualBlock irb1(conv1OutC, irb1OutC, 1, 1);  // t=1
    GlobalAvgPoolNode gap;
    FullyConnectedNode fc(irb1OutC, numClasses);

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
    NeuralNet net(netGlobalDevice, 1, 1);
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
    Buffer outBuffer = netGlobalDevice.createBuffer({
        numClasses * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        });

    netGlobalDevice.newCommandBuffer(queue_compute)
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

        // ù ��° Conv ����ġ Ȯ��
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
    const char* imagePath = PROJECT_CURRENT_DIR "/images/cat.jpg";  // �ϳ��� �׽�Ʈ

    // 1. �̹��� �ε�
    int w, h, c;
    uint8_t* data = stbi_load(imagePath, &w, &h, &c, 3);
    if (!data) {
        printf("Failed to load image: %s\n", imagePath);
        printf("Please put an image in the images folder!\n");
        return;
    }
    printf("Loaded image: %s (%d x %d)\n", imagePath, w, h);

    // 2. 224x224�� �������� + ����ȭ
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

    // 3. ����ġ �ε�
    SafeTensorsParser weights(weightsPath);
    printf("Loaded weights\n");

    // 4. ù ��° Conv + BN + ReLU6 ���̾ �׽�Ʈ
    ConvBNReLU6 conv1(3, 32, 3, 2, 1);

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

    // 5. �Է� �ټ� ����
    Tensor inputTensor(224, 224, 3);
    inputTensor.set(input);

    // 6. ��Ʈ��ũ ����
    GlobalAvgPoolNode gap;

    NeuralNet net(netGlobalDevice, 1, 1);
    net.input(0) - conv1 - gap - net.output(0);

    printf("Running first layer...\n");
    auto results = net(std::move(inputTensor));

    // 7. ��� ���
    const auto& outShape = results[0].shape();
    size_t outSize = 1;
    for (auto d : outShape) outSize *= d;

    Buffer outBuffer = netGlobalDevice.createBuffer({
        outSize * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        });

    netGlobalDevice.newCommandBuffer(queue_compute)
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

#include "neuralNet.h"
#include "neuralNodes.h"
#include "jsonParser.h"
#include "timeChecker.hpp"
#include <stb/stb_image.h>
#include <cstring>  // memcpy

#include "SuffleNet.h"

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

// TODO: weight initialization
Tensor eval_shufflenet(const std::vector<float>& srcImage, const JsonParser& json, uint32_t iter) // srcImage layout: [H][W][C]
{
    SuffleNet suffleNet(netGlobalDevice);

    suffleNet["conv0.weight"] = Tensor(json["layer1.0.weight"]).reshape(32, 1*3*3).permute(1, 0);
    suffleNet["conv0.bias"] = Tensor(json["layer1.0.bias"]);
    suffleNet["conv1.weight"] = Tensor(json["layer2.0.weight"]).reshape(64, 32*3*3).permute(1, 0);
    suffleNet["conv1.bias"] = Tensor(json["layer2.0.bias"]);
    suffleNet["weight"] = Tensor(json["fc.weight"]).reshape(10, 64, 7*7).permute(2, 1, 0).reshape(7*7*64, 10);
    suffleNet["bias"] = Tensor(json["fc.bias"]);
    
    Tensor result;
    Tensor inputTensor = Tensor(28, 28, 1).set(srcImage);

    for (uint32_t i = 0; i < iter; ++i)
        result = suffleNet(inputTensor)[0];

    return result;
}

void eval_adaptive_avgpool()
{
    void loadShaders();
    loadShaders();

    const uint32_t H = 5, W = 5, C = 3;   // 5x5x3 input
    const uint32_t outH = 1, outW = 1;    // 1x1 adaptive average pooling

    // Create a simple input pattern (HWC layout)
    // Value encodes position and channel to make verification easy
    std::vector<float> input(H * W * C);
    for (uint32_t h = 0; h < H; ++h)
        for (uint32_t w = 0; w < W; ++w)
            for (uint32_t c = 0; c < C; ++c)
                input[(h * W + w) * C + c] = float(h * W + w + c);

    // Build a tiny net: input -> AdaptiveAvgPooling(outH,outW) -> output
    NeuralNet net(netGlobalDevice, 1, 1);
    AdaptiveAvgPoolingNode aap(outH, outW);
    net.input(0) - aap - net.output(0);

    Tensor out = net(Tensor(H, W, C).set(input))[0];

    // Print input for verification (per-channel 5x5)
    printf("AdaptiveAvgPool input (H=%u, W=%u, C=%u):\n", H, W, C);
    for (uint32_t c = 0; c < C; ++c) {
        printf("  Channel %u:\n", c);
        for (uint32_t h = 0; h < H; ++h) {
            printf("    ");
            for (uint32_t w = 0; w < W; ++w) {
                float v = input[(h * W + w) * C + c];
                printf("%6.2f ", v);
            }
            printf("\n");
        }
    }

    // Read back and print (1x1xC -> C floats)
    uint32_t outCount = outH * outW * C;
    vk::Buffer outBuf = netGlobalDevice.createBuffer({
        outCount * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });

    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuf, out.buffer())
        .end()
        .submit()
        .wait();

    std::vector<float> host(outCount);
    memcpy(host.data(), outBuf.map(), outCount * sizeof(float));

    printf("AdaptiveAvgPool test: %ux%ux%u -> %ux%ux%u\n", H, W, C, outH, outW, C);
    for (uint32_t c = 0; c < C; ++c)
        printf("  out[0,0,%u] = %f\n", c, host[c]);
}

void eval_hs()
{
    void loadShaders();
    loadShaders();

    const uint32_t H = 3, W = 3, C = 3; // 3x3x3 input

    // Build input (HWC). Values easy to track
    std::vector<float> input(H * W * C);
    for (uint32_t h = 0; h < H; ++h)
        for (uint32_t w = 0; w < W; ++w)
            for (uint32_t c = 0; c < C; ++c)
                input[(h * W + w) * C + c] = float((int)h - 1) + float((int)w - 1) + float(c); // some negatives & positives

    NeuralNet net(netGlobalDevice, 1, 1);
    HSNode hs;
    net.input(0) - hs - net.output(0);

    Tensor out = net(Tensor(H, W, C).set(input))[0];

    // Print input
    printf("HS input (H=%u,W=%u,C=%u):\n", H, W, C);
    for (uint32_t c = 0; c < C; ++c) {
        printf("  Channel %u:\n", c);
        for (uint32_t h = 0; h < H; ++h) {
            printf("    ");
            for (uint32_t w = 0; w < W; ++w) {
                float v = input[(h * W + w) * C + c];
                printf("%6.2f ", v);
            }
            printf("\n");
        }
    }

    // Read back out and print
    uint32_t outCount = H * W * C;
    vk::Buffer outBuf = netGlobalDevice.createBuffer({
        outCount * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });

    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuf, out.buffer())
        .end()
        .submit()
        .wait();

    std::vector<float> host(outCount);
    memcpy(host.data(), outBuf.map(), outCount * sizeof(float));

    printf("HS output: \n");
    for (uint32_t c = 0; c < C; ++c) {
        printf("  Channel %u:\n", c);
        for (uint32_t h = 0; h < H; ++h) {
            printf("    ");
            for (uint32_t w = 0; w < W; ++w) {
                float v = host[(h * W + w) * C + c];
                printf("%6.3f ", v);
            }
            printf("\n");
        }
    }
}

void eval_multiply()
{
    void loadShaders();
    loadShaders();

    const uint32_t H = 3, W = 3, C = 3; // 3x3x3 input

    // Input HWC
    std::vector<float> input(H * W * C);
    for (uint32_t h = 0; h < H; ++h)
        for (uint32_t w = 0; w < W; ++w)
            for (uint32_t c = 0; c < C; ++c)
                input[(h * W + w) * C + c] = float(h * W + w + c);

    // Attention 1x1xC (broadcast over H and W)
    std::vector<float> atten(C);
    for (uint32_t c = 0; c < C; ++c)
        atten[c] = 0.5f + 0.5f * float(c); // [0.5, 1.0, 1.5]

    NeuralNet net(netGlobalDevice, 2, 1);
    MultiplyNode mul;

    // Connect first input to mul.in0 and second input to mul.atten
    net.input(0) - mul;                  // to in0
    net.input(1) - ("atten" / mul);     // to atten
    mul - net.output(0);

    auto outputs = net(Tensor(H, W, C).set(input), Tensor(1, 1, C).set(atten));
    Tensor out = outputs[0];

    // Print inputs
    printf("Multiply input X (H=%u,W=%u,C=%u):\n", H, W, C);
    for (uint32_t c = 0; c < C; ++c) {
        printf("  Channel %u:\n", c);
        for (uint32_t h = 0; h < H; ++h) {
            printf("    ");
            for (uint32_t w = 0; w < W; ++w) {
                float v = input[(h * W + w) * C + c];
                printf("%6.2f ", v);
            }
            printf("\n");
        }
    }
    printf("Attention vector (1x1xC): ");
    for (uint32_t c = 0; c < C; ++c)
        printf("%5.2f ", atten[c]);
    printf("\n");

    // Read back and print output
    uint32_t outCount = H * W * C;
    vk::Buffer outBuf = netGlobalDevice.createBuffer({
        outCount * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });

    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuf, out.buffer())
        .end()
        .submit()
        .wait();

    std::vector<float> host(outCount);
    memcpy(host.data(), outBuf.map(), outCount * sizeof(float));

    printf("Multiply output Y = X * atten[c]:\n");
    for (uint32_t c = 0; c < C; ++c) {
        printf("  Channel %u:\n", c);
        for (uint32_t h = 0; h < H; ++h) {
            printf("    ");
            for (uint32_t w = 0; w < W; ++w) {
                float v = host[(h * W + w) * C + c];
                printf("%6.2f ", v);
            }
            printf("\n");
        }
    }
}

void eval_batchnorm2d()
{
    void loadShaders();
    loadShaders();

    const uint32_t H = 3, W = 3, C = 3; // 3x3x3 input

    // Input HWC with a simple pattern
    std::vector<float> input(H * W * C);
    for (uint32_t h = 0; h < H; ++h)
        for (uint32_t w = 0; w < W; ++w)
            for (uint32_t c = 0; c < C; ++c)
                input[(h * W + w) * C + c] = float(h * W + w) + 0.1f * float(c);

    // BN params per channel
    std::vector<float> gamma = {1.0f, 0.5f, 2.0f};
    std::vector<float> beta  = {0.0f, 1.0f, -1.0f};
    std::vector<float> mean  = {3.0f, 4.0f, 5.0f};
    std::vector<float> var   = {1.0f, 0.25f, 4.0f};

    NeuralNet net(netGlobalDevice, 5, 1);
    BatchNormNode bn(C);

    // input(0) -> in0, input(1)->gamma, (2)->beta, (3)->running_mean, (4)->running_var
    net.input(0) - bn;                              // in0
    net.input(1) - ("gamma" / bn);
    net.input(2) - ("beta" / bn);
    net.input(3) - ("running_mean" / bn);
    net.input(4) - ("running_var" / bn);
    bn - net.output(0);

    auto outputs = net(
        Tensor(H, W, C).set(input),
        Tensor(C).set(gamma),
        Tensor(C).set(beta),
        Tensor(C).set(mean),
        Tensor(C).set(var)
    );
    Tensor out = outputs[0];

    // Print inputs
    printf("BN input X (H=%u,W=%u,C=%u):\n", H, W, C);
    for (uint32_t c = 0; c < C; ++c) {
        printf("  Channel %u:\n", c);
        for (uint32_t h = 0; h < H; ++h) {
            printf("    ");
            for (uint32_t w = 0; w < W; ++w) {
                float v = input[(h * W + w) * C + c];
                printf("%6.2f ", v);
            }
            printf("\n");
        }
    }
    printf("gamma: "); for (auto v: gamma) printf("%5.2f ", v); printf("\n");
    printf("beta : "); for (auto v: beta ) printf("%5.2f ", v); printf("\n");
    printf("mean : "); for (auto v: mean ) printf("%5.2f ", v); printf("\n");
    printf("var  : "); for (auto v: var  ) printf("%5.2f ", v); printf("\n");

    // Read back and print output
    uint32_t outCount = H * W * C;
    vk::Buffer outBuf = netGlobalDevice.createBuffer({
        outCount * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });

    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuf, out.buffer())
        .end()
        .submit()
        .wait();

    std::vector<float> host(outCount);
    memcpy(host.data(), outBuf.map(), outCount * sizeof(float));

    printf("BN output Y: \n");
    for (uint32_t c = 0; c < C; ++c) {
        printf("  Channel %u:\n", c);
        for (uint32_t h = 0; h < H; ++h) {
            printf("    ");
            for (uint32_t w = 0; w < W; ++w) {
                float v = host[(h * W + w) * C + c];
                printf("%7.3f ", v);
            }
            printf("\n");
        }
    }
}

void Run()
{
    void loadShaders();
    loadShaders();

    const uint32_t channels = 1;
    auto [srcImage, width, height] = readImage<channels>(PROJECT_CURRENT_DIR"/data/0.png");
    _ASSERT(width == 28 && height == 28);
    _ASSERT(width * height * channels == srcImage.size());

    std::vector<float> inputData(width * height * channels);
    for (size_t i = 0; i < srcImage.size(); ++i)
        inputData[i] = srcImage[i] / 255.0f;

    JsonParser json = JsonParser(PROJECT_CURRENT_DIR"/weights.json");

    uint32_t iter = 1;
    Tensor eval;

    {
        TimeChecker timer("(VAI) SuffleNet evaluation: {} iterations", iter);
        eval = eval_shufflenet(inputData, json, iter);
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

    float data[10];
    memcpy(data, outBuffer.map(), 10 * sizeof(float));

    for(int i=0; i<10; ++i)
        printf("data[%d] = %f\n", i, data[i]);
}

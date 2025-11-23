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

void eval_channel_shuffle_concat()
{
    void loadShaders();
    loadShaders();

    const uint32_t H = 3, W = 3, C = 4; // 3x3x4 input

    // Build input (HWC) with a simple increasing pattern
    std::vector<float> input(H * W * C);
    for (uint32_t h = 0; h < H; ++h)
        for (uint32_t w = 0; w < W; ++w)
            for (uint32_t c = 0; c < C; ++c)
                input[(h * W + w) * C + c] = float((h * W + w) * C + c);

    NeuralNet net(netGlobalDevice, 1, 1);
    ChannelShuffleNode cs(C);
    ConcatNode concat;

    // input -> channel shuffle -> concat -> output
    net.input(0) - ("in0" / cs);
    (cs / "out_even") - ("in0" / concat);
    (cs / "out_odd")  - ("in1" / concat);
    concat - net.output(0);

    auto outputs = net(Tensor(H, W, C).set(input));
    Tensor out = outputs[0];

    // Read back output
    uint32_t count = H * W * C;
    vk::Buffer outBuf = netGlobalDevice.createBuffer({
        count * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });

    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuf, out.buffer())
        .end()
        .submit()
        .wait();

    std::vector<float> host(count);
    memcpy(host.data(), outBuf.map(), count * sizeof(float));

    printf("ChannelShuffle + Concat test (H=%u,W=%u,C=%u):\n", H, W, C);
    printf("Input vs Reconstructed Output:\n");
    for (uint32_t h = 0; h < H; ++h) {
        for (uint32_t w = 0; w < W; ++w) {
            printf("  (h=%u,w=%u): ", h, w);
            for (uint32_t c = 0; c < C; ++c) {
                uint32_t idx = (h * W + w) * C + c;
                printf("[%5.1f -> %5.1f] ", input[idx], host[idx]);
            }
            printf("\n");
        }
    }
}

static void load_shufflenet_weights(SuffleNetV2& net, const JsonParser& json, const std::vector<int>& architecture)
{
    auto set = [&](const std::string& key, const JsonParserRef& ref) {
        try { net[key] = Tensor(ref); } catch (...) {}
    };

    // Stem
    try {
        auto stem = json["stem"];
        set("stem.conv.weight", stem["conv.weight"]);
        set("stem.conv.bias",   stem["conv.bias"]);
        set("stem.bn.gamma",    stem["bn.gamma"]);
        set("stem.bn.beta",     stem["bn.beta"]);
        set("stem.bn.running_mean", stem["bn.running_mean"]);
        set("stem.bn.running_var",  stem["bn.running_var"]);
    } catch (...) {}

    // Features
    auto feats = json["features"];
    auto is_stride2 = [&](size_t idx){ return idx==0 || idx==4 || idx==8 || idx==16; };

    for (size_t i = 0; i < architecture.size(); ++i)
    {
        auto bj = feats[static_cast<uint32_t>(i)];
        char idxs[32]; sprintf(idxs, "%zu", i);
        std::string base = std::string("features.") + idxs + ".";
        bool stride2 = is_stride2(i);
        if (architecture[i] == 3) // Xception
        {
            set(base+"dw1.weight", bj["dw1.weight"]);
            set(base+"dw1.bias",   bj["dw1.bias"]);
            set(base+"bn1.gamma",  bj["bn1.gamma"]);
            set(base+"bn1.beta",   bj["bn1.beta"]);
            set(base+"bn1.running_mean", bj["bn1.running_mean"]);
            set(base+"bn1.running_var",  bj["bn1.running_var"]);

            set(base+"pw1.weight", bj["pw1.weight"]);
            set(base+"pw1.bias",   bj["pw1.bias"]);
            set(base+"bn1p.gamma", bj["bn1p.gamma"]);
            set(base+"bn1p.beta",  bj["bn1p.beta"]);
            set(base+"bn1p.running_mean", bj["bn1p.running_mean"]);
            set(base+"bn1p.running_var",  bj["bn1p.running_var"]);

            set(base+"dw2.weight", bj["dw2.weight"]);
            set(base+"dw2.bias",   bj["dw2.bias"]);
            set(base+"bn2.gamma",  bj["bn2.gamma"]);
            set(base+"bn2.beta",   bj["bn2.beta"]);
            set(base+"bn2.running_mean", bj["bn2.running_mean"]);
            set(base+"bn2.running_var",  bj["bn2.running_var"]);

            set(base+"pw2.weight", bj["pw2.weight"]);
            set(base+"pw2.bias",   bj["pw2.bias"]);
            set(base+"bn2p.gamma", bj["bn2p.gamma"]);
            set(base+"bn2p.beta",  bj["bn2p.beta"]);
            set(base+"bn2p.running_mean", bj["bn2p.running_mean"]);
            set(base+"bn2p.running_var",  bj["bn2p.running_var"]);

            set(base+"dw3.weight", bj["dw3.weight"]);
            set(base+"dw3.bias",   bj["dw3.bias"]);
            set(base+"bn3.gamma",  bj["bn3.gamma"]);
            set(base+"bn3.beta",   bj["bn3.beta"]);
            set(base+"bn3.running_mean", bj["bn3.running_mean"]);
            set(base+"bn3.running_var",  bj["bn3.running_var"]);

            set(base+"pw3.weight", bj["pw3.weight"]);
            set(base+"pw3.bias",   bj["pw3.bias"]);
            set(base+"bn3p.gamma", bj["bn3p.gamma"]);
            set(base+"bn3p.beta",  bj["bn3p.beta"]);
            set(base+"bn3p.running_mean", bj["bn3p.running_mean"]);
            set(base+"bn3p.running_var",  bj["bn3p.running_var"]);

            if (stride2)
            {
                set(base+"proj.dw.weight", bj["proj.dw.weight"]);
                set(base+"proj.dw.bias",   bj["proj.dw.bias"]);
                set(base+"proj.bn1.gamma", bj["proj.bn1.gamma"]);
                set(base+"proj.bn1.beta",  bj["proj.bn1.beta"]);
                set(base+"proj.bn1.running_mean", bj["proj.bn1.running_mean"]);
                set(base+"proj.bn1.running_var",  bj["proj.bn1.running_var"]);
                set(base+"proj.pw.weight", bj["proj.pw.weight"]);
                set(base+"proj.pw.bias",   bj["proj.pw.bias"]);
                set(base+"proj.bn2.gamma", bj["proj.bn2.gamma"]);
                set(base+"proj.bn2.beta",  bj["proj.bn2.beta"]);
                set(base+"proj.bn2.running_mean", bj["proj.bn2.running_mean"]);
                set(base+"proj.bn2.running_var",  bj["proj.bn2.running_var"]);
            }
        }
        else // ShuffleUnit
        {
            set(base+"pw1.weight", bj["pw1.weight"]);
            set(base+"pw1.bias",   bj["pw1.bias"]);
            set(base+"bn1.gamma",  bj["bn1.gamma"]);
            set(base+"bn1.beta",   bj["bn1.beta"]);
            set(base+"bn1.running_mean", bj["bn1.running_mean"]);
            set(base+"bn1.running_var",  bj["bn1.running_var"]);

            set(base+"dw.weight", bj["dw.weight"]);
            set(base+"dw.bias",   bj["dw.bias"]);
            set(base+"bn2.gamma", bj["bn2.gamma"]);
            set(base+"bn2.beta",  bj["bn2.beta"]);
            set(base+"bn2.running_mean", bj["bn2.running_mean"]);
            set(base+"bn2.running_var",  bj["bn2.running_var"]);

            set(base+"pw2.weight", bj["pw2.weight"]);
            set(base+"pw2.bias",   bj["pw2.bias"]);
            set(base+"bn3.gamma",  bj["bn3.gamma"]);
            set(base+"bn3.beta",   bj["bn3.beta"]);
            set(base+"bn3.running_mean", bj["bn3.running_mean"]);
            set(base+"bn3.running_var",  bj["bn3.running_var"]);

            if (stride2)
            {
                set(base+"proj.dw.weight", bj["proj.dw.weight"]);
                set(base+"proj.dw.bias",   bj["proj.dw.bias"]);
                set(base+"proj.bn1.gamma", bj["proj.bn1.gamma"]);
                set(base+"proj.bn1.beta",  bj["proj.bn1.beta"]);
                set(base+"proj.bn1.running_mean", bj["proj.bn1.running_mean"]);
                set(base+"proj.bn1.running_var",  bj["proj.bn1.running_var"]);
                set(base+"proj.pw.weight", bj["proj.pw.weight"]);
                set(base+"proj.pw.bias",   bj["proj.pw.bias"]);
                set(base+"proj.bn2.gamma", bj["proj.bn2.gamma"]);
                set(base+"proj.bn2.beta",  bj["proj.bn2.beta"]);
                set(base+"proj.bn2.running_mean", bj["proj.bn2.running_mean"]);
                set(base+"proj.bn2.running_var",  bj["proj.bn2.running_var"]);
            }

            // optional SE
            try {
                auto se = bj["se"];
                set(base+"se.conv1.weight", se["conv1.weight"]);
                set(base+"se.conv1.bias",   se["conv1.bias"]);
                set(base+"se.bn.gamma",     se["bn.gamma"]);
                set(base+"se.bn.beta",      se["bn.beta"]);
                set(base+"se.bn.running_mean", se["bn.running_mean"]);
                set(base+"se.bn.running_var",  se["bn.running_var"]);
                set(base+"se.conv2.weight", se["conv2.weight"]);
                set(base+"se.conv2.bias",   se["conv2.bias"]);
            } catch (...) {}
        }
    }

    // Tail
    try {
        auto last = json["last"];
        set("last.conv.weight", last["conv.weight"]);
        set("last.conv.bias",   last["conv.bias"]);
        set("last.bn.gamma",    last["bn.gamma"]);
        set("last.bn.beta",     last["bn.beta"]);
        set("last.bn.running_mean", last["bn.running_mean"]);
        set("last.bn.running_var",  last["bn.running_var"]);
    } catch (...) {}

    // Last SE (optional)
    try {
        auto se = json["lastSE"];
        set("lastSE.conv1.weight", se["conv1.weight"]);
        set("lastSE.conv1.bias",   se["conv1.bias"]);
        set("lastSE.bn.gamma",     se["bn.gamma"]);
        set("lastSE.bn.beta",      se["bn.beta"]);
        set("lastSE.bn.running_mean", se["bn.running_mean"]);
        set("lastSE.bn.running_var",  se["bn.running_var"]);
        set("lastSE.conv2.weight", se["conv2.weight"]);
        set("lastSE.conv2.bias",   se["conv2.bias"]);
    } catch (...) {}

    // FCs
    try { auto fc = json["fc1"]; set("fc1.weight", fc["weight"]); set("fc1.bias", fc["bias"]); } catch (...) {}
    try { auto clf = json["classifier"]; set("classifier.weight", clf["weight"]); set("classifier.bias", clf["bias"]); } catch (...) {}
}

void Run()
{
    void loadShaders();
    loadShaders();

    // 1) Load exported weights
    JsonParser json = JsonParser(PROJECT_CURRENT_DIR"/weights.json");
    printf("load weight\n");

    // 2) Build ShuffleNetV2+ (Small) architecture to match exporter
    std::vector<int> architecture = {0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2};
    SuffleNetV2 net(netGlobalDevice, architecture, 1000);
    printf("set model\n");
    load_shufflenet_weights(net, json, architecture);

    printf("load model\n");

    // 3) Prepare a dummy RGB input (224x224x3). Replace with image if desired.
    const uint32_t H = 224, W = 224, C = 3;
    std::vector<float> input(H * W * C, 0.5f);

    // 4) Run inference
    Tensor logits = net(Tensor(H, W, C).set(input))[0];

    // 5) Read back and print first 10 outputs
    const uint32_t outN = 10; // print top-10 indices raw (not sorted)
    vk::Buffer outBuffer = netGlobalDevice.createBuffer({
        outN * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });

    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuffer, logits.buffer())
        .end()
        .submit()
        .wait();

    float data[outN];
    memcpy(data, outBuffer.map(), outN * sizeof(float));
    for (uint32_t i = 0; i < outN; ++i)
        printf("logits[%u] = %f\n", i, data[i]);
}

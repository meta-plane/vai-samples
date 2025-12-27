#include "./core/neuralNodes.h"
#include "./core/jsonParser.h"
#include "./core/timeChecker.hpp"
#include "./core/safeTensorsParser.h"
#include "./utils/utils.h"
#include <stb/stb_image.h>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <string>
#include <random>


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


void test()
{
    printf("============================================================\n");
    printf("    MobileNetV2 Full Classification (17 IRB blocks)\n");
    printf("============================================================\n");

    const char* weightsPath = PROJECT_CURRENT_DIR "/weights/mobilenet_v2_imagenet1k.safetensors";
    const char* imagePath = PROJECT_CURRENT_DIR "/images/shark.png";

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
    printf("\nPreprocessed to 224x224\n");

    printf("\nLoading weights...\n");
    SafeTensorsParser weights(weightsPath);
    printf("  Loaded %zu tensors\n", weights.getTensorNames().size());

    printf("\nBuilding full MobileNetV2...\n");
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
    FullyConnectedNode fc(1280, 1000);
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

    
    printf("\nRunning inference...\n");

    Tensor inputTensor(224, 224, 3); // [H, W, C]
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
    for (int rank = 0; rank < 5; ++rank) {
        int classIdx = top5[rank];
        float confidence = probs[classIdx];
        printf("   #%d: Class %d - Confidence: %.4f%%\n", rank + 1, classIdx, confidence * 100.0f);
	}

    printf("\n============================================================\n");
}

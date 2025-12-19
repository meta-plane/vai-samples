#include "mobilenetv2.h"
#include "../utils/utils.h"
#include <cstdio>
#include <cstring>

static void loadIRBWeightsWithExpansion(
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
    std::vector<float> dw_w_t(9 * hiddenDim);
    for (int k = 0; k < 9; ++k) {
        for (int c = 0; c < hiddenDim; ++c) {
            dw_w_t[k * hiddenDim + c] = dw_w[c * 9 + k];
        }
    }
    Tensor t_dw_w(9, hiddenDim);  // Shape: [K*K, C] = [9, hiddenDim]
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

static void loadIRBWeightsNoExpansion(
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

MobileNetV2::MobileNetV2(Device& device, uint32_t numClasses)
    : NeuralNet(device, 1, 1),
      conv0(3, 32, 3, 2, 1),
      irb1(32, 16, 1, 1),
      irb2(16, 24, 6, 2),
      irb3(24, 24, 6, 1),
      irb4(24, 32, 6, 2),
      irb5(32, 32, 6, 1),
      irb6(32, 32, 6, 1),
      irb7(32, 64, 6, 2),
      irb8(64, 64, 6, 1),
      irb9(64, 64, 6, 1),
      irb10(64, 64, 6, 1),
      irb11(64, 96, 6, 1),
      irb12(96, 96, 6, 1),
      irb13(96, 96, 6, 1),
      irb14(96, 160, 6, 2),
      irb15(160, 160, 6, 1),
      irb16(160, 160, 6, 1),
      irb17(160, 320, 6, 1),
      lastConv(320, 1280),
      fc(1280, numClasses)
{
    // Build the network architecture
    input(0)
        - conv0
        - irb1 - irb2 - irb3 - irb4 - irb5 - irb6 - irb7
        - irb8 - irb9 - irb10 - irb11 - irb12 - irb13
        - irb14 - irb15 - irb16 - irb17
        - lastConv - gap - fc
        - output(0);
}

void MobileNetV2::loadWeights(const char* weightsPath)
{
    printf("\nLoading weights from %s...\n", weightsPath);
    SafeTensorsParser weights(weightsPath);
    printf("  Loaded %zu tensors\n", weights.getTensorNames().size());

    printf("\nBuilding full MobileNetV2...\n");

    // === features.0: First Conv ===
    auto w = weights["features.0.0.weight"].parseNDArray();
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

    printf("  features.0: Conv(3->32, s=2) 224->112\n");

    // === features.1: IRB(32->16, t=1, s=1) ===
    loadIRBWeightsNoExpansion(irb1, weights, 1, 32, 16);
    printf("  features.1: IRB(32->16, t=1, s=1)\n");

    // === features.2: IRB(16->24, t=6, s=2) ===
    loadIRBWeightsWithExpansion(irb2, weights, 2, 16, 24, 6);
    printf("  features.2: IRB(16->24, t=6, s=2) 112->56\n");

    // === features.3: IRB(24->24, t=6, s=1) ===
    loadIRBWeightsWithExpansion(irb3, weights, 3, 24, 24, 6);
    printf("  features.3: IRB(24->24, t=6, s=1)\n");

    // === features.4: IRB(24->32, t=6, s=2) ===
    loadIRBWeightsWithExpansion(irb4, weights, 4, 24, 32, 6);
    printf("  features.4: IRB(24->32, t=6, s=2) 56->28\n");

    // === features.5: IRB(32->32, t=6, s=1) ===
    loadIRBWeightsWithExpansion(irb5, weights, 5, 32, 32, 6);
    printf("  features.5: IRB(32->32, t=6, s=1)\n");

    // === features.6: IRB(32->32, t=6, s=1) ===
    loadIRBWeightsWithExpansion(irb6, weights, 6, 32, 32, 6);
    printf("  features.6: IRB(32->32, t=6, s=1)\n");

    // === features.7: IRB(32->64, t=6, s=2) ===
    loadIRBWeightsWithExpansion(irb7, weights, 7, 32, 64, 6);
    printf("  features.7: IRB(32->64, t=6, s=2) 28->14\n");

    // === features.8: IRB(64->64, t=6, s=1) ===
    loadIRBWeightsWithExpansion(irb8, weights, 8, 64, 64, 6);
    printf("  features.8: IRB(64->64, t=6, s=1)\n");

    // === features.9: IRB(64->64, t=6, s=1) ===
    loadIRBWeightsWithExpansion(irb9, weights, 9, 64, 64, 6);
    printf("  features.9: IRB(64->64, t=6, s=1)\n");

    // === features.10: IRB(64->64, t=6, s=1) ===
    loadIRBWeightsWithExpansion(irb10, weights, 10, 64, 64, 6);
    printf("  features.10: IRB(64->64, t=6, s=1)\n");

    // === features.11: IRB(64->96, t=6, s=1) ===
    loadIRBWeightsWithExpansion(irb11, weights, 11, 64, 96, 6);
    printf("  features.11: IRB(64->96, t=6, s=1)\n");

    // === features.12: IRB(96->96, t=6, s=1) ===
    loadIRBWeightsWithExpansion(irb12, weights, 12, 96, 96, 6);
    printf("  features.12: IRB(96->96, t=6, s=1)\n");

    // === features.13: IRB(96->96, t=6, s=1) ===
    loadIRBWeightsWithExpansion(irb13, weights, 13, 96, 96, 6);
    printf("  features.13: IRB(96->96, t=6, s=1)\n");

    // === features.14: IRB(96->160, t=6, s=2) ===
    loadIRBWeightsWithExpansion(irb14, weights, 14, 96, 160, 6);
    printf("  features.14: IRB(96->160, t=6, s=2) 14->7\n");

    // === features.15: IRB(160->160, t=6, s=1) ===
    loadIRBWeightsWithExpansion(irb15, weights, 15, 160, 160, 6);
    printf("  features.15: IRB(160->160, t=6, s=1)\n");

    // === features.16: IRB(160->160, t=6, s=1) ===
    loadIRBWeightsWithExpansion(irb16, weights, 16, 160, 160, 6);
    printf("  features.16: IRB(160->160, t=6, s=1)\n");

    // === features.17: IRB(160->320, t=6, s=1) ===
    loadIRBWeightsWithExpansion(irb17, weights, 17, 160, 320, 6);
    printf("  features.17: IRB(160->320, t=6, s=1)\n");

    // === features.18: Last Conv (320->1280) ===
    {
        auto w = weights["features.18.0.weight"].parseNDArray();
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
    printf("  GAP: 7x7x1280 -> 1280\n");

    // === Classifier ===
    {
        auto w = weights["classifier.1.weight"].parseNDArray();
        auto b = weights["classifier.1.bias"].parseNDArray();
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
}


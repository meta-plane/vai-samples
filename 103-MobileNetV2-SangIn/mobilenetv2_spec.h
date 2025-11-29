#pragma once
#include "neuralNet.h"

/*
 * MobileNetV2 Specification
 * 
 * Maps PyTorch model keys to internal VAI neural network specification.
 * Each ConvBnSpec describes:
 *   - JSON keys for weight/bias and BN parameters (mean, var, eps)
 *   - Internal node names and format flags for tensor transpose if needed
 *   - Convolution output/input channels and kernel size
 * 
 * Structure:
 *   1. Stem: Conv 3x3 (3->32)
 *   2. 17 Bottleneck blocks with expansion/depthwise/projection
 *   3. Final expansion: Conv 1x1 (320->1280)
 *   4. Classifier: Linear (1280->1000)
 * 
 * Notes:
 *   - Depthwise convolutions: IC == OC (groups=IC in PyTorch; no transpose)
 *   - transpose_OC_ICk: indicates if weight tensor needs transpose
 *     (PyTorch Conv2d: [OC, IC, H, W]; VAI may use [IC*H*W, OC] internally for GEMM)
 *   - BN params: running_mean, running_var, and eps are used for BN folding into weights
 */

struct ConvBnSpec
{
    // PyTorch state_dict keys
    std::string json_conv_w;        // weight tensor
    std::string json_conv_b;        // bias (empty if not present)
    std::string json_bn_w;          // BN weight (gamma)
    std::string json_bn_b;          // BN bias (beta)
    std::string json_bn_rm;         // BN running_mean
    std::string json_bn_rv;         // BN running_var
    std::string json_bn_eps;        // BN eps

    // Internal VAI network names and format
    std::string internal_conv_w;    // internal weight name
    std::string internal_conv_b;    // internal bias name
    bool        transpose_OC_ICk;   // whether to transpose weight for VAI format

    // Convolution shape (for validation/allocation)
    uint32_t OC;                    // output channels
    uint32_t IC;                    // input channels
    uint32_t kH;                    // kernel height
    uint32_t kW;                    // kernel width
};

// Stem: Conv 3x3, 3 in -> 32 out
static const ConvBnSpec g_mobilenetv2_convs[] = {
    {
        // [0] Stem Conv 3x3
        "features.0.0.weight",              "",
        "features.0.1.weight",              "features.0.1.bias",
        "features.0.1.running_mean",        "features.0.1.running_var",
        "features.0.1.eps",
        "stem.weight",                      "stem.bias",
        true,
        32, 3, 3, 3
    },

    // Bottleneck 1: expand 32->32, depthwise 3x3, project 32->16
    {
        // [1] Bottleneck 1 - Expand Conv 1x1 (32->32)
        "features.1.conv.0.0.weight",       "",
        "features.1.conv.0.1.weight",       "features.1.conv.0.1.bias",
        "features.1.conv.0.1.running_mean", "features.1.conv.0.1.running_var",
        "features.1.conv.0.1.eps",
        "bneck1.expand.weight",             "bneck1.expand.bias",
        true,
        32, 32, 1, 1
    },
    {
        // [2] Bottleneck 1 - Depthwise Conv 3x3 (32->32)
        "features.1.conv.1.weight",         "",
        "features.1.conv.1.weight",         "",
        "features.1.conv.1.running_mean",   "features.1.conv.1.running_var",
        "features.1.conv.1.eps",
        "bneck1.dw.weight",                 "bneck1.dw.bias",
        false,
        32, 32, 3, 3
    },
    {
        // [3] Bottleneck 1 - Project Conv 1x1 (32->16)
        "features.1.conv.2.weight",         "",
        "features.1.conv.2.weight",         "",
        "features.1.conv.2.running_mean",   "features.1.conv.2.running_var",
        "features.1.conv.2.eps",
        "bneck1.project.weight",            "bneck1.project.bias",
        true,
        16, 32, 1, 1
    },

    // Bottleneck 2: expand 16->96, depthwise 3x3, project 96->24
    {
        // [4] Bottleneck 2 - Expand Conv 1x1 (16->96)
        "features.2.conv.0.0.weight",       "",
        "features.2.conv.0.1.weight",       "features.2.conv.0.1.bias",
        "features.2.conv.0.1.running_mean", "features.2.conv.0.1.running_var",
        "features.2.conv.0.1.eps",
        "bneck2.expand.weight",             "bneck2.expand.bias",
        true,
        96, 16, 1, 1
    },
    {
        // [5] Bottleneck 2 - Depthwise Conv 3x3 (96->96)
        "features.2.conv.1.0.weight",       "",
        "features.2.conv.1.1.weight",       "features.2.conv.1.1.bias",
        "features.2.conv.1.1.running_mean", "features.2.conv.1.1.running_var",
        "features.2.conv.1.1.eps",
        "bneck2.dw.weight",                 "bneck2.dw.bias",
        false,
        96, 96, 3, 3
    },
    {
        // [6] Bottleneck 2 - Project Conv 1x1 (96->24)
        "features.2.conv.2.weight",         "",
        "features.2.conv.3.weight",         "",
        "features.2.conv.3.running_mean",   "features.2.conv.3.running_var",
        "features.2.conv.3.eps",
        "bneck2.project.weight",            "bneck2.project.bias",
        true,
        24, 96, 1, 1
    },

    // Bottleneck 3: expand 24->144, depthwise 3x3, project 144->24
    {
        // [7] Bottleneck 3 - Expand Conv 1x1 (24->144)
        "features.3.conv.0.0.weight",       "",
        "features.3.conv.0.1.weight",       "features.3.conv.0.1.bias",
        "features.3.conv.0.1.running_mean", "features.3.conv.0.1.running_var",
        "features.3.conv.0.1.eps",
        "bneck3.expand.weight",             "bneck3.expand.bias",
        true,
        144, 24, 1, 1
    },
    {
        // [8] Bottleneck 3 - Depthwise Conv 3x3 (144->144)
        "features.3.conv.1.0.weight",       "",
        "features.3.conv.1.1.weight",       "features.3.conv.1.1.bias",
        "features.3.conv.1.1.running_mean", "features.3.conv.1.1.running_var",
        "features.3.conv.1.1.eps",
        "bneck3.dw.weight",                 "bneck3.dw.bias",
        false,
        144, 144, 3, 3
    },
    {
        // [9] Bottleneck 3 - Project Conv 1x1 (144->24)
        "features.3.conv.2.weight",         "",
        "features.3.conv.3.weight",         "",
        "features.3.conv.3.running_mean",   "features.3.conv.3.running_var",
        "features.3.conv.3.eps",
        "bneck3.project.weight",            "bneck3.project.bias",
        true,
        24, 144, 1, 1
    },

    // Bottleneck 4: expand 24->144, depthwise 3x3, project 144->32
    {
        // [10] Bottleneck 4 - Expand Conv 1x1 (24->144)
        "features.4.conv.0.0.weight",       "",
        "features.4.conv.0.1.weight",       "features.4.conv.0.1.bias",
        "features.4.conv.0.1.running_mean", "features.4.conv.0.1.running_var",
        "features.4.conv.0.1.eps",
        "bneck4.expand.weight",             "bneck4.expand.bias",
        true,
        144, 24, 1, 1
    },
    {
        // [11] Bottleneck 4 - Depthwise Conv 3x3 (144->144)
        "features.4.conv.1.0.weight",       "",
        "features.4.conv.1.1.weight",       "features.4.conv.1.1.bias",
        "features.4.conv.1.1.running_mean", "features.4.conv.1.1.running_var",
        "features.4.conv.1.1.eps",
        "bneck4.dw.weight",                 "bneck4.dw.bias",
        false,
        144, 144, 3, 3
    },
    {
        // [12] Bottleneck 4 - Project Conv 1x1 (144->32)
        "features.4.conv.2.weight",         "",
        "features.4.conv.3.weight",         "",
        "features.4.conv.3.running_mean",   "features.4.conv.3.running_var",
        "features.4.conv.3.eps",
        "bneck4.project.weight",            "bneck4.project.bias",
        true,
        32, 144, 1, 1
    },

    // Bottleneck 5: expand 32->192, depthwise 3x3, project 192->32
    {
        // [13] Bottleneck 5 - Expand Conv 1x1 (32->192)
        "features.5.conv.0.0.weight",       "",
        "features.5.conv.0.1.weight",       "features.5.conv.0.1.bias",
        "features.5.conv.0.1.running_mean", "features.5.conv.0.1.running_var",
        "features.5.conv.0.1.eps",
        "bneck5.expand.weight",             "bneck5.expand.bias",
        true,
        192, 32, 1, 1
    },
    {
        // [14] Bottleneck 5 - Depthwise Conv 3x3 (192->192)
        "features.5.conv.1.0.weight",       "",
        "features.5.conv.1.1.weight",       "features.5.conv.1.1.bias",
        "features.5.conv.1.1.running_mean", "features.5.conv.1.1.running_var",
        "features.5.conv.1.1.eps",
        "bneck5.dw.weight",                 "bneck5.dw.bias",
        false,
        192, 192, 3, 3
    },
    {
        // [15] Bottleneck 5 - Project Conv 1x1 (192->32)
        "features.5.conv.2.weight",         "",
        "features.5.conv.3.weight",         "",
        "features.5.conv.3.running_mean",   "features.5.conv.3.running_var",
        "features.5.conv.3.eps",
        "bneck5.project.weight",            "bneck5.project.bias",
        true,
        32, 192, 1, 1
    },

    // Bottleneck 6: expand 32->192, depthwise 3x3, project 192->32
    {
        // [16] Bottleneck 6 - Expand Conv 1x1 (32->192)
        "features.6.conv.0.0.weight",       "",
        "features.6.conv.0.1.weight",       "features.6.conv.0.1.bias",
        "features.6.conv.0.1.running_mean", "features.6.conv.0.1.running_var",
        "features.6.conv.0.1.eps",
        "bneck6.expand.weight",             "bneck6.expand.bias",
        true,
        192, 32, 1, 1
    },
    {
        // [17] Bottleneck 6 - Depthwise Conv 3x3 (192->192)
        "features.6.conv.1.0.weight",       "",
        "features.6.conv.1.1.weight",       "features.6.conv.1.1.bias",
        "features.6.conv.1.1.running_mean", "features.6.conv.1.1.running_var",
        "features.6.conv.1.1.eps",
        "bneck6.dw.weight",                 "bneck6.dw.bias",
        false,
        192, 192, 3, 3
    },
    {
        // [18] Bottleneck 6 - Project Conv 1x1 (192->32)
        "features.6.conv.2.weight",         "",
        "features.6.conv.3.weight",         "",
        "features.6.conv.3.running_mean",   "features.6.conv.3.running_var",
        "features.6.conv.3.eps",
        "bneck6.project.weight",            "bneck6.project.bias",
        true,
        32, 192, 1, 1
    },

    // Bottleneck 7: expand 32->192, depthwise 3x3, project 192->32
    {
        // [19] Bottleneck 7 - Expand Conv 1x1 (32->192)
        "features.7.conv.0.0.weight",       "",
        "features.7.conv.0.1.weight",       "features.7.conv.0.1.bias",
        "features.7.conv.0.1.running_mean", "features.7.conv.0.1.running_var",
        "features.7.conv.0.1.eps",
        "bneck7.expand.weight",             "bneck7.expand.bias",
        true,
        192, 32, 1, 1
    },
    {
        // [20] Bottleneck 7 - Depthwise Conv 3x3 (192->192)
        "features.7.conv.1.0.weight",       "",
        "features.7.conv.1.1.weight",       "features.7.conv.1.1.bias",
        "features.7.conv.1.1.running_mean", "features.7.conv.1.1.running_var",
        "features.7.conv.1.1.eps",
        "bneck7.dw.weight",                 "bneck7.dw.bias",
        false,
        192, 192, 3, 3
    },
    {
        // [21] Bottleneck 7 - Project Conv 1x1 (192->64)
        "features.7.conv.2.weight",         "",
        "features.7.conv.3.weight",         "",
        "features.7.conv.3.running_mean",   "features.7.conv.3.running_var",
        "features.7.conv.3.eps",
        "bneck7.project.weight",            "bneck7.project.bias",
        true,
        64, 192, 1, 1
    },

    // Bottleneck 8: expand 64->384, depthwise 3x3, project 384->64
    {
        // [22] Bottleneck 8 - Expand Conv 1x1 (64->384)
        "features.8.conv.0.0.weight",       "",
        "features.8.conv.0.1.weight",       "features.8.conv.0.1.bias",
        "features.8.conv.0.1.running_mean", "features.8.conv.0.1.running_var",
        "features.8.conv.0.1.eps",
        "bneck8.expand.weight",             "bneck8.expand.bias",
        true,
        384, 64, 1, 1
    },
    {
        // [23] Bottleneck 8 - Depthwise Conv 3x3 (384->384)
        "features.8.conv.1.0.weight",       "",
        "features.8.conv.1.1.weight",       "features.8.conv.1.1.bias",
        "features.8.conv.1.1.running_mean", "features.8.conv.1.1.running_var",
        "features.8.conv.1.1.eps",
        "bneck8.dw.weight",                 "bneck8.dw.bias",
        false,
        384, 384, 3, 3
    },
    {
        // [24] Bottleneck 8 - Project Conv 1x1 (384->64)
        "features.8.conv.2.weight",         "",
        "features.8.conv.3.weight",         "",
        "features.8.conv.3.running_mean",   "features.8.conv.3.running_var",
        "features.8.conv.3.eps",
        "bneck8.project.weight",            "bneck8.project.bias",
        true,
        64, 384, 1, 1
    },

    // Bottleneck 9: expand 64->384, depthwise 3x3, project 384->64
    {
        // [25] Bottleneck 9 - Expand Conv 1x1 (64->384)
        "features.9.conv.0.0.weight",       "",
        "features.9.conv.0.1.weight",       "features.9.conv.0.1.bias",
        "features.9.conv.0.1.running_mean", "features.9.conv.0.1.running_var",
        "features.9.conv.0.1.eps",
        "bneck9.expand.weight",             "bneck9.expand.bias",
        true,
        384, 64, 1, 1
    },
    {
        // [26] Bottleneck 9 - Depthwise Conv 3x3 (384->384)
        "features.9.conv.1.0.weight",       "",
        "features.9.conv.1.1.weight",       "features.9.conv.1.1.bias",
        "features.9.conv.1.1.running_mean", "features.9.conv.1.1.running_var",
        "features.9.conv.1.1.eps",
        "bneck9.dw.weight",                 "bneck9.dw.bias",
        false,
        384, 384, 3, 3
    },
    {
        // [27] Bottleneck 9 - Project Conv 1x1 (384->64)
        "features.9.conv.2.weight",         "",
        "features.9.conv.3.weight",         "",
        "features.9.conv.3.running_mean",   "features.9.conv.3.running_var",
        "features.9.conv.3.eps",
        "bneck9.project.weight",            "bneck9.project.bias",
        true,
        64, 384, 1, 1
    },

    // Bottleneck 10: expand 64->384, depthwise 3x3, project 384->64
    {
        // [28] Bottleneck 10 - Expand Conv 1x1 (64->384)
        "features.10.conv.0.0.weight",      "",
        "features.10.conv.0.1.weight",      "features.10.conv.0.1.bias",
        "features.10.conv.0.1.running_mean","features.10.conv.0.1.running_var",
        "features.10.conv.0.1.eps",
        "bneck10.expand.weight",            "bneck10.expand.bias",
        true,
        384, 64, 1, 1
    },
    {
        // [29] Bottleneck 10 - Depthwise Conv 3x3 (384->384)
        "features.10.conv.1.0.weight",      "",
        "features.10.conv.1.1.weight",      "features.10.conv.1.1.bias",
        "features.10.conv.1.1.running_mean","features.10.conv.1.1.running_var",
        "features.10.conv.1.1.eps",
        "bneck10.dw.weight",                "bneck10.dw.bias",
        false,
        384, 384, 3, 3
    },
    {
        // [30] Bottleneck 10 - Project Conv 1x1 (384->64)
        "features.10.conv.2.weight",        "",
        "features.10.conv.3.weight",        "",
        "features.10.conv.3.running_mean",  "features.10.conv.3.running_var",
        "features.10.conv.3.eps",
        "bneck10.project.weight",           "bneck10.project.bias",
        true,
        64, 384, 1, 1
    },

    // Bottleneck 11: expand 64->384, depthwise 3x3, project 384->96
    {
        // [31] Bottleneck 11 - Expand Conv 1x1 (64->384)
        "features.11.conv.0.0.weight",      "",
        "features.11.conv.0.1.weight",      "features.11.conv.0.1.bias",
        "features.11.conv.0.1.running_mean","features.11.conv.0.1.running_var",
        "features.11.conv.0.1.eps",
        "bneck11.expand.weight",            "bneck11.expand.bias",
        true,
        384, 64, 1, 1
    },
    {
        // [32] Bottleneck 11 - Depthwise Conv 3x3 (384->384)
        "features.11.conv.1.0.weight",      "",
        "features.11.conv.1.1.weight",      "features.11.conv.1.1.bias",
        "features.11.conv.1.1.running_mean","features.11.conv.1.1.running_var",
        "features.11.conv.1.1.eps",
        "bneck11.dw.weight",                "bneck11.dw.bias",
        false,
        384, 384, 3, 3
    },
    {
        // [33] Bottleneck 11 - Project Conv 1x1 (384->96)
        "features.11.conv.2.weight",        "",
        "features.11.conv.3.weight",        "",
        "features.11.conv.3.running_mean",  "features.11.conv.3.running_var",
        "features.11.conv.3.eps",
        "bneck11.project.weight",           "bneck11.project.bias",
        true,
        96, 384, 1, 1
    },

    // Bottleneck 12: expand 96->576, depthwise 3x3, project 576->96
    {
        // [34] Bottleneck 12 - Expand Conv 1x1 (96->576)
        "features.12.conv.0.0.weight",      "",
        "features.12.conv.0.1.weight",      "features.12.conv.0.1.bias",
        "features.12.conv.0.1.running_mean","features.12.conv.0.1.running_var",
        "features.12.conv.0.1.eps",
        "bneck12.expand.weight",            "bneck12.expand.bias",
        true,
        576, 96, 1, 1
    },
    {
        // [35] Bottleneck 12 - Depthwise Conv 3x3 (576->576)
        "features.12.conv.1.0.weight",      "",
        "features.12.conv.1.1.weight",      "features.12.conv.1.1.bias",
        "features.12.conv.1.1.running_mean","features.12.conv.1.1.running_var",
        "features.12.conv.1.1.eps",
        "bneck12.dw.weight",                "bneck12.dw.bias",
        false,
        576, 576, 3, 3
    },
    {
        // [36] Bottleneck 12 - Project Conv 1x1 (576->96)
        "features.12.conv.2.weight",        "",
        "features.12.conv.3.weight",        "",
        "features.12.conv.3.running_mean",  "features.12.conv.3.running_var",
        "features.12.conv.3.eps",
        "bneck12.project.weight",           "bneck12.project.bias",
        true,
        96, 576, 1, 1
    },

    // Bottleneck 13: expand 96->576, depthwise 3x3, project 576->96
    {
        // [37] Bottleneck 13 - Expand Conv 1x1 (96->576)
        "features.13.conv.0.0.weight",      "",
        "features.13.conv.0.1.weight",      "features.13.conv.0.1.bias",
        "features.13.conv.0.1.running_mean","features.13.conv.0.1.running_var",
        "features.13.conv.0.1.eps",
        "bneck13.expand.weight",            "bneck13.expand.bias",
        true,
        576, 96, 1, 1
    },
    {
        // [38] Bottleneck 13 - Depthwise Conv 3x3 (576->576)
        "features.13.conv.1.0.weight",      "",
        "features.13.conv.1.1.weight",      "features.13.conv.1.1.bias",
        "features.13.conv.1.1.running_mean","features.13.conv.1.1.running_var",
        "features.13.conv.1.1.eps",
        "bneck13.dw.weight",                "bneck13.dw.bias",
        false,
        576, 576, 3, 3
    },
    {
        // [39] Bottleneck 13 - Project Conv 1x1 (576->96)
        "features.13.conv.2.weight",        "",
        "features.13.conv.3.weight",        "",
        "features.13.conv.3.running_mean",  "features.13.conv.3.running_var",
        "features.13.conv.3.eps",
        "bneck13.project.weight",           "bneck13.project.bias",
        true,
        96, 576, 1, 1
    },

    // Bottleneck 14: expand 96->576, depthwise 3x3, project 576->160
    {
        // [40] Bottleneck 14 - Expand Conv 1x1 (96->576)
        "features.14.conv.0.0.weight",      "",
        "features.14.conv.0.1.weight",      "features.14.conv.0.1.bias",
        "features.14.conv.0.1.running_mean","features.14.conv.0.1.running_var",
        "features.14.conv.0.1.eps",
        "bneck14.expand.weight",            "bneck14.expand.bias",
        true,
        576, 96, 1, 1
    },
    {
        // [41] Bottleneck 14 - Depthwise Conv 3x3 (576->576)
        "features.14.conv.1.0.weight",      "",
        "features.14.conv.1.1.weight",      "features.14.conv.1.1.bias",
        "features.14.conv.1.1.running_mean","features.14.conv.1.1.running_var",
        "features.14.conv.1.1.eps",
        "bneck14.dw.weight",                "bneck14.dw.bias",
        false,
        576, 576, 3, 3
    },
    {
        // [42] Bottleneck 14 - Project Conv 1x1 (576->160)
        "features.14.conv.2.weight",        "",
        "features.14.conv.3.weight",        "",
        "features.14.conv.3.running_mean",  "features.14.conv.3.running_var",
        "features.14.conv.3.eps",
        "bneck14.project.weight",           "bneck14.project.bias",
        true,
        160, 576, 1, 1
    },

    // Bottleneck 15: expand 160->960, depthwise 3x3, project 960->160
    {
        // [43] Bottleneck 15 - Expand Conv 1x1 (160->960)
        "features.15.conv.0.0.weight",      "",
        "features.15.conv.0.1.weight",      "features.15.conv.0.1.bias",
        "features.15.conv.0.1.running_mean","features.15.conv.0.1.running_var",
        "features.15.conv.0.1.eps",
        "bneck15.expand.weight",            "bneck15.expand.bias",
        true,
        960, 160, 1, 1
    },
    {
        // [44] Bottleneck 15 - Depthwise Conv 3x3 (960->960)
        "features.15.conv.1.0.weight",      "",
        "features.15.conv.1.1.weight",      "features.15.conv.1.1.bias",
        "features.15.conv.1.1.running_mean","features.15.conv.1.1.running_var",
        "features.15.conv.1.1.eps",
        "bneck15.dw.weight",                "bneck15.dw.bias",
        false,
        960, 960, 3, 3
    },
    {
        // [45] Bottleneck 15 - Project Conv 1x1 (960->160)
        "features.15.conv.2.weight",        "",
        "features.15.conv.3.weight",        "",
        "features.15.conv.3.running_mean",  "features.15.conv.3.running_var",
        "features.15.conv.3.eps",
        "bneck15.project.weight",           "bneck15.project.bias",
        true,
        160, 960, 1, 1
    },

    // Bottleneck 16: expand 160->960, depthwise 3x3, project 960->160
    {
        // [46] Bottleneck 16 - Expand Conv 1x1 (160->960)
        "features.16.conv.0.0.weight",      "",
        "features.16.conv.0.1.weight",      "features.16.conv.0.1.bias",
        "features.16.conv.0.1.running_mean","features.16.conv.0.1.running_var",
        "features.16.conv.0.1.eps",
        "bneck16.expand.weight",            "bneck16.expand.bias",
        true,
        960, 160, 1, 1
    },
    {
        // [47] Bottleneck 16 - Depthwise Conv 3x3 (960->960)
        "features.16.conv.1.0.weight",      "",
        "features.16.conv.1.1.weight",      "features.16.conv.1.1.bias",
        "features.16.conv.1.1.running_mean","features.16.conv.1.1.running_var",
        "features.16.conv.1.1.eps",
        "bneck16.dw.weight",                "bneck16.dw.bias",
        false,
        960, 960, 3, 3
    },
    {
        // [48] Bottleneck 16 - Project Conv 1x1 (960->160)
        "features.16.conv.2.weight",        "",
        "features.16.conv.3.weight",        "",
        "features.16.conv.3.running_mean",  "features.16.conv.3.running_var",
        "features.16.conv.3.eps",
        "bneck16.project.weight",           "bneck16.project.bias",
        true,
        160, 960, 1, 1
    },

    // Bottleneck 17: expand 160->960, depthwise 3x3, project 960->160
    {
        // [49] Bottleneck 17 - Expand Conv 1x1 (160->960)
        "features.17.conv.0.0.weight",      "",
        "features.17.conv.0.1.weight",      "features.17.conv.0.1.bias",
        "features.17.conv.0.1.running_mean","features.17.conv.0.1.running_var",
        "features.17.conv.0.1.eps",
        "bneck17.expand.weight",            "bneck17.expand.bias",
        true,
        960, 160, 1, 1
    },
    {
        // [50] Bottleneck 17 - Depthwise Conv 3x3 (960->960)
        "features.17.conv.1.0.weight",      "",
        "features.17.conv.1.1.weight",      "features.17.conv.1.1.bias",
        "features.17.conv.1.1.running_mean","features.17.conv.1.1.running_var",
        "features.17.conv.1.1.eps",
        "bneck17.dw.weight",                "bneck17.dw.bias",
        false,
        960, 960, 3, 3
    },
    {
        // [51] Bottleneck 17 - Project Conv 1x1 (960->320)
        "features.17.conv.2.weight",        "",
        "features.17.conv.3.weight",        "",
        "features.17.conv.3.running_mean",  "features.17.conv.3.running_var",
        "features.17.conv.3.eps",
        "bneck17.project.weight",           "bneck17.project.bias",
        true,
        320, 960, 1, 1
    },

    // Final expansion layer: Conv 1x1, 320->1280
    {
        // [52] Final Expand Conv 1x1 (320->1280)
        "features.18.0.weight",             "",
        "features.18.1.weight",             "features.18.1.bias",
        "features.18.1.running_mean",       "features.18.1.running_var",
        "features.18.1.eps",
        "final_expand.weight",              "final_expand.bias",
        true,
        1280, 320, 1, 1
    },

    // Classifier layer (note: this is a fully connected layer, not a conv)
    // Typically handled separately from ConvBnSpec
    {
        // [53] Classifier FC layer (1280->1000)
        // Note: This might be better handled separately as it's not a Conv+BN
        "classifier.1.weight",              "",
        "",                                 "",
        "",                                 "",
        "",
        "classifier.weight",                "classifier.bias",
        false,
        1000, 1280, 1, 1
    }
};

// Helper: get total number of layers
static constexpr uint32_t g_mobilenetv2_num_convs = sizeof(g_mobilenetv2_convs) / sizeof(g_mobilenetv2_convs[0]);

#endif // MOBILENETV2_SPEC_H

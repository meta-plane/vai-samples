#include "efficientNet.h"
#include <cmath>
#include <algorithm>

// EfficientNet 버전별 설정 반환
EfficientNetConfig getEfficientNetConfig(EfficientNetVersion version)
{
    switch (version)
    {
        case EfficientNetVersion::B0: return {1.0f, 1.0f, 224};
        case EfficientNetVersion::B1: return {1.1f, 1.0f, 240};
        case EfficientNetVersion::B2: return {1.2f, 1.1f, 260};
        case EfficientNetVersion::B3: return {1.4f, 1.2f, 300};
        case EfficientNetVersion::B4: return {1.8f, 1.4f, 380};
        case EfficientNetVersion::B5: return {2.0f, 1.6f, 456};
        case EfficientNetVersion::B6: return {2.2f, 1.8f, 528};
        case EfficientNetVersion::B7: return {2.6f, 2.0f, 600};
        default: return {1.0f, 1.0f, 224};
    }
}

// B0 기본 구조 (multiplier 적용 전)
static std::vector<MBConvConfig> getB0BaseConfig()
{
    return {
        {32, 16, 1, 3, 1, 0.25f},       // Block 1
        {16, 24, 6, 3, 2, 0.25f},       // Block 2
        {24, 24, 6, 3, 1, 0.25f},       // Block 3
        {24, 40, 6, 5, 2, 0.25f},       // Block 4
        {40, 40, 6, 5, 1, 0.25f},       // Block 5
        {40, 80, 6, 3, 2, 0.25f},       // Block 6
        {80, 80, 6, 3, 1, 0.25f},       // Block 7
        {80, 112, 6, 5, 1, 0.25f},      // Block 8
        {112, 112, 6, 5, 1, 0.25f},     // Block 9
        {112, 192, 6, 5, 2, 0.25f},     // Block 10
        {192, 192, 6, 5, 1, 0.25f},     // Block 11
        {192, 320, 6, 3, 1, 0.25f},     // Block 12
    };
}

// 각 블록의 반복 횟수 (B0 기준)
static std::vector<uint32_t> getBlockRepeatCounts()
{
    return {1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7};
}

// 채널 수를 width multiplier로 스케일링 (8의 배수로 반올림)
static uint32_t scaleChannels(uint32_t channels, float width_mult)
{
    return static_cast<uint32_t>(std::round(channels * width_mult / 8.0f)) * 8;
}

// 버전별 config 생성
std::vector<MBConvConfig> EfficientNet::generateConfig(EfficientNetVersion version)
{
    auto config = getEfficientNetConfig(version);
    auto baseConfig = getB0BaseConfig();
    auto repeatCounts = getBlockRepeatCounts();
    
    std::vector<MBConvConfig> result;
    
    for (size_t i = 0; i < baseConfig.size(); ++i)
    {
        const auto& base = baseConfig[i];
        uint32_t repeats = static_cast<uint32_t>(std::round(repeatCounts[i] * config.depth_multiplier));
        
        bool isFirst = true;
        for (uint32_t r = 0; r < repeats; ++r)
        {
            MBConvConfig scaled;
            scaled.in_channels = scaleChannels(base.in_channels, config.width_multiplier);
            scaled.out_channels = scaleChannels(base.out_channels, config.width_multiplier);
            scaled.expand_ratio = base.expand_ratio;
            scaled.kernel_size = base.kernel_size;
            scaled.stride = (isFirst) ? base.stride : 1;
            scaled.se_ratio = base.se_ratio;
            
            result.push_back(scaled);
            isFirst = false;
        }
    }
    
    // 블록 간 채널 연결 수정
    for (size_t i = 1; i < result.size(); ++i)
    {
        result[i].in_channels = result[i-1].out_channels;
    }
    
    return result;
}

EfficientNet::EfficientNet(Device& device, EfficientNetVersion version, uint32_t numClasses)
: EfficientNet(device, 
               generateConfig(version), 
               numClasses,
               scaleChannels(32, getEfficientNetConfig(version).width_multiplier),
               scaleChannels(1280, getEfficientNetConfig(version).width_multiplier))
{
}

EfficientNet::EfficientNet(Device& device, const std::vector<MBConvConfig>& blockConfigs, uint32_t numClasses, uint32_t stemOutChannels, uint32_t headOutChannels)
: NeuralNet(device, 1, 1)
, stem(3, stemOutChannels, 3)
, headConv(blockConfigs.empty() ? stemOutChannels : blockConfigs.back().out_channels, headOutChannels, 1)
, globalAvgPool()
, flatten()
, classifier(headOutChannels, numClasses)
{
    // 첫 번째 블록의 in_channels를 Stem 출력과 일치시킴
    std::vector<MBConvConfig> adjustedConfigs = blockConfigs;
    if (!adjustedConfigs.empty())
    {
        adjustedConfigs[0].in_channels = stemOutChannels;
    }
    
    // Connect input node to stem
    input(0).slot("out0") - stem.slot("in0");
    NodeSlot* lastOutput = &stem.slot("out0");

    // Add MBConv blocks
    for (const auto& config : adjustedConfigs)
    {
        auto block = std::make_unique<MBConvBlockNode>(config);
        *lastOutput - block->slot("in0");
        lastOutput = &block->slot("out0");
        mbconvBlocks.push_back(std::move(block));
    }

    // Connect Head
    *lastOutput - headConv.slot("in0");
    lastOutput = &headConv.slot("out0");
    
    *lastOutput - globalAvgPool.slot("in0");
    lastOutput = &globalAvgPool.slot("out0");
    
    *lastOutput - flatten.slot("in0");
    lastOutput = &flatten.slot("out0");

    *lastOutput - classifier.slot("in0");
    
    // Connect output node
    classifier.slot("out0") - output(0).slot("in0");
}

Tensor& EfficientNet::operator[](const std::string& name)
{
    if (name.starts_with("stem."))
        return stem[name.substr(5)];
    
    if (name.starts_with("blocks."))
    {
        // Format: blocks.0.expand.weight
        size_t firstDot = name.find('.', 7);
        if (firstDot != std::string::npos)
        {
            int blockIdx = std::stoi(name.substr(7, firstDot - 7));
            if (blockIdx < mbconvBlocks.size())
                return (*mbconvBlocks[blockIdx])[name.substr(firstDot + 1)];
        }
    }
    
    if (name.starts_with("head."))
        return headConv[name.substr(5)];
        
    if (name.starts_with("classifier."))
        return classifier[name.substr(11)];
        
    throw std::runtime_error("Weight not found: " + name);
}

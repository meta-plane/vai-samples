#ifndef EFFICIENTNET_H
#define EFFICIENTNET_H

#include "../library/neuralNet.h"
#include "../library/neuralNodes.h"
#include <vector>
#include <memory>


enum class EfficientNetVersion
{
    B0, B1, B2, B3, B4, B5, B6, B7
};


struct EfficientNetConfig
{
    float depth_multiplier;   // d: 블록 반복 횟수 조절
    float width_multiplier;    // w: 채널 수 조절
    uint32_t resolution;        // r: 입력 이미지 해상도
};


// EfficientNet 버전별 설정 반환
EfficientNetConfig getEfficientNetConfig(EfficientNetVersion version);


class EfficientNet : public NeuralNet
{
    // Stem layer
    ConvBNSwishNode stem;
    
    // MBConv blocks
    std::vector<std::unique_ptr<MBConvBlockNode>> mbconvBlocks;
    
    // Head layers
    ConvBNSwishNode headConv; // 1x1 Conv to expand features before pooling
    GlobalAvgPoolNode globalAvgPool;
    FlattenNode flatten;
    FullyConnectedNode classifier;

public:
    // 직접 config를 받는 생성자 (내부용, stemOut과 headOut 채널 지정)
    EfficientNet(Device& device, const std::vector<MBConvConfig>& blockConfigs, uint32_t numClasses, uint32_t stemOutChannels, uint32_t headOutChannels);
    
    // 버전을 받아서 자동으로 config를 생성하는 생성자
    EfficientNet(Device& device, EfficientNetVersion version, uint32_t numClasses = 1000);
    
    Tensor& operator[](const std::string& name);
    
    // 입력 텐서 접근
    Tensor& getInputTensor() { return stem.slot("in0").getValueRef(); }
    
    // 출력 텐서 접근
    Tensor& getOutputTensor() { return output(0).slot("out0").getValueRef(); }
    
    // 버전별 config 생성 헬퍼 함수
    static std::vector<MBConvConfig> generateConfig(EfficientNetVersion version);
};

#endif // EFFICIENTNET_H


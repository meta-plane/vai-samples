#ifndef MOBILENETV2_H 
#define MOBILENETV2_H // 중복 선언 방지

#include "../library/neuralNet.h"
#include "../library/neuralNodes.h"
#include <vector>
#include <memory>


class MobileNetV2 : public NeuralNet
{
    std::unique_ptr<ConvBN> initialConv;                                        // Initial Conv-BN layer
    std::vector<std::unique_ptr<InvertedResidualBlock>> invertedResidualBlocks; // Inverted Residual Blocks
    std::unique_ptr<GlobalAvgPoolNode> globalAvgPool;                           // Global Average Pooling layer
    std::unique_ptr<FullyConnectedNode> fc;                                     // Fully Connected layer for classification

public:
    MobileNetV2(Device& device, uint32_t numClasses = 1000);

    Tensor& operator[](const std::string& name) override;
    Tensor& getInputTensor();
    Tensor& getOutputTensor();
};


#endif // MOBILENETV2_H
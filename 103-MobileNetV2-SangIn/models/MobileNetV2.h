#ifndef MOBILENETV2_H 
#define MOBILENETV2_H // 중복 선언 방지

#include "../library/neuralNet.h"
#include "../library/neuralNodes.h"
#include <vector>
#include <memory>


// Inverted Residual Blocks configuration
struct IRBConfig {
    uint32_t outChannels;
    uint32_t expansionFactor;
    uint32_t stride;
};

class MobileNetV2 : public NeuralNet
{
    // Stem layer
    ConvBNReLU6 stem;

    // Inverted Residual Blocks
    std::vector<std::unique_ptr<InvertedResidualBlock>> invertedResidualBlocks;

    // Final layers
    PWConvBNReLU6 finalConv;
    GlobalAvgPoolNode globalAvgPool;
    FlattenNode flatten;
    FullyConnectedNode classifier;
    SoftmaxNode softmax;

public:
    MobileNetV2(Device& device, uint32_t numClasses = 1000);

    Tensor& operator[](const std::string& name);
    Tensor& getInputTensor() { return stem.slot("in0").getValueRef(); };
    Tensor& getOutputTensor() { return output(0).slot("out0").getValueRef(); }

    void setNodeNameFromParam(const std::string& cppName); // Set node names based on parameter names

    // Access to Inverted Residual Blocks for weight loading
    std::vector<std::unique_ptr<InvertedResidualBlock>>& blocks() {
        return invertedResidualBlocks;
    }

    const std::vector<std::unique_ptr<InvertedResidualBlock>>& blocks() const {
        return invertedResidualBlocks;
    }
};

#endif // MOBILENETV2_H
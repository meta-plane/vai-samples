#pragma once

#include "../core/neuralNet.h"
#include "../core/neuralNodes.h"
#include "../core/safeTensorsParser.h"

class MobileNetV2 : public NeuralNet
{
    // Network layers
    ConvBnRelu6Group conv0;
    InvertedResidualBlock irb1, irb2, irb3, irb4, irb5, irb6, irb7;
    InvertedResidualBlock irb8, irb9, irb10, irb11, irb12, irb13;
    InvertedResidualBlock irb14, irb15, irb16, irb17;
    PwConvBnRelu6Group lastConv;
    GlobalAvgPoolNode gap;
    FullyConnectedNode fc;

public:
    MobileNetV2(Device& device, uint32_t numClasses = 1000);

    // Load pre-trained weights from safetensors file
    void loadWeights(const char* weightsPath);

    // Helper methods to access input/output tensors
    Tensor& getInputTensor() { return input(0).slot("in0").getValueRef(); }
    Tensor& getOutputTensor() { return output(0).slot("out0").getValueRef(); }
};

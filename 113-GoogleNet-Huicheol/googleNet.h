#ifndef GOOGLENET_H
#define GOOGLENET_H

#include "neuralNet.h"
#include "neuralNodes.h"
#include "safeTensorsParser.h"
#include <vector>
#include <memory>
#include <string>

class JsonParser;
class SafeTensorsParser;

// Concat Node  
class ConcatenationNode : public Node
{
    uint32_t numInputs;
    ComputePipeline concat;
    DescriptorSet concatDescSet;

public:
    ConcatenationNode(uint32_t numInputs);
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

// Inception Block Node
class InceptionBlockNode : public NodeGroup
{
    // Fan-out for input so all branches receive the same tensor
    std::unique_ptr<IdentityNode> inputFan;

    uint32_t inChannels;
    uint32_t ch1x1Out;
    uint32_t ch3x3redOut;
    uint32_t ch3x3Out;
    uint32_t ch5x5redOut;
    uint32_t ch5x5Out;
    uint32_t poolProjOut;

    // 1x1 branch
    std::unique_ptr<ConvolutionNode> conv1x1;
    std::unique_ptr<ReluNode> relu1x1;

    // 3x3 branch
    std::unique_ptr<ConvolutionNode> conv3x3_reduce;
    std::unique_ptr<ReluNode> relu3x3_reduce;
    std::unique_ptr<ConvolutionNode> conv3x3;
    std::unique_ptr<ReluNode> relu3x3;

    // 5x5 branch
    std::unique_ptr<ConvolutionNode> conv5x5_reduce;
    std::unique_ptr<ReluNode> relu5x5_reduce;
    std::unique_ptr<ConvolutionNode> conv5x5;
    std::unique_ptr<ReluNode> relu5x5;

    // Pooling branch
    std::unique_ptr<MaxPoolingNode> pool;
    std::unique_ptr<ConvolutionNode> pool_proj;
    std::unique_ptr<ReluNode> relu_pool;

    // Concatenation
    std::unique_ptr<ConcatenationNode> concat;

public:
    InceptionBlockNode(uint32_t inChannels, uint32_t ch1x1, uint32_t ch3x3red, uint32_t ch3x3, uint32_t ch5x5red, uint32_t ch5x5, uint32_t poolProj);
    
    Tensor& operator[](const std::string& name);
    void loadWeights(const JsonParser* json, const SafeTensorsParser* safetensors, const std::string& prefix);
};

class GoogleNet : public NeuralNet
{
    uint32_t numClasses;
    // Initial layers
    ConvolutionNode conv1;
    ReluNode relu1;
    MaxPoolingNode pool1;
    ConvolutionNode conv2_reduce;
    ReluNode relu2_reduce;
    ConvolutionNode conv2;
    ReluNode relu2;
    MaxPoolingNode pool2;

    // Inception blocks
    std::unique_ptr<InceptionBlockNode> inception3a;
    std::unique_ptr<InceptionBlockNode> inception3b;
    MaxPoolingNode pool3;

    std::unique_ptr<InceptionBlockNode> inception4a;
    std::unique_ptr<InceptionBlockNode> inception4b;
    std::unique_ptr<InceptionBlockNode> inception4c;
    std::unique_ptr<InceptionBlockNode> inception4d;
    std::unique_ptr<InceptionBlockNode> inception4e;
    MaxPoolingNode pool4;

    std::unique_ptr<InceptionBlockNode> inception5a;
    std::unique_ptr<InceptionBlockNode> inception5b;

    // Output layers
    GlobalAvgPoolNode avgPool;
    FlattenNode flatten; // >>>>> Optional if GlobalAvgPool returns 1x1xC <<<< 
    FullyConnectedNode fc;

public:
    GoogleNet(Device& device, uint32_t numClasses = 1000);
    
    Tensor& operator[](const std::string& name);
    void loadWeights(const JsonParser* json = nullptr, const SafeTensorsParser* safetensors = nullptr);
    void loadWeights(const SafeTensorsParser* safetensors);
};

#endif // GOOGLENET_H

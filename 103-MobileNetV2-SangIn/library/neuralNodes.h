#ifndef NEURAL_NODES_H
#define NEURAL_NODES_H


#include "neuralNet.h"


class ConvolutionNode : public Node
{
    uint32_t C; // C: input channels
    uint32_t F; // F: output channels
    uint32_t K; // K: kernel size
    uint32_t S; // stride
    uint32_t P; // padding
    uint32_t gemmTileSize;

    ComputePipeline im2col;
    ComputePipeline gemm;

    DescriptorSet im2colDescSet;
    DescriptorSet gemmDescSet;
    
public:
    ConvolutionNode(uint32_t inChannels,
                    uint32_t outChannels,
                    uint32_t kernelWidth,
                    uint32_t stride = 1,
                    uint32_t padding = 0);

    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

class DepthwiseConvNode : public Node
{
    uint32_t C, K, S;   // C: input/output channels, K: kernel width, S: Stride

    ComputePipeline depthwiseConv; // im2col, gemm 등 연산을 pipeline에 한 번에 구현
    DescriptorSet depthwiseConvDescSet;

public:
    DepthwiseConvNode(uint32_t channels, uint32_t kernelWidth, uint32_t stride);
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

class PointwiseConvNode : public Node
{
    uint32_t C, F;   // C: input channels, F: output channels (Num of filters)

    ComputePipeline gemm;
    DescriptorSet gemmDescSet;
    uint32_t gemmTileSize;

    public:
    PointwiseConvNode(uint32_t inChannels, uint32_t outChannels);
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

class AddNode : public Node
{
    ComputePipeline add;
    DescriptorSet addDescSet;

public:
    AddNode();
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

class BatchNormNode : public Node
{
    ComputePipeline batchNorm;
    DescriptorSet batchNormDescSet;
    float eps;

public:
    BatchNormNode(float epsilon = 1e-5f);
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

class ReluNode : public Node
{
    ComputePipeline relu;
    DescriptorSet reluDescSet;

public:
    ReluNode();
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

class Relu6Node : public Node
{
    ComputePipeline relu6;
    DescriptorSet relu6DescSet;

public:
    Relu6Node();
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

class MaxPoolingNode : public Node
{
    const bool discardTail = true; // If true, discard the tail elements that don't fit into the pooling window
    uint32_t P;

    ComputePipeline maxpool;
    DescriptorSet maxpoolDescSet;

public:
    MaxPoolingNode(uint32_t poolSize);
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

class GlobalAvgPoolNode : public Node
{
    ComputePipeline globalAvgPool;
    DescriptorSet globalAvgPoolDescSet;

public:
    GlobalAvgPoolNode();
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


class FlattenNode : public Node
{
    ComputePipeline copy;
    DescriptorSet copyDescSet;

public:
    FlattenNode();
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

class FullyConnectedNode : public Node
{
    uint32_t I, O; // I: input size, O: output size
    ComputePipeline gemm;
    DescriptorSet gemmDescSet;
    uint32_t gemmTileSize;

    ComputePipeline setZero;
    DescriptorSet setZeroDescSet;

public:
    FullyConnectedNode(uint32_t inDim, uint32_t outDim);
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

// Node Groups
class ConvBNReLU6 : public NodeGroup
{
    std::unique_ptr<ConvolutionNode> conv;
    std::unique_ptr<BatchNormNode>   bn;
    std::unique_ptr<Relu6Node>       relu;

public:
    // kernel, stride, padding 모두 받도록
    ConvBNReLU6(uint32_t inChannels,
                uint32_t outChannels,
                uint32_t kernel,
                uint32_t stride = 1,
                uint32_t padding = 0);

    Tensor& operator[](const std::string& slotName);
};

class PWConvBNReLU6 : public NodeGroup
{
    std::unique_ptr<PointwiseConvNode> pointwiseConv;
    std::unique_ptr<BatchNormNode>     bn;
    std::unique_ptr<Relu6Node>         relu;

public:
    PWConvBNReLU6(uint32_t inChannels,
                  uint32_t outChannels);
    
    Tensor& operator[](const std::string& slotName);
};

class PWConvBN : public NodeGroup
{
    std::unique_ptr<PointwiseConvNode> pointwiseConv;
    std::unique_ptr<BatchNormNode>     bn;

public:
    PWConvBN(uint32_t inChannels,
              uint32_t outChannels);
    
    Tensor& operator[](const std::string& slotName);
};

class DWConvBNReLU6 : public NodeGroup
{
    std::unique_ptr<DepthwiseConvNode> depthwiseConv;
    std::unique_ptr<BatchNormNode>     bn;
    std::unique_ptr<Relu6Node>         relu;

public:
    DWConvBNReLU6(uint32_t channels,
                  uint32_t kernel,
                  uint32_t stride = 1,
                  uint32_t padding = 0);

    Tensor& operator[](const std::string& slotName);
};

class InvertedResidualBlock : public NodeGroup
{
    std::unique_ptr<PWConvBNReLU6>     pwConvBNReLU6;  // expansion
    std::unique_ptr<DWConvBNReLU6>     dwConvBNReLU6;  // depthwise conv
    std::unique_ptr<PWConvBN>          pwConvBN;       // projection
    std::unique_ptr<AddNode>           add;
    std::unique_ptr<InputNode>         inputSplit;     // optional: skip + conv 입력 분기

    bool useResidual = false;

public:
    InvertedResidualBlock(uint32_t inChannels,
                          uint32_t outChannels,
                          uint32_t expansionFactor,
                          uint32_t stride);

    Tensor& operator[](const std::string& slotName);
};

extern Device netGlobalDevice; // Global device for neural network operations

#endif // NEURAL_NODES_H
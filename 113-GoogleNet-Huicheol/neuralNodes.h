#ifndef NEURAL_NODES_H
#define NEURAL_NODES_H


#include "neuralNet.h"


class ConvolutionNode : public Node
{
    uint32_t C, F, K;   // C: input channels, F: output channels, K: kernel width

    ComputePipeline im2col;
    ComputePipeline gemm;
    DescriptorSet im2colDescSet;
    DescriptorSet gemmDescSet;
    uint32_t gemmTileSize;

public:
    ConvolutionNode(uint32_t inChannels, uint32_t outChannels, uint32_t kernelWidth);
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


class MaxPoolingNode : public Node
{
    const bool discardTail = true; 
    uint32_t P;
    uint32_t S;
    uint32_t padding;

    ComputePipeline maxpool;
    DescriptorSet maxpoolDescSet;

public:
    MaxPoolingNode(uint32_t poolSize, uint32_t stride = 0, uint32_t padding = 0);
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


class GlobalAvgPoolNode : public Node
{
    ComputePipeline avgpool;
    DescriptorSet avgpoolDescSet;

public:
    GlobalAvgPoolNode();
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


class IdentityNode : public Node
{
public:
    IdentityNode();
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


extern Device netGlobalDevice;



#endif

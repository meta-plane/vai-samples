#ifndef NEURAL_NODES_H
#define NEURAL_NODES_H

#include "../shaders/shaders.hpp"
#include "neuralNet.h"

class ConCatNode : public Node
{
    uint32_t        dim_;

    ComputePipeline concat;
    DescriptorSet concatDescSet;

public:
    ConCatNode(uint32_t dim);
    ~ConCatNode() override = default;

    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

class ConvTransposeNode : public Node
{
public:
    ConvTransposeNode(uint32_t inChannels,uint32_t outChannels,uint32_t kernelSize,uint32_t stride = 2,uint32_t padding = 0);

    void prepare() override;
    void run(CommandBuffer cmdBuff) override;

private:
    uint32_t C_in;   // 입력 채널
    uint32_t C_out;  // 출력 채널
    uint32_t K;      // 커널 크기 (K x K)
    uint32_t S;      // stride
    uint32_t P;      // padding

    ComputePipeline convtranspose;
    DescriptorSet convtransposeDescSet;
};

class BatchNormNode : public Node
{
    ComputePipeline bacthnrom;
    DescriptorSet   bacthnromDescSet;

public:
    BatchNormNode();
    ~BatchNormNode() override = default;
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


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
    const bool discardTail = true; // If true, discard the tail elements that don't fit into the pooling window
    uint32_t P;

    ComputePipeline maxpool;
    DescriptorSet maxpoolDescSet;

public:
    MaxPoolingNode(uint32_t poolSize);
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


extern Device netGlobalDevice; // Global device for neural network operations



#endif // NEURAL_NODES_H
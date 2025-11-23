#ifndef NEURAL_NODES_H
#define NEURAL_NODES_H


#include "neuralNet.h"
#include <cstdint>


class ConvolutionNode : public Node
{
    uint32_t C, F, K;   // C: input channels, F: output channels, K: kernel width
    uint32_t stride;
    int32_t padding;

    ComputePipeline im2col;
    ComputePipeline gemm;
    DescriptorSet im2colDescSet;
    DescriptorSet gemmDescSet;
    uint32_t gemmTileSize;

public:
    ConvolutionNode(uint32_t inChannels, uint32_t outChannels, uint32_t kernelWidth, uint32_t stride = 1, int32_t padding = 0);
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

// TODO: Node Implementation
class AdaptiveAvgPoolingNode : public Node
{
    uint32_t outH = 1;
    uint32_t outW = 1;

    ComputePipeline avgpool;
    DescriptorSet avgpoolDescSet;

public:
    // Square output (e.g., 1x1 GAP)
    AdaptiveAvgPoolingNode(uint32_t outputSize);
    // Rectangular output
    AdaptiveAvgPoolingNode(uint32_t outH, uint32_t outW);
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

class HSNode : public Node
{
    ComputePipeline hs;
    DescriptorSet hsDescSet;

public:
    HSNode();
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

class MultiplyNode : public Node
{
    ComputePipeline multiply;
    DescriptorSet multiplyDescSet;

public:
    MultiplyNode();
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

class BatchNormNode : public Node
{
    uint32_t C;

    ComputePipeline bn;
    DescriptorSet bnDescSet;

public:
    BatchNormNode(uint32_t channel);
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

class ChannelShuffleNode : public Node
{
    uint32_t C;

    ComputePipeline cs;
    DescriptorSet csDescSet;

public:
    ChannelShuffleNode(uint32_t channels);
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

class ConcatNode : public Node
{
    ComputePipeline concat;
    DescriptorSet concatDescSet;

public:
    ConcatNode();
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

// SplitNode: fan-out one input tensor to two identical outputs
class SplitNode : public Node
{
    ComputePipeline dup2; // copy to two outputs
    DescriptorSet dup2DescSet;

public:
    SplitNode();
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

extern Device netGlobalDevice; // Global device for neural network operations

#endif // NEURAL_NODES_H

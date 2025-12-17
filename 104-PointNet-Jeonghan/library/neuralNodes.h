#ifndef NEURAL_NODES_H
#define NEURAL_NODES_H


#include "neuralNet.h"
#include "vulkanApp.h"


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


class PointWiseMLPNode : public Node
{
    uint32_t Cin, Cout;

    ComputePipeline gemm;
    DescriptorSet gemmDesc;
    
    ComputePipeline batchnorm;
    DescriptorSet batchnormDesc;
    
    ComputePipeline relu;
    DescriptorSet reluDesc;

public:
    PointWiseMLPNode(uint32_t inDim, uint32_t outDim);

    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


class PointWiseConvNode : public Node
{
    uint32_t Cin, Cout;

    ComputePipeline gemm;
    DescriptorSet gemmDesc;
    
    ComputePipeline batchnorm;
    DescriptorSet batchnormDesc;

public:
    PointWiseConvNode(uint32_t inDim, uint32_t outDim);

    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


class BatchNorm1DNode : public Node
{
    uint32_t C;  // channels

    ComputePipeline batchnorm;
    DescriptorSet batchnormDesc;

public:
    BatchNorm1DNode(uint32_t channels);

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

class MaxPooling1DNode : public Node
{

    uint32_t C;

    ComputePipeline maxpool;
    DescriptorSet desc;

public:
    MaxPooling1DNode();
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


// BroadcastNode: [1, C] → [N, C]
// Broadcasts a global feature vector to all points
class BroadcastNode : public Node
{
    ComputePipeline broadcast;
    DescriptorSet broadcastDescSet;

public:
    BroadcastNode();
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


// ConcatNode: [N, C1] + [N, C2] → [N, C1+C2]
// Concatenates two tensors along the channel dimension
class ConcatNode : public Node
{
    ComputePipeline concat;
    DescriptorSet concatDescSet;

public:
    ConcatNode();
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

class ReShapeNode : public Node
{
    ComputePipeline copy;
    DescriptorSet copyDescSet;
    std::vector<uint32_t> targetShape;

public:
    ReShapeNode();
    ReShapeNode(std::vector<uint32_t> shape);
    void setTargetShape(std::vector<uint32_t> shape);
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

// Matrix multiplication: [N, K] @ [K, M] -> [N, M]
class MatMulNode : public Node
{
    ComputePipeline matmul;
    DescriptorSet matmulDescSet;

public:
    MatMulNode();
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

// AddIdentityNode - Adds identity matrix to input [K, K]
// Used in TNet to add identity to transformation matrix
class AddIdentityNode : public Node
{
    ComputePipeline addIdentity;
    DescriptorSet addIdentityDescSet;
    uint32_t K;

public:
    AddIdentityNode();
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

// IdentityNode - Pass-through with multiple outputs for signal splitting
class IdentityNode : public Node
{
public:
    IdentityNode();
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

extern Device netGlobalDevice; // Global device for neural network operations

// Load all compute shaders (must be called before using any nodes)
void loadShaders();

#endif // NEURAL_NODES_H
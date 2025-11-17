#ifndef TRANSFORMER_NODE_H
#define TRANSFORMER_NODE_H

#include "../../core/neuralNet.h"
#include "../../core/vulkanApp.h"
#include "../attention/attentionNode.h"
#include <memory>

using namespace vk;

// Global device and descriptor pool (defined in test file)
extern Device netGlobalDevice;
extern DescriptorPool gDestSetPool;

/**
 * Layer Normalization
 * Input: [batch, seq_len, d_model]
 * Output: [batch, seq_len, d_model]
 *
 * Normalizes over the last dimension: mean=0, variance=1
 * Then applies learnable scale and shift parameters
 *
 * Formula: output = scale * (x - mean) / sqrt(var + eps) + shift
 */
class LayerNormNode : public Node
{
    uint32_t normalized_shape;  // d_model
    float eps;

    ComputePipeline layerNormPipeline;
    DescriptorSet layerNormDescSet;

public:
    LayerNormNode(uint32_t normalized_shape, float eps = 1e-5f);

    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

/**
 * GELU Activation Function
 * Input: [batch, seq_len, d_model]
 * Output: [batch, seq_len, d_model]
 *
 * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
class GELUNode : public Node
{
    ComputePipeline geluPipeline;
    DescriptorSet geluDescSet;

public:
    GELUNode();

    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

/**
 * Feed Forward Network (MLP)
 * Input: [batch, seq_len, d_model]
 * Output: [batch, seq_len, d_model]
 *
 * Architecture:
 * - Linear: d_model → 4*d_model
 * - GELU activation
 * - Linear: 4*d_model → d_model
 */
class FeedForwardNode : public Node
{
    uint32_t d_model;
    uint32_t hidden_dim;  // 4 * d_model

    ComputePipeline linear1Pipeline;
    ComputePipeline geluPipeline;
    ComputePipeline linear2Pipeline;

    DescriptorSet linear1DescSet;
    DescriptorSet geluDescSet;
    DescriptorSet linear2DescSet;

public:
    FeedForwardNode(uint32_t d_model);

    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

/**
 * Identity Node (for fan-out/routing)
 * Input: in0
 * Output: out0 (same as input)
 * Allows one input to connect to multiple downstream nodes
 */
class IdentityNode : public Node
{
public:
    IdentityNode();

    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

/**
 * Add Node (for residual connections)
 * Inputs: in0 (residual), in1 (main path)
 * Output: out0 = in0 + in1
 */
class AddNode : public Node
{
    ComputePipeline addPipeline;
    DescriptorSet addDescSet;

public:
    AddNode();

    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

/**
 * Transformer Block (NodeGroup)
 * Input: [batch, seq_len, d_model]
 * Output: [batch, seq_len, d_model]
 *
 * Architecture (pre-norm):
 * - x = x + MultiHeadAttention(LayerNorm(x))
 * - x = x + FeedForward(LayerNorm(x))
 */
class TransformerBlockNode : public NodeGroup
{
    uint32_t d_model;
    uint32_t num_heads;

    // Internal nodes (using smart pointers)
    std::unique_ptr<IdentityNode> inputRouter;  // Routes input to both main path and residual
    std::unique_ptr<LayerNormNode> norm1;
    std::unique_ptr<MultiHeadAttentionNode> attention;
    std::unique_ptr<AddNode> add1;
    std::unique_ptr<LayerNormNode> norm2;
    std::unique_ptr<FeedForwardNode> feedforward;
    std::unique_ptr<AddNode> add2;

public:
    TransformerBlockNode(uint32_t d_model, uint32_t num_heads);

    // Provide weight access
    Tensor& operator[](const std::string& name);
};

#endif // TRANSFORMER_NODE_H

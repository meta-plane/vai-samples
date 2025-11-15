#ifndef EMBEDDING_NODE_H
#define EMBEDDING_NODE_H

#include "../../core/neuralNet.h"

// Token Embedding Node
// Converts token IDs to dense vectors using Vulkan compute shader
class TokenEmbeddingNode : public Node
{
    uint32_t V;  // vocab_size
    uint32_t E;  // embedding_dim

    ComputePipeline tokenEmbedding;
    DescriptorSet tokenEmbeddingDescSet;

public:
    TokenEmbeddingNode(uint32_t vocab_size, uint32_t embedding_dim);
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


// Positional Embedding Node
// Adds position information to embeddings using Vulkan compute shader
class PositionalEmbeddingNode : public Node
{
    uint32_t M;  // max_length
    uint32_t E;  // embedding_dim

    ComputePipeline positionalEmbedding;
    DescriptorSet positionalEmbeddingDescSet;

public:
    PositionalEmbeddingNode(uint32_t max_length, uint32_t embedding_dim);
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


// GPT Embedding Node (Token + Positional)
// Combines token and positional embeddings
class GPTEmbeddingNode : public Node
{
    uint32_t V;  // vocab_size
    uint32_t M;  // max_length
    uint32_t E;  // embedding_dim

    ComputePipeline tokenEmbedding;
    ComputePipeline positionalEmbedding;
    ComputePipeline addEmbeddings;

    DescriptorSet tokenEmbeddingDescSet;
    DescriptorSet positionalEmbeddingDescSet;
    DescriptorSet addEmbeddingsDescSet;

public:
    GPTEmbeddingNode(uint32_t vocab_size, uint32_t max_length, uint32_t embedding_dim);
    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};


#endif // EMBEDDING_NODE_H

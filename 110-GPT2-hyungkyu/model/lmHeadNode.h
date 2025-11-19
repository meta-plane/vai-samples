#ifndef LM_HEAD_NODE_H
#define LM_HEAD_NODE_H

#include "../core/neuralNet.h"

using namespace vk;

// Global device and descriptor pool
extern Device netGlobalDevice;
extern DescriptorPool gDestSetPool;

/**
 * Language Modeling Head Node
 * Projects hidden states to vocabulary logits using weight tying
 *
 * Input: [batch, seq_len, d_model] hidden states
 * Output: [batch, seq_len, vocab_size] logits
 */
class LMHeadNode : public Node
{
    uint32_t d_model;
    uint32_t vocab_size;

    ComputePipeline lmHeadPipeline;
    DescriptorSet lmHeadDescSet;

public:
    LMHeadNode(uint32_t d_model, uint32_t vocab_size);

    void prepare() override;
    void run(CommandBuffer cmdBuff) override;
};

#endif // LM_HEAD_NODE_H

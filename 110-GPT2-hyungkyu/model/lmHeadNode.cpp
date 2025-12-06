#include "lmHeadNode.h"
#include "../core/globalContext.h"
#include "../core/error.h"
#include <unordered_map>
#include <iostream>

using namespace vk;

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// Shader for LM head matrix multiplication
// logits[b,s,v] = sum_d(hidden[b,s,d] * weight[v,d])
const char* src_lm_head = R"(
#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) buffer Logits { float logits[]; };    // [BS, vocab_size]
layout(set = 0, binding = 1) buffer Hidden { float hidden[]; };    // [BS, d_model]
layout(set = 0, binding = 2) buffer Weight { float weight[]; };    // [vocab_size, d_model]

layout(push_constant) uniform PushConstants {
    int BS;          // batch_size * seq_len
    int d_model;
    int vocab_size;
};

void main() {
    int row = int(gl_GlobalInvocationID.x);  // position in batch*seq (0..BS-1)
    int col = int(gl_GlobalInvocationID.y);  // vocabulary index (0..vocab_size-1)

    if (row >= BS || col >= vocab_size) return;

    // Compute logits[row, col] = sum(hidden[row, d] * weight[col, d])
    float sum = 0.0;
    for (int d = 0; d < d_model; ++d) {
        sum += hidden[row * d_model + d] * weight[col * d_model + d];
    }

    logits[row * vocab_size + col] = sum;
}
)";

LMHeadNode::LMHeadNode(uint32_t d_model, uint32_t vocab_size)
    : d_model(d_model), vocab_size(vocab_size)
{
    addSlot("in0", NodeSlot::input);     // Input: hidden states [B, S, d_model]
    addSlot("weight", NodeSlot::input);  // Weight: token embeddings [vocab_size, d_model] (learnable parameter)
    addSlot("out0", NodeSlot::output);   // Output: logits [B, S, vocab_size]

    lmHeadPipeline = requestPipeline(src_lm_head);
    lmHeadDescSet = lmHeadPipeline.descSetLayout(0).newDescSet(gDestSetPool);
}

void LMHeadNode::prepare()
{
    Tensor& input = (*this)["in0"];
    _ASSERT(input.validShape());
    _ASSERT(input.shape().size() == 3);  // [B, S, D]

    uint32_t B = input.shape()[0];
    uint32_t S = input.shape()[1];
    uint32_t D = input.shape()[2];
    _ASSERT(D == d_model);

    // Always create output tensor (like LayerNormNode and FeedForwardBlock)
    // Use rvalue assignment, not lvalue!
    (*this)["out0"] = Tensor(B, S, vocab_size);
}

void LMHeadNode::run(CommandBuffer cmdBuff)
{
    Tensor& hidden = (*this)["in0"];     // [B, S, d_model]
    Tensor& weight = (*this)["weight"];  // [vocab_size, d_model]
    Tensor& logits = (*this)["out0"];    // [B, S, vocab_size]

    uint32_t B = hidden.shape()[0];
    uint32_t S = hidden.shape()[1];
    uint32_t BS = B * S;

    // Bind buffers
    lmHeadDescSet.write({
        logits.buffer(),
        hidden.buffer(),
        weight.buffer()
    });

    // Push constants
    int constants[] = {(int)BS, (int)d_model, (int)vocab_size};

    // Execute shader
    cmdBuff
        .bindPipeline(lmHeadPipeline)
        .setPushConstants(0, sizeof(constants), constants)
        .bindDescSets({lmHeadDescSet})
        .dispatch0(CEIL_DIV(BS, 16), CEIL_DIV(vocab_size, 16));
}

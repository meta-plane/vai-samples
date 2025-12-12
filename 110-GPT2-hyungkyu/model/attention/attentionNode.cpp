#include "attentionNode.h"
#include "../../core/error.h"
#include "../../core/globalContext.h"
#include <cmath>

using namespace vk;

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// ============================================================================
// LinearNode: Y = X @ W^T
// ============================================================================

const char* src_linear = R"(
#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) buffer Output { float y[]; };      // [B*S*O]
layout(set = 0, binding = 1) buffer Input { float x[]; };       // [B*S*I]
layout(set = 0, binding = 2) buffer Weight { float w[]; };      // [O*I]
layout(set = 0, binding = 3) buffer Bias { float b[]; };        // [O] - NEW: bias

layout(push_constant) uniform PushConstants {
    int B;   // batch size
    int S;   // sequence length
    int I;   // in_features
    int O;   // out_features
};

void main() {
    int bs = int(gl_GlobalInvocationID.x);  // batch * seq index
    int o = int(gl_GlobalInvocationID.y);   // output feature index

    int BS = B * S;
    if (bs >= BS || o >= O) return;

    float sum = 0.0;
    for (int i = 0; i < I; ++i) {
        sum += x[bs * I + i] * w[o * I + i];
    }

    y[bs * O + o] = sum + b[o];
}
)";

// LinearNode GEMV version (for M=1 case with subgroup optimization)
static const char* src_linear_gemv = R"(
#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_ballot : enable

layout(local_size_x = 32, local_size_y = 1) in;

layout(set = 0, binding = 0) buffer Output { float y[]; };      // [M*O]
layout(set = 0, binding = 1) buffer Input { float x[]; };       // [M*I]
layout(set = 0, binding = 2) buffer Weight { float w[]; };      // [O*I]
layout(set = 0, binding = 3) buffer Bias { float b[]; };        // [O]

layout(push_constant) uniform PushConstants {
    int M;   // batch_size * seq_len (usually 1 in KV cache)
    int I;   // in_features
    int O;   // out_features
};

void main() {
    uint tid = gl_LocalInvocationID.x;   // Thread ID [0, 31]
    uint row_idx = gl_WorkGroupID.x;     // Which row (M dimension)
    uint col_idx = gl_WorkGroupID.y;     // Which output feature (O dimension)

    if (row_idx >= M || col_idx >= O) return;

    // Compute partial dot product over strided I
    float partial_sum = 0.0;
    for (uint i = tid; i < I; i += 32) {
        partial_sum += x[row_idx * I + i] * w[col_idx * I + i];
    }

    // Subgroup reduction
    float total_sum = subgroupAdd(partial_sum);

    // First thread writes result with bias
    if (subgroupElect()) {
        y[row_idx * O + col_idx] = total_sum + b[col_idx];
    }
}
)";

LinearNode::LinearNode(uint32_t in_features, uint32_t out_features)
    : in_features(in_features), out_features(out_features)
{
    addSlot("in0", NodeSlot::input);
    addSlot("weight", NodeSlot::input);  // learnable parameter
    addSlot("bias", NodeSlot::input);    // learnable parameter (bias)
    addSlot("out0", NodeSlot::output);

    linearPipeline = requestPipeline(src_linear);
    linearDescSet = linearPipeline.descSetLayout(0).newDescSet(gDestSetPool);
}

void LinearNode::prepare()
{
    Tensor& input = (*this)["in0"];
    _ASSERT(input.validShape());
    _ASSERT(input.shape().size() == 3);  // [B, S, I]

    uint32_t B = input.shape()[0];
    uint32_t S = input.shape()[1];
    uint32_t I = input.shape()[2];
    _ASSERT(I == in_features);

    Tensor& weight = (*this)["weight"];
    if (!weight.validShape()) {
        weight = Tensor(out_features, in_features);
    }

    Tensor& bias = (*this)["bias"];
    if (!bias.validShape()) {
        bias = Tensor(out_features);
        // Initialize bias to zero if not provided
        std::vector<float> zero_bias(out_features, 0.0f);
        bias.set(zero_bias);
    }

    (*this)["out0"] = Tensor(B, S, out_features);
}

void LinearNode::run(CommandBuffer cmdBuff)
{
    Tensor& input = (*this)["in0"];
    Tensor& weight = (*this)["weight"];
    Tensor& bias = (*this)["bias"];
    Tensor& output = (*this)["out0"];

    uint32_t B = input.shape()[0];
    uint32_t S = input.shape()[1];
    uint32_t BS = B * S;

    linearDescSet.write({
        output.buffer(),
        input.buffer(),
        weight.buffer(),
        bias.buffer()  // Add bias buffer binding
    });

    int constants[] = {(int)B, (int)S, (int)in_features, (int)out_features};

    cmdBuff
        .bindPipeline(linearPipeline)
        .setPushConstants(0, sizeof(constants), constants)
        .bindDescSets({linearDescSet})
        .dispatch0(CEIL_DIV(BS, 16), CEIL_DIV(out_features, 16))
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / output.buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}

// ============================================================================
// SoftmaxNode: Numerically stable softmax
// ============================================================================

const char* src_softmax = R"(
#version 450
layout(local_size_x = 64) in;

layout(set = 0, binding = 0) buffer Output { float y[]; };
layout(set = 0, binding = 1) buffer Input { float x[]; };

layout(push_constant) uniform PushConstants {
    int num_rows;
    int row_size;
};

void main() {
    int row = int(gl_GlobalInvocationID.x);
    if (row >= num_rows) return;

    int offset = row * row_size;

    // Find max value
    float max_val = x[offset];
    for (int i = 1; i < row_size; ++i) {
        max_val = max(max_val, x[offset + i]);
    }

    // Compute exp and sum
    float sum_exp = 0.0;
    for (int i = 0; i < row_size; ++i) {
        sum_exp += exp(x[offset + i] - max_val);
    }

    // Normalize
    for (int i = 0; i < row_size; ++i) {
        y[offset + i] = exp(x[offset + i] - max_val) / sum_exp;
    }
}
)";

SoftmaxNode::SoftmaxNode()
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);

    softmaxPipeline = requestPipeline(src_softmax);
    softmaxDescSet = softmaxPipeline.descSetLayout(0).newDescSet(gDestSetPool);
}

void SoftmaxNode::prepare()
{
    Tensor& input = (*this)["in0"];
    _ASSERT(input.validShape());

    // Output has same shape as input
    (*this)["out0"] = Tensor(input.shape());
}

void SoftmaxNode::run(CommandBuffer cmdBuff)
{
    Tensor& input = (*this)["in0"];
    Tensor& output = (*this)["out0"];

    auto shape = input.shape();
    int num_rows = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) {
        num_rows *= shape[i];
    }
    int row_size = shape.back();

    softmaxDescSet.write({
        output.buffer(),
        input.buffer()
    });

    int constants[] = {num_rows, row_size};

    cmdBuff
        .bindPipeline(softmaxPipeline)
        .setPushConstants(0, sizeof(constants), constants)
        .bindDescSets({softmaxDescSet})
        .dispatch0(CEIL_DIV(num_rows, 64))
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / output.buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}

// ============================================================================
// MultiHeadAttentionNode
// ============================================================================

// Shader 1: Project input to Q, K, V (3 separate linear projections)
const char* src_qkv_projection = R"(
#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) buffer Q { float q[]; };        // [B*S*D_out]
layout(set = 0, binding = 1) buffer K { float k[]; };        // [B*S*D_out]
layout(set = 0, binding = 2) buffer V { float v[]; };        // [B*S*D_out]
layout(set = 0, binding = 3) buffer Input { float x[]; };    // [B*S*D_in]
layout(set = 0, binding = 4) buffer Wq { float wq[]; };      // [D_out*D_in]
layout(set = 0, binding = 5) buffer Wk { float wk[]; };      // [D_out*D_in]
layout(set = 0, binding = 6) buffer Wv { float wv[]; };      // [D_out*D_in]
layout(set = 0, binding = 7) buffer Bq { float bq[]; };      // [D_out] - NEW: Q bias
layout(set = 0, binding = 8) buffer Bk { float bk[]; };      // [D_out] - NEW: K bias
layout(set = 0, binding = 9) buffer Bv { float bv[]; };      // [D_out] - NEW: V bias

layout(push_constant) uniform PushConstants {
    int B, S, D_in, D_out;
};

void main() {
    int bs = int(gl_GlobalInvocationID.x);
    int d_out = int(gl_GlobalInvocationID.y);

    int BS = B * S;
    if (bs >= BS || d_out >= D_out) return;

    // Q = X @ Wq^T + Bq
    // X: [bs, D_in], Wq: [D_out, D_in] -> Q: [bs, D_out]
    float q_val = 0.0;
    for (int i = 0; i < D_in; ++i) {
        q_val += x[bs * D_in + i] * wq[d_out * D_in + i];
    }
    q[bs * D_out + d_out] = q_val + bq[d_out];

    // K = X @ Wk^T + Bk
    float k_val = 0.0;
    for (int i = 0; i < D_in; ++i) {
        k_val += x[bs * D_in + i] * wk[d_out * D_in + i];
    }
    k[bs * D_out + d_out] = k_val + bk[d_out];

    // V = X @ Wv^T + Bv
    float v_val = 0.0;
    for (int i = 0; i < D_in; ++i) {
        v_val += x[bs * D_in + i] * wv[d_out * D_in + i];
    }
    v[bs * D_out + d_out] = v_val + bv[d_out];
}
)";

// Shader 1b: QKV Projection with Subgroup Optimization (for M=1 case)
const char* src_qkv_projection_gemv = R"(
#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_ballot : enable

layout(local_size_x = 32, local_size_y = 1) in;

layout(set = 0, binding = 0) buffer Q { float q[]; };        // [M*D_out]
layout(set = 0, binding = 1) buffer K { float k[]; };        // [M*D_out]
layout(set = 0, binding = 2) buffer V { float v[]; };        // [M*D_out]
layout(set = 0, binding = 3) buffer Input { float x[]; };    // [M*D_in]
layout(set = 0, binding = 4) buffer Wq { float wq[]; };      // [D_out*D_in]
layout(set = 0, binding = 5) buffer Wk { float wk[]; };      // [D_out*D_in]
layout(set = 0, binding = 6) buffer Wv { float wv[]; };      // [D_out*D_in]
layout(set = 0, binding = 7) buffer Bq { float bq[]; };      // [D_out]
layout(set = 0, binding = 8) buffer Bk { float bk[]; };      // [D_out]
layout(set = 0, binding = 9) buffer Bv { float bv[]; };      // [D_out]

layout(push_constant) uniform PushConstants {
    int M;      // batch_size * seq_len (usually 1 in KV cache)
    int D_in;   // input features
    int D_out;  // output features
};

void main() {
    uint tid = gl_LocalInvocationID.x;   // Thread ID [0, 31]
    uint row_idx = gl_WorkGroupID.x;     // Which row (M dimension)
    uint col_idx = gl_WorkGroupID.y;     // Which output feature (D_out dimension)

    if (row_idx >= M || col_idx >= D_out) return;

    // Compute Q: partial dot product over strided D_in
    float q_partial = 0.0;
    for (uint i = tid; i < D_in; i += 32) {
        q_partial += x[row_idx * D_in + i] * wq[col_idx * D_in + i];
    }
    float q_sum = subgroupAdd(q_partial);

    // Compute K: partial dot product over strided D_in
    float k_partial = 0.0;
    for (uint i = tid; i < D_in; i += 32) {
        k_partial += x[row_idx * D_in + i] * wk[col_idx * D_in + i];
    }
    float k_sum = subgroupAdd(k_partial);

    // Compute V: partial dot product over strided D_in
    float v_partial = 0.0;
    for (uint i = tid; i < D_in; i += 32) {
        v_partial += x[row_idx * D_in + i] * wv[col_idx * D_in + i];
    }
    float v_sum = subgroupAdd(v_partial);

    // First thread writes results with bias
    if (subgroupElect()) {
        q[row_idx * D_out + col_idx] = q_sum + bq[col_idx];
        k[row_idx * D_out + col_idx] = k_sum + bk[col_idx];
        v[row_idx * D_out + col_idx] = v_sum + bv[col_idx];
    }
}
)";

// Shader 2: Compute attention scores: Q @ K^T / sqrt(head_dim)
// Input Q, K: [B, S, d_in] where d_in = H * HD
// Output scores: [B, H, S, S]
const char* src_attention_scores = R"(
#version 450
layout(local_size_x = 8, local_size_y = 8) in;

layout(set = 0, binding = 0) buffer Scores { float scores[]; };  // [B*H*S*S]
layout(set = 0, binding = 1) buffer Q { float q[]; };             // [B*S*d_in]
layout(set = 0, binding = 2) buffer K { float k[]; };             // [B*S*d_in]

layout(push_constant) uniform PushConstants {
    int B, H, S, HD;
    float scale;
};

void main() {
    int bh = int(gl_GlobalInvocationID.x);  // batch * head
    int s1 = int(gl_GlobalInvocationID.y);  // query position

    int BH = B * H;
    if (bh >= BH || s1 >= S) return;

    int b = bh / H;
    int h = bh % H;
    int D = H * HD;

    // Compute scores for all key positions
    for (int s2 = 0; s2 < S; ++s2) {
        float score = 0.0;
        for (int hd = 0; hd < HD; ++hd) {
            // Q[b, s1, h*HD + hd]
            float q_val = q[b * S * D + s1 * D + h * HD + hd];
            // K[b, s2, h*HD + hd]
            float k_val = k[b * S * D + s2 * D + h * HD + hd];
            score += q_val * k_val;
        }
        scores[bh * S * S + s1 * S + s2] = score * scale;
    }
}
)";

// Shader 2b: Compute attention scores with different Q and K sequence lengths
// Used when KV cache is active
// Input Q: [B, H, S_q, HD] (reshaped from flat)
// Input K: [B, H, S_kv, HD] (concatenated cache + new)
// Output scores: [B, H, S_q, S_kv]
const char* src_attention_scores_cached = R"(
#version 450
layout(local_size_x = 8, local_size_y = 8) in;

layout(set = 0, binding = 0) buffer Scores { float scores[]; };  // [B*H*S_q*S_kv]
layout(set = 0, binding = 1) buffer Q { float q[]; };             // [B*H*S_q*HD]
layout(set = 0, binding = 2) buffer K { float k[]; };             // [B*H*S_kv*HD]

layout(push_constant) uniform PushConstants {
    int B, H, S_q, S_kv, HD;
    float scale;
};

void main() {
    int bh = int(gl_GlobalInvocationID.x);  // batch * head
    int sq = int(gl_GlobalInvocationID.y);  // query position

    int BH = B * H;
    if (bh >= BH || sq >= S_q) return;

    // Compute scores for all key positions
    for (int skv = 0; skv < S_kv; ++skv) {
        float score = 0.0;
        for (int hd = 0; hd < HD; ++hd) {
            // Q[bh, sq, hd]
            float q_val = q[bh * S_q * HD + sq * HD + hd];
            // K[bh, skv, hd]
            float k_val = k[bh * S_kv * HD + skv * HD + hd];
            score += q_val * k_val;
        }
        scores[bh * S_q * S_kv + sq * S_kv + skv] = score * scale;
    }
}
)";

// Shader 3: Apply causal mask (set upper triangle to -inf)
const char* src_causal_mask = R"(
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer Scores { float scores[]; };

layout(push_constant) uniform PushConstants {
    int B, H, S;
};

void main() {
    int idx = int(gl_GlobalInvocationID.x);
    int BHS2 = B * H * S * S;

    if (idx >= BHS2) return;

    // Decode indices
    int bhs2 = idx;
    int s2 = bhs2 % S;
    bhs2 /= S;
    int s1 = bhs2 % S;

    // Apply causal mask: if s2 > s1, set to -inf
    if (s2 > s1) {
        scores[idx] = -1e38;  // -inf approximation
    }
}
)";

// Shader 3b: Apply causal mask for cached attention
// When using cache, Q has S_q tokens and K has S_kv tokens
// The causal mask logic: query token at position (cache_len + sq) can attend to keys at positions [0, cache_len + sq]
// So: if skv > (cache_len + sq), mask it
const char* src_causal_mask_cached = R"(
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer Scores { float scores[]; };

layout(push_constant) uniform PushConstants {
    int B, H, S_q, S_kv, cache_len;
};

void main() {
    int idx = int(gl_GlobalInvocationID.x);
    int total = B * H * S_q * S_kv;

    if (idx >= total) return;

    // Decode indices: [bh, sq, skv]
    int temp = idx;
    int skv = temp % S_kv;
    temp /= S_kv;
    int sq = temp % S_q;
    int bh = temp / S_q;

    // Query absolute position in full sequence
    int q_abs_pos = cache_len + sq;

    // Apply causal mask: query at q_abs_pos can only attend to keys at [0, q_abs_pos]
    if (skv > q_abs_pos) {
        scores[idx] = -1e38;  // -inf approximation
    }
}
)";

// Shader 4: Weighted sum: context = attn_weights @ V
// Input V: [B, S, d_in] where d_in = H * HD
// Output context: [B, H, S, HD]
const char* src_weighted_sum = R"(
#version 450
layout(local_size_x = 8, local_size_y = 8) in;

layout(set = 0, binding = 0) buffer Context { float ctx[]; };         // [B*H*S*HD]
layout(set = 0, binding = 1) buffer AttnWeights { float attn[]; };    // [B*H*S*S]
layout(set = 0, binding = 2) buffer V { float v[]; };                 // [B*S*d_in]

layout(push_constant) uniform PushConstants {
    int B, H, S, HD;
};

void main() {
    int bh = int(gl_GlobalInvocationID.x);
    int s = int(gl_GlobalInvocationID.y);

    int BH = B * H;
    if (bh >= BH || s >= S) return;

    int b = bh / H;
    int h = bh % H;
    int D = H * HD;

    // context[bh, s, :] = attn_weights[bh, s, :] @ V[bh, :, :]
    for (int hd = 0; hd < HD; ++hd) {
        float sum = 0.0;
        for (int s2 = 0; s2 < S; ++s2) {
            float weight = attn[bh * S * S + s * S + s2];
            // V[b, s2, h*HD + hd]
            float v_val = v[b * S * D + s2 * D + h * HD + hd];
            sum += weight * v_val;
        }
        ctx[bh * S * HD + s * HD + hd] = sum;
    }
}
)";

// Shader 4b: Weighted sum for cached attention
// Input V: [B, H, S_kv, HD] (concatenated cache + new)
// Input attn: [B, H, S_q, S_kv]
// Output context: [B, H, S_q, HD]
const char* src_weighted_sum_cached = R"(
#version 450
layout(local_size_x = 8, local_size_y = 8) in;

layout(set = 0, binding = 0) buffer Context { float ctx[]; };         // [B*H*S_q*HD]
layout(set = 0, binding = 1) buffer AttnWeights { float attn[]; };    // [B*H*S_q*S_kv]
layout(set = 0, binding = 2) buffer V { float v[]; };                 // [B*H*S_kv*HD]

layout(push_constant) uniform PushConstants {
    int B, H, S_q, S_kv, HD;
};

void main() {
    int bh = int(gl_GlobalInvocationID.x);
    int sq = int(gl_GlobalInvocationID.y);

    int BH = B * H;
    if (bh >= BH || sq >= S_q) return;

    // context[bh, sq, :] = attn_weights[bh, sq, :] @ V[bh, :, :]
    for (int hd = 0; hd < HD; ++hd) {
        float sum = 0.0;
        for (int skv = 0; skv < S_kv; ++skv) {
            float weight = attn[bh * S_q * S_kv + sq * S_kv + skv];
            // V[bh, skv, hd]
            float v_val = v[bh * S_kv * HD + skv * HD + hd];
            sum += weight * v_val;
        }
        ctx[bh * S_q * HD + sq * HD + hd] = sum;
    }
}
)";

// Shader 5: Combine heads and reshape
// Input: [B, H, S, HD]
// Output: [B, S, d_in] where d_in = H * HD
const char* src_combine_heads = R"(
#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) buffer Output { float out0[]; };     // [B*S*d_in]
layout(set = 0, binding = 1) buffer Context { float ctx[]; };     // [B*H*S*HD]

layout(push_constant) uniform PushConstants {
    int B, H, S, HD;
};

void main() {
    int bs = int(gl_GlobalInvocationID.x);
    int d = int(gl_GlobalInvocationID.y);

    int BS = B * S;
    int D = H * HD;

    if (bs >= BS || d >= D) return;

    int b = bs / S;
    int s = bs % S;
    int h = d / HD;
    int hd = d % HD;

    // out[b, s, d] = context[b, h, s, hd]
    out0[bs * D + d] = ctx[b * H * S * HD + h * S * HD + s * HD + hd];
}
)";

// ============================================================================
// Reshape Shader for KV Cache
// ============================================================================

/**
 * Reshape Q/K/V from flat format to multi-head format
 * Input: [B, S, D] where D = H * HD (flat format)
 * Output: [B, H, S, HD] (multi-head format)
 */
const char* src_reshape_to_heads = R"(
#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) buffer Output { float out_data[]; };  // [B*H*S*HD]
layout(set = 0, binding = 1) buffer Input { float in_data[]; };    // [B*S*D]

layout(push_constant) uniform PushConstants {
    int B, S, H, HD;
};

void main() {
    int bs = int(gl_GlobalInvocationID.x);
    int hhd = int(gl_GlobalInvocationID.y);

    int BS = B * S;
    int D = H * HD;

    if (bs >= BS || hhd >= D) return;

    int b = bs / S;
    int s = bs % S;
    int h = hhd / HD;
    int hd = hhd % HD;

    // in: [b, s, h*HD + hd]
    int in_idx = b * S * D + s * D + h * HD + hd;
    // out: [b, h, s, hd]
    int out_idx = b * H * S * HD + h * S * HD + s * HD + hd;

    out_data[out_idx] = in_data[in_idx];
}
)";

// ============================================================================
// KV Cache Update Shader
// ============================================================================

/**
 * Update cache with new K/V values
 * Copies new K or V data into the cache at the specified offset
 *
 * Inputs:
 *   - new_data: [B, H, new_len, HD] - New K or V to add to cache
 * Output:
 *   - cache: [B, H, max_len, HD] - Cache buffer (updated in-place)
 *
 * The new data is written at offset [cache_offset : cache_offset + new_len]
 */
const char* src_update_cache = R"(
#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) buffer Cache { float cache[]; };        // [B*H*max_len*HD]
layout(set = 0, binding = 1) buffer NewData { float new_data[]; };   // [B*H*new_len*HD]

layout(push_constant) uniform PushConstants {
    int B;              // Batch size
    int H;              // Number of heads
    int cache_offset;   // Offset in cache where to write new data
    int new_len;        // Length of new data
    int max_len;        // Maximum cache length
    int HD;             // Head dimension
};

void main() {
    int bh = int(gl_GlobalInvocationID.x);  // Batch * head index
    int s = int(gl_GlobalInvocationID.y);   // Sequence index in new data

    int BH = B * H;
    if (bh >= BH || s >= new_len) return;

    int b = bh / H;
    int h = bh % H;

    // Source offset in new_data: [b, h, s, :]
    int src_offset = (b * H * new_len * HD) + (h * new_len * HD) + (s * HD);

    // Destination offset in cache: [b, h, cache_offset + s, :]
    int dst_offset = (b * H * max_len * HD) + (h * max_len * HD) + ((cache_offset + s) * HD);

    // Copy entire HD-dimensional vector
    for (int hd = 0; hd < HD; ++hd) {
        cache[dst_offset + hd] = new_data[src_offset + hd];
    }
}
)";

// ============================================================================
// KV Cache Concatenation Shader
// ============================================================================

/**
 * Concatenate cached K/V with new K/V
 *
 * Inputs:
 *   - cached: [B, H, cache_len, HD] - K or V from previous tokens
 *   - new_kv: [B, H, new_len, HD]   - K or V from new tokens
 * Output:
 *   - full: [B, H, cache_len + new_len, HD] - Combined K or V
 *
 * This shader copies cached data first, then appends new data.
 * Used during autoregressive generation with KV caching.
 */
const char* src_concatenate_kv = R"(
#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) buffer Output { float full[]; };   // [B*H*(cache_len+new_len)*HD]
layout(set = 0, binding = 1) buffer Cached { float cached[]; }; // [B*H*max_len*HD] - FULL cache buffer!
layout(set = 0, binding = 2) buffer NewKV { float new_kv[]; };  // [B*H*new_len*HD]

layout(push_constant) uniform PushConstants {
    int B;          // Batch size
    int H;          // Number of heads
    int cache_len;  // Length of cached sequence (actual used length)
    int new_len;    // Length of new sequence
    int max_len;    // Maximum cache length (stride for cache buffer)
    int HD;         // Head dimension
};

void main() {
    int bh = int(gl_GlobalInvocationID.x);  // Batch * head index
    int s = int(gl_GlobalInvocationID.y);   // Sequence index in output

    int BH = B * H;
    int total_len = cache_len + new_len;

    if (bh >= BH || s >= total_len) return;

    int b = bh / H;
    int h = bh % H;

    // Copy one entire [HD] vector per thread
    // Each thread handles one (b, h, s) position
    int out_offset = (b * H * total_len * HD) + (h * total_len * HD) + (s * HD);

    if (s < cache_len) {
        // Copy from cached data - use max_len as stride!
        int cache_offset = (b * H * max_len * HD) + (h * max_len * HD) + (s * HD);
        for (int hd = 0; hd < HD; ++hd) {
            full[out_offset + hd] = cached[cache_offset + hd];
        }
    } else {
        // Copy from new data
        int new_s = s - cache_len;
        int new_offset = (b * H * new_len * HD) + (h * new_len * HD) + (new_s * HD);
        for (int hd = 0; hd < HD; ++hd) {
            full[out_offset + hd] = new_kv[new_offset + hd];
        }
    }
}
)";

// ============================================================================
// Flash Attention KV Cache Shader (llama.cpp scalar path style)
// ============================================================================

const char* src_flash_attention_kvcache = R"(
// Accurate port of llama.cpp Flash Attention for KV cache mode
// Simplified for: Br=1 (single query), FP32, no quantization
#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(local_size_x = 32, local_size_y = 1) in;

layout(set = 0, binding = 0) buffer Context { float ctx[]; };
layout(set = 0, binding = 1) buffer Q { float q[]; };
layout(set = 0, binding = 2) buffer K { float k[]; };
layout(set = 0, binding = 3) buffer V { float v[]; };

layout(push_constant) uniform PushConstants {
    int B, H, S_q, S_kv, HD;
    float scale;
    int cache_len;
};

// Shared memory for Q and reductions
shared float s_Q[64];  // HD=64
shared float tmpsh[32];  // For reduction

void main() {
    const uint tid = gl_LocalInvocationID.x;
    const uint bh = gl_WorkGroupID.x;
    const uint sq = gl_WorkGroupID.y;

    if (bh >= B * H || sq >= S_q) return;

    const int q_abs_pos = cache_len + int(sq);
    const int q_offset = int(bh * S_q * HD + sq * HD);

    const float NEG_FLT_MAX_OVER_2 = uintBitsToFloat(0xFEFFFFFF);
    const int Bc = 32;  // Process 32 KV positions per iteration

    // Load Q into shared memory
    for (int d = int(tid); d < HD; d += 32) {
        s_Q[d] = q[q_offset + d] * scale;  // Apply scale during load
    }
    barrier();

    // Initialize accumulation variables (Br=1, so no array needed)
    float Lf = 0.0;  // Sum of exp(scores - max)
    float Mf = NEG_FLT_MAX_OVER_2;  // Running max
    float Of[64];  // Output accumulation
    for (int d = 0; d < HD; ++d) {
        Of[d] = 0.0;
    }

    // Process KV in blocks of Bc=32
    const int num_blocks = (S_kv + Bc - 1) / Bc;

    for (int j = 0; j < num_blocks; ++j) {
        const int kv_idx = j * Bc + int(tid);

        // Compute attention score for this KV position
        float Sf = 0.0;  // Score for this thread
        if (kv_idx < S_kv && kv_idx <= q_abs_pos) {
            const int k_offset = int(bh * S_kv * HD + kv_idx * HD);
            for (int d = 0; d < HD; ++d) {
                Sf += s_Q[d] * k[k_offset + d];
            }
        } else {
            Sf = NEG_FLT_MAX_OVER_2;  // Out of bounds
        }

        // Compute rowmax across this block (within thread's view)
        // Each thread has computed one score, no need to max locally
        float rowmaxf = Sf;

        // Save old max
        float Moldf = Mf;

        // Update max
        Mf = max(rowmaxf, Moldf);

        // Compute softmax probability
        float Pf = exp(Sf - Mf);

        // Compute rescale factor
        float eMf = exp(Moldf - Mf);

        // Update sum
        Lf = eMf * Lf + Pf;

        // Rescale previous output
        for (int d = 0; d < HD; ++d) {
            Of[d] = eMf * Of[d];
        }

        // Accumulate weighted V
        if (kv_idx < S_kv && kv_idx <= q_abs_pos) {
            const int v_offset = int(bh * S_kv * HD + kv_idx * HD);
            for (int d = 0; d < HD; ++d) {
                Of[d] += Pf * v[v_offset + d];
            }
        }
    }

    // Final reduction across threads using shared memory tree reduction
    // This matches llama.cpp's approach (lines 261-314)

    // Reduce Mf (max)
    tmpsh[tid] = Mf;
    barrier();
    for (int s = 16; s > 0; s >>= 1) {
        if (tid < s) {
            tmpsh[tid] = max(tmpsh[tid], tmpsh[tid + s]);
        }
        barrier();
    }
    float rowmaxf = tmpsh[0];
    barrier();

    // Rescale based on global max
    float Moldf = Mf;
    Mf = max(rowmaxf, Moldf);
    float eMf = exp(Moldf - Mf);
    Lf = eMf * Lf;

    // Reduce Lf (sum)
    tmpsh[tid] = Lf;
    barrier();
    for (int s = 16; s > 0; s >>= 1) {
        if (tid < s) {
            tmpsh[tid] = tmpsh[tid] + tmpsh[tid + s];
        }
        barrier();
    }
    Lf = tmpsh[0];
    barrier();

    // Reduce Of (output) - accumulate across threads
    for (int d = 0; d < HD; ++d) {
        // Rescale and reduce this dimension
        float val = eMf * Of[d];
        tmpsh[tid] = val;
        barrier();

        // Tree reduction for this dimension
        for (int s = 16; s > 0; s >>= 1) {
            if (tid < s) {
                tmpsh[tid] = tmpsh[tid] + tmpsh[tid + s];
            }
            barrier();
        }

        if (tid == 0) {
            Of[d] = tmpsh[0];
        }
        barrier();
    }

    // Thread 0 writes final normalized output
    if (tid == 0) {
        float Lfrcp = (Lf == 0.0) ? 0.0 : (1.0 / Lf);
        const int out_offset = int(bh * S_q * HD + sq * HD);

        for (int d = 0; d < HD; ++d) {
            ctx[out_offset + d] = Of[d] * Lfrcp;
        }
    }
}
)";

MultiHeadAttentionNode::MultiHeadAttentionNode(uint32_t d_in, uint32_t d_out, uint32_t num_heads)
    : d_in(d_in), d_out(d_out), num_heads(num_heads)
{
    _ASSERT(d_out % num_heads == 0);
    head_dim = d_out / num_heads;

    addSlot("in0", NodeSlot::input);
    addSlot("W_query", NodeSlot::input);  // learnable parameter
    addSlot("W_key", NodeSlot::input);    // learnable parameter
    addSlot("W_value", NodeSlot::input);  // learnable parameter
    addSlot("W_out", NodeSlot::input);    // learnable parameter
    addSlot("B_query", NodeSlot::input);  // NEW: bias parameter
    addSlot("B_key", NodeSlot::input);    // NEW: bias parameter
    addSlot("B_value", NodeSlot::input);  // NEW: bias parameter
    addSlot("B_out", NodeSlot::input);    // NEW: bias parameter
    addSlot("out0", NodeSlot::output);

    // Create pipelines - standard (no cache)
    qkvProjection = requestPipeline(src_qkv_projection);
    qkvProjectionGEMV = requestPipeline(src_qkv_projection_gemv);  // GEMV version for M=1
    attentionScores = requestPipeline(src_attention_scores);
    applyCausalMask = requestPipeline(src_causal_mask);
    softmaxPipeline = requestPipeline(src_softmax);
    weightedSum = requestPipeline(src_weighted_sum);
    combineHeads = requestPipeline(src_combine_heads);

    // Create pipelines - KV cache support
    reshapeForHeads = requestPipeline(src_reshape_to_heads);
    concatenateKV = requestPipeline(src_concatenate_kv);
    updateCache = requestPipeline(src_update_cache);
    scoresPipelineCached = requestPipeline(src_attention_scores_cached);
    maskPipelineCached = requestPipeline(src_causal_mask_cached);
    weightedSumPipelineCached = requestPipeline(src_weighted_sum_cached);
    flashAttentionKVCache = requestPipeline(src_flash_attention_kvcache);

    // Create descriptor sets
    qkvProjDescSet = qkvProjection.descSetLayout(0).newDescSet(gDestSetPool);
    qkvProjDescSetGEMV = qkvProjectionGEMV.descSetLayout(0).newDescSet(gDestSetPool);  // GEMV descriptor set
    reshapeDescSetQ = reshapeForHeads.descSetLayout(0).newDescSet(gDestSetPool);
    reshapeDescSetK = reshapeForHeads.descSetLayout(0).newDescSet(gDestSetPool);
    reshapeDescSetV = reshapeForHeads.descSetLayout(0).newDescSet(gDestSetPool);
    concatDescSetK = concatenateKV.descSetLayout(0).newDescSet(gDestSetPool);
    concatDescSetV = concatenateKV.descSetLayout(0).newDescSet(gDestSetPool);
    updateCacheDescSetK = updateCache.descSetLayout(0).newDescSet(gDestSetPool);
    updateCacheDescSetV = updateCache.descSetLayout(0).newDescSet(gDestSetPool);
    scoresDescSet = attentionScores.descSetLayout(0).newDescSet(gDestSetPool);
    scoresCachedDescSet = scoresPipelineCached.descSetLayout(0).newDescSet(gDestSetPool);
    maskDescSet = applyCausalMask.descSetLayout(0).newDescSet(gDestSetPool);
    maskCachedDescSet = maskPipelineCached.descSetLayout(0).newDescSet(gDestSetPool);
    softmaxDescSet = softmaxPipeline.descSetLayout(0).newDescSet(gDestSetPool);
    weightedSumDescSet = weightedSum.descSetLayout(0).newDescSet(gDestSetPool);
    weightedSumCachedDescSet = weightedSumPipelineCached.descSetLayout(0).newDescSet(gDestSetPool);
    combineDescSet = combineHeads.descSetLayout(0).newDescSet(gDestSetPool);
    flashAttnDescSet = flashAttentionKVCache.descSetLayout(0).newDescSet(gDestSetPool);

    // Output projection pipeline and descriptor set (used in both modes)
    outputProjection = requestPipeline(src_linear);
    outputProjectionGEMV = requestPipeline(src_linear_gemv);  // GEMV version for M=1 (from transformer.cpp)
    outProjDescSet = outputProjection.descSetLayout(0).newDescSet(gDestSetPool);
    outProjDescSetGEMV = outputProjectionGEMV.descSetLayout(0).newDescSet(gDestSetPool);
}

void MultiHeadAttentionNode::prepare()
{
    Tensor& input = (*this)["in0"];
    _ASSERT(input.validShape());
    _ASSERT(input.shape().size() == 3);

    uint32_t B = input.shape()[0];
    uint32_t S = input.shape()[1];
    uint32_t D = input.shape()[2];  // Should be d_in
    _ASSERT(D == d_in);

    // Initialize weights if not set
    Tensor& W_q = (*this)["W_query"];
    Tensor& W_k = (*this)["W_key"];
    Tensor& W_v = (*this)["W_value"];
    Tensor& W_out = (*this)["W_out"];

    // Q, K, V projections: (d_out, d_in) - project input to attention space
    if (!W_q.validShape()) W_q = Tensor(d_out, d_in);
    if (!W_k.validShape()) W_k = Tensor(d_out, d_in);
    if (!W_v.validShape()) W_v = Tensor(d_out, d_in);
    // Output projection: (d_out, d_out) - final transformation in output space
    if (!W_out.validShape()) W_out = Tensor(d_out, d_out);

    // Initialize biases if not set
    Tensor& B_q = (*this)["B_query"];
    Tensor& B_k = (*this)["B_key"];
    Tensor& B_v = (*this)["B_value"];
    Tensor& B_out = (*this)["B_out"];

    if (!B_q.validShape()) B_q = Tensor(d_out);
    if (!B_k.validShape()) B_k = Tensor(d_out);
    if (!B_v.validShape()) B_v = Tensor(d_out);
    if (!B_out.validShape()) B_out = Tensor(d_out);

    (*this)["out0"] = Tensor(B, S, d_out);
}

void MultiHeadAttentionNode::run(CommandBuffer cmdBuff)
{
    Tensor& input = (*this)["in0"];
    Tensor& W_q = (*this)["W_query"];
    Tensor& W_k = (*this)["W_key"];
    Tensor& W_v = (*this)["W_value"];
    Tensor& W_out = (*this)["W_out"];
    Tensor& B_q = (*this)["B_query"];
    Tensor& B_k = (*this)["B_key"];
    Tensor& B_v = (*this)["B_value"];
    Tensor& B_out = (*this)["B_out"];
    Tensor& output = (*this)["out0"];

    uint32_t B = input.shape()[0];
    uint32_t S = input.shape()[1];
    uint32_t D_in = d_in;   // Input dimension
    uint32_t D_out = d_out; // Output/attention space dimension
    uint32_t H = num_heads;
    uint32_t HD = head_dim;

    // Use standard path if cache is disabled OR if cache is empty (cache_len=0)
    // This ensures the prompt phase produces identical results to non-cached mode
    bool use_standard_path = !use_cache || kv_cache == nullptr || kv_cache->current_len == 0;

    if (use_standard_path) {
        // Standard path: no KV cache OR cache is empty (first token)
        IntermediateTensors tensors = allocateIntermediateBuffers(B, S, D_out, H, HD);
        computeQKVProjection(cmdBuff, input, tensors, W_q, W_k, W_v, B_q, B_k, B_v, B, S, D_in, D_out);
        computeAttentionScores(cmdBuff, tensors, B, H, S, HD);
        applyCausalMaskToScores(cmdBuff, tensors, B, H, S);
        computeSoftmax(cmdBuff, tensors, B, H, S);
        computeWeightedSum(cmdBuff, tensors, B, H, S, HD);
        combineHeadsAndProject(cmdBuff, tensors, W_out, B_out, output, B, S, D_out, H, HD);

        // If cache is enabled but empty, update it with the computed K/V
        if (use_cache && kv_cache != nullptr && kv_cache->current_len == 0) {
            // We need to reshape K/V and update cache
            // Extract K and V from tensors.K_flat and tensors.V_flat
            Tensor K_reshaped = Tensor(B, H, S, HD);
            Tensor V_reshaped = Tensor(B, H, S, HD);
            BufferPool& pool = BufferPool::get();
            K_reshaped.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*H*S*HD*sizeof(float)));
            V_reshaped.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*H*S*HD*sizeof(float)));

            reshapeToHeads(cmdBuff, tensors.K_flat, K_reshaped, reshapeDescSetK, B, S, H, HD);
            reshapeToHeads(cmdBuff, tensors.V_flat, V_reshaped, reshapeDescSetV, B, S, H, HD);

            updateCacheWithNewKV(cmdBuff, K_reshaped, V_reshaped, B, H, S, 0, kv_cache->max_len, HD);
        }
    } else {
        // Cache path: use KV cache (cache has data from previous tokens)
        uint32_t cache_len = kv_cache->current_len;
        uint32_t new_S = S;  // Number of new tokens to process
        uint32_t total_S = cache_len + new_S;  // Total sequence length after concatenation

        // Step 1: Allocate intermediate buffers for cached attention
        IntermediateTensors tensors = allocateIntermediateBuffersCached(B, new_S, total_S, D_out, H, HD);

        // Step 2: Compute Q, K, V for new tokens
        computeQKVProjection(cmdBuff, input, tensors, W_q, W_k, W_v, B_q, B_k, B_v, B, new_S, D_in, D_out);

        // Step 3: Reshape Q, K, V to multi-head format [B, H, new_S, HD]
        Tensor Q_reshaped, K_new_reshaped, V_new_reshaped;
        reshapeQKVForCache(cmdBuff, tensors, Q_reshaped, K_new_reshaped, V_new_reshaped, B, new_S, H, HD);

        // Step 4: Concatenate with cache
        if (cache_len > 0) {
            // Store reshaped tensors for concatenation
            tensors.K_flat = K_new_reshaped;
            tensors.V_flat = V_new_reshaped;

            // Allocate full tensors
            BufferPool& pool = BufferPool::get();
            tensors.K_full = Tensor(B, H, total_S, HD);
            tensors.V_full = Tensor(B, H, total_S, HD);
            tensors.K_full.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*H*total_S*HD*sizeof(float)));
            tensors.V_full.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*H*total_S*HD*sizeof(float)));
            concatenateWithCache(cmdBuff, tensors, B, H, new_S, cache_len, HD);
        } else {
            // No cached data, use reshaped tensors directly (they already have buffers bound)
            tensors.K_full = K_new_reshaped;
            tensors.V_full = V_new_reshaped;
        }

        // Step 5: Compute attention with cached K, V
        // Flash Attention: Enabled (with barrier fix)
        if (new_S == 1) {
            // Flash Attention path (autoregressive only)
            flashAttnDescSet.write({
                tensors.context.buffer(),
                Q_reshaped.buffer(),
                tensors.K_full.buffer(),
                tensors.V_full.buffer()
            });

            float scale = 1.0f / std::sqrt(static_cast<float>(HD));
            struct {
                int B, H, S_q, S_kv, HD;
                float scale;
                int cache_len;
            } constants = {
                (int)B, (int)H, (int)new_S, (int)total_S, (int)HD, scale, (int)cache_len
            };

            // Dispatch: (B*H, S_q) workgroups, each with 32 threads (llama.cpp style)
            cmdBuff.bindPipeline(flashAttentionKVCache)
                   .setPushConstants(0, sizeof(constants), &constants)
                   .bindDescSets({flashAttnDescSet})
                   .dispatch0(B * H, new_S)
                   .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / tensors.context.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ));
        } else {
            // Standard 4-pass attention for batch processing
            computeAttentionScoresCached(cmdBuff, Q_reshaped, tensors.K_full, tensors.scores, B, H, new_S, total_S, HD);
            applyCausalMaskCached(cmdBuff, tensors.scores, B, H, new_S, total_S, cache_len);
            computeSoftmaxCached(cmdBuff, tensors, B, H, new_S, total_S);
            computeWeightedSumCached(cmdBuff, tensors, B, H, new_S, total_S, HD);
        }

        // Step 6: Combine heads and project (same as standard path)
        combineHeadsAndProject(cmdBuff, tensors, W_out, B_out, output, B, new_S, D_out, H, HD);

        // Step 7: Update cache with new K, V
        updateCacheWithNewKV(cmdBuff, K_new_reshaped, V_new_reshaped, B, H, new_S, cache_len, kv_cache->max_len, HD);

        // Note: Cache length will be updated by GPT2Net after forward pass completes
        // We don't update kv_cache->current_len here because the command buffer hasn't executed yet
    }
}

// ============================================================================
// Private Helper Functions for MultiHeadAttentionNode
// ============================================================================

MultiHeadAttentionNode::IntermediateTensors
MultiHeadAttentionNode::allocateIntermediateBuffers(uint32_t B, uint32_t S, uint32_t D, uint32_t H, uint32_t HD)
{
    BufferPool& pool = BufferPool::get();
    IntermediateTensors tensors;

    tensors.Q_flat = Tensor(B, S, D);
    tensors.K_flat = Tensor(B, S, D);
    tensors.V_flat = Tensor(B, S, D);
    tensors.Q_flat.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*S*D*sizeof(float)));
    tensors.K_flat.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*S*D*sizeof(float)));
    tensors.V_flat.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*S*D*sizeof(float)));

    tensors.scores = Tensor(B, H, S, S);
    tensors.scores.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*H*S*S*sizeof(float)));

    tensors.attn_weights = Tensor(B, H, S, S);
    tensors.attn_weights.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*H*S*S*sizeof(float)));

    tensors.context = Tensor(B, H, S, HD);
    tensors.context.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*H*S*HD*sizeof(float)));

    tensors.context_combined = Tensor(B, S, D);
    tensors.context_combined.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*S*D*sizeof(float)));

    return tensors;
}

MultiHeadAttentionNode::IntermediateTensors
MultiHeadAttentionNode::allocateIntermediateBuffersCached(uint32_t B, uint32_t new_S, uint32_t total_S, uint32_t D, uint32_t H, uint32_t HD)
{
    BufferPool& pool = BufferPool::get();
    IntermediateTensors tensors;

    // Q, K, V for new tokens only
    tensors.Q_flat = Tensor(B, new_S, D);
    tensors.K_flat = Tensor(B, new_S, D);
    tensors.V_flat = Tensor(B, new_S, D);
    tensors.Q_flat.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*new_S*D*sizeof(float)));
    tensors.K_flat.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*new_S*D*sizeof(float)));
    tensors.V_flat.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*new_S*D*sizeof(float)));

    // Scores and weights use total_S (cached + new)
    tensors.scores = Tensor(B, H, new_S, total_S);
    tensors.scores.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*H*new_S*total_S*sizeof(float)));

    tensors.attn_weights = Tensor(B, H, new_S, total_S);
    tensors.attn_weights.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*H*new_S*total_S*sizeof(float)));

    // Context uses new_S
    tensors.context = Tensor(B, H, new_S, HD);
    tensors.context.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*H*new_S*HD*sizeof(float)));

    tensors.context_combined = Tensor(B, new_S, D);
    tensors.context_combined.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*new_S*D*sizeof(float)));

    return tensors;
}

void MultiHeadAttentionNode::reshapeQKVForCache(CommandBuffer& cmdBuff, IntermediateTensors& tensors,
                                                 Tensor& Q_reshaped, Tensor& K_reshaped, Tensor& V_reshaped,
                                                 uint32_t B, uint32_t new_S, uint32_t H, uint32_t HD)
{
    BufferPool& pool = BufferPool::get();

    // Allocate reshaped tensors [B, H, new_S, HD]
    Q_reshaped = Tensor(B, H, new_S, HD);
    K_reshaped = Tensor(B, H, new_S, HD);
    V_reshaped = Tensor(B, H, new_S, HD);
    Q_reshaped.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*H*new_S*HD*sizeof(float)));
    K_reshaped.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*H*new_S*HD*sizeof(float)));
    V_reshaped.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*H*new_S*HD*sizeof(float)));

    // Reshape Q, K, V to multi-head format
    reshapeToHeads(cmdBuff, tensors.Q_flat, Q_reshaped, reshapeDescSetQ, B, new_S, H, HD);
    reshapeToHeads(cmdBuff, tensors.K_flat, K_reshaped, reshapeDescSetK, B, new_S, H, HD);
    reshapeToHeads(cmdBuff, tensors.V_flat, V_reshaped, reshapeDescSetV, B, new_S, H, HD);
}

void MultiHeadAttentionNode::computeQKVProjection(CommandBuffer& cmdBuff, const Tensor& input, IntermediateTensors& tensors,
                                                   const Tensor& W_q, const Tensor& W_k, const Tensor& W_v,
                                                   const Tensor& B_q, const Tensor& B_k, const Tensor& B_v,
                                                   uint32_t B, uint32_t S, uint32_t D_in, uint32_t D_out)
{
    uint32_t M = B * S;

    if (M == 1) {
        // GEMV path: Use subgroup-optimized kernel for M=1 (KV cache mode)
        qkvProjDescSetGEMV.write({
            tensors.Q_flat.buffer(), tensors.K_flat.buffer(), tensors.V_flat.buffer(),
            input.buffer(),
            W_q.buffer(), W_k.buffer(), W_v.buffer(),
            B_q.buffer(), B_k.buffer(), B_v.buffer()
        });

        int constants[] = {(int)M, (int)D_in, (int)D_out};  // M, K, N

        cmdBuff
            .bindPipeline(qkvProjectionGEMV)
            .setPushConstants(0, sizeof(constants), constants)
            .bindDescSets({qkvProjDescSetGEMV})
            .dispatch0(M, D_out)  // M×D_out workgroups (each uses 32 threads with subgroup ops)
            .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / tensors.Q_flat.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ))
            .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / tensors.K_flat.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ))
            .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / tensors.V_flat.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ));
    } else {
        // GEMM path: Use standard kernel for M>1
        qkvProjDescSet.write({
            tensors.Q_flat.buffer(), tensors.K_flat.buffer(), tensors.V_flat.buffer(),
            input.buffer(),
            W_q.buffer(), W_k.buffer(), W_v.buffer(),
            B_q.buffer(), B_k.buffer(), B_v.buffer()
        });

        int constants[] = {(int)B, (int)S, (int)D_in, (int)D_out};

        cmdBuff
            .bindPipeline(qkvProjection)
            .setPushConstants(0, sizeof(constants), constants)
            .bindDescSets({qkvProjDescSet})
            .dispatch0(CEIL_DIV(B*S, 16), CEIL_DIV(D_out, 16))
            .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / tensors.Q_flat.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ))
            .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / tensors.K_flat.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ))
            .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / tensors.V_flat.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ));
    }
}

void MultiHeadAttentionNode::computeAttentionScores(CommandBuffer& cmdBuff, IntermediateTensors& tensors,
                                                      uint32_t B, uint32_t H, uint32_t S, uint32_t HD)
{
    scoresDescSet.write({
        tensors.scores.buffer(),
        tensors.Q_flat.buffer(),
        tensors.K_flat.buffer()
    });

    float scale = 1.0f / std::sqrt(static_cast<float>(HD));
    struct { int B, H, S, HD; float scale; } constants = {(int)B, (int)H, (int)S, (int)HD, scale};

    cmdBuff
        .bindPipeline(attentionScores)
        .setPushConstants(0, sizeof(constants), &constants)
        .bindDescSets({scoresDescSet})
        .dispatch0(CEIL_DIV(B*H, 8), CEIL_DIV(S, 8))
        .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / tensors.scores.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ));
}

void MultiHeadAttentionNode::applyCausalMaskToScores(CommandBuffer& cmdBuff, IntermediateTensors& tensors,
                                                       uint32_t B, uint32_t H, uint32_t S)
{
    maskDescSet.write({tensors.scores.buffer()});

    int constants[] = {(int)B, (int)H, (int)S};

    cmdBuff
        .bindPipeline(applyCausalMask)
        .setPushConstants(0, sizeof(constants), constants)
        .bindDescSets({maskDescSet})
        .dispatch0(CEIL_DIV(B*H*S*S, 256))
        .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / tensors.scores.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ));
}

void MultiHeadAttentionNode::computeSoftmax(CommandBuffer& cmdBuff, IntermediateTensors& tensors,
                                              uint32_t B, uint32_t H, uint32_t S)
{
    softmaxDescSet.write({
        tensors.attn_weights.buffer(),
        tensors.scores.buffer()
    });

    int constants[] = {(int)(B * H * S), (int)S};

    cmdBuff
        .bindPipeline(softmaxPipeline)
        .setPushConstants(0, sizeof(constants), constants)
        .bindDescSets({softmaxDescSet})
        .dispatch0(CEIL_DIV(B * H * S, 64))
        .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / tensors.attn_weights.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ));
}

void MultiHeadAttentionNode::computeWeightedSum(CommandBuffer& cmdBuff, IntermediateTensors& tensors,
                                                  uint32_t B, uint32_t H, uint32_t S, uint32_t HD)
{
    weightedSumDescSet.write({
        tensors.context.buffer(),
        tensors.attn_weights.buffer(),
        tensors.V_flat.buffer()
    });

    int constants[] = {(int)B, (int)H, (int)S, (int)HD};

    cmdBuff
        .bindPipeline(weightedSum)
        .setPushConstants(0, sizeof(constants), constants)
        .bindDescSets({weightedSumDescSet})
        .dispatch0(CEIL_DIV(B*H, 8), CEIL_DIV(S, 8))
        .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / tensors.context.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ));
}

void MultiHeadAttentionNode::combineHeadsAndProject(CommandBuffer& cmdBuff, IntermediateTensors& tensors,
                                                      const Tensor& W_out, const Tensor& B_out, Tensor& output,
                                                      uint32_t B, uint32_t S, uint32_t D, uint32_t H, uint32_t HD)
{
    // Combine heads
    combineDescSet.write({
        tensors.context_combined.buffer(),
        tensors.context.buffer()
    });

    int constants1[] = {(int)B, (int)H, (int)S, (int)HD};

    cmdBuff
        .bindPipeline(combineHeads)
        .setPushConstants(0, sizeof(constants1), constants1)
        .bindDescSets({combineDescSet})
        .dispatch0(CEIL_DIV(B*S, 16), CEIL_DIV(D, 16))
        .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / tensors.context_combined.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ));

    // Output projection
    uint32_t M = B * S;

    if (M == 1) {
        // GEMV path: Use subgroup-optimized kernel for M=1
        outProjDescSetGEMV.write({
            output.buffer(),
            tensors.context_combined.buffer(),
            W_out.buffer(),
            B_out.buffer()
        });

        int constants2[] = {(int)M, (int)D, (int)d_out};  // M, I, O

        cmdBuff
            .bindPipeline(outputProjectionGEMV)
            .setPushConstants(0, sizeof(constants2), constants2)
            .bindDescSets({outProjDescSetGEMV})
            .dispatch0(M, d_out)  // M×d_out workgroups (each uses 32 threads with subgroup ops)
            .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / output.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ));
    } else {
        // GEMM path: Use standard kernel for M>1
        outProjDescSet.write({
            output.buffer(),
            tensors.context_combined.buffer(),
            W_out.buffer(),
            B_out.buffer()
        });

        int constants2[] = {(int)B, (int)S, (int)D, (int)d_out};

        cmdBuff
            .bindPipeline(outputProjection)
            .setPushConstants(0, sizeof(constants2), constants2)
            .bindDescSets({outProjDescSet})
            .dispatch0(CEIL_DIV(B*S, 16), CEIL_DIV(d_out, 16))
            .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / output.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ));
    }
}

// ============================================================================
// KV Cache Support Methods
// ============================================================================

void MultiHeadAttentionNode::setCache(LayerKVCache* cache)
{
    kv_cache = cache;
    use_cache = (cache != nullptr);
}

void MultiHeadAttentionNode::disableCache()
{
    kv_cache = nullptr;
    use_cache = false;
}

void MultiHeadAttentionNode::reshapeToHeads(CommandBuffer& cmdBuff, const Tensor& flat, Tensor& reshaped, DescriptorSet& descSet,
                                            uint32_t B, uint32_t S, uint32_t H, uint32_t HD)
{
    // Use pre-allocated descriptor set (passed as parameter)
    descSet.write({
        reshaped.buffer(),
        flat.buffer()
    });

    int constants[] = {(int)B, (int)S, (int)H, (int)HD};
    uint32_t D = H * HD;

    cmdBuff
        .bindPipeline(reshapeForHeads)
        .setPushConstants(0, sizeof(constants), constants)
        .bindDescSets({descSet})
        .dispatch0(CEIL_DIV(B*S, 16), CEIL_DIV(D, 16))
        .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / reshaped.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ));
}

void MultiHeadAttentionNode::concatenateWithCache(CommandBuffer& cmdBuff, IntermediateTensors& tensors,
                                                   uint32_t B, uint32_t H, uint32_t new_S, uint32_t cache_len, uint32_t HD)
{
    _ASSERT(kv_cache != nullptr);
    _ASSERT(use_cache);

    uint32_t total_len = cache_len + new_S;
    uint32_t max_len = kv_cache->max_len;

    // Note: This version expects K_new and V_new to already be in tensors
    // It will concatenate them with the cache

    // Concatenate K: [cached_K, K_new] → K_full (use pre-allocated descriptor set)
    concatDescSetK.write({
        tensors.K_full.buffer(),
        kv_cache->K.buffer(),
        tensors.K_flat.buffer()  // K_new in [B, H, new_S, HD] format (assigned from K_new_reshaped)
    });

    int k_constants[] = {(int)B, (int)H, (int)cache_len, (int)new_S, (int)max_len, (int)HD};

    cmdBuff
        .bindPipeline(concatenateKV)
        .setPushConstants(0, sizeof(k_constants), k_constants)
        .bindDescSets({concatDescSetK})
        .dispatch0(CEIL_DIV(B * H, 16), CEIL_DIV(total_len, 16))
        .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / tensors.K_full.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ));

    // Concatenate V: [cached_V, V_new] → V_full (use pre-allocated descriptor set)
    concatDescSetV.write({
        tensors.V_full.buffer(),
        kv_cache->V.buffer(),
        tensors.V_flat.buffer()  // V_new in [B, H, new_S, HD] format (assigned from V_new_reshaped)
    });

    cmdBuff
        .bindPipeline(concatenateKV)
        .setPushConstants(0, sizeof(k_constants), k_constants)
        .bindDescSets({concatDescSetV})
        .dispatch0(CEIL_DIV(B * H, 16), CEIL_DIV(total_len, 16))
        .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / tensors.V_full.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ));
}

void MultiHeadAttentionNode::updateCacheWithNewKV(CommandBuffer& cmdBuff, const Tensor& K_new, const Tensor& V_new,
                                                   uint32_t B, uint32_t H, uint32_t new_S, uint32_t cache_offset, uint32_t max_len, uint32_t HD)
{
    _ASSERT(kv_cache != nullptr);
    _ASSERT(use_cache);

    // Update K cache (use pre-allocated descriptor set)
    updateCacheDescSetK.write({
        kv_cache->K.buffer(),
        K_new.buffer()
    });

    int constants[] = {(int)B, (int)H, (int)cache_offset, (int)new_S, (int)max_len, (int)HD};

    cmdBuff
        .bindPipeline(updateCache)
        .setPushConstants(0, sizeof(constants), constants)
        .bindDescSets({updateCacheDescSetK})
        .dispatch0(CEIL_DIV(B * H, 16), CEIL_DIV(new_S, 16))
        .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / kv_cache->K.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ));

    // Update V cache (use pre-allocated descriptor set)
    updateCacheDescSetV.write({
        kv_cache->V.buffer(),
        V_new.buffer()
    });

    cmdBuff
        .bindPipeline(updateCache)
        .setPushConstants(0, sizeof(constants), constants)
        .bindDescSets({updateCacheDescSetV})
        .dispatch0(CEIL_DIV(B * H, 16), CEIL_DIV(new_S, 16))
        .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / kv_cache->V.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ));
}

void MultiHeadAttentionNode::computeAttentionScoresCached(CommandBuffer& cmdBuff, const Tensor& Q, const Tensor& K, Tensor& scores,
                                                          uint32_t B, uint32_t H, uint32_t S_q, uint32_t S_kv, uint32_t HD)
{
    // Use member descriptor set (OK because this function is called only once per layer per forward pass)
    scoresCachedDescSet.write({
        scores.buffer(),
        Q.buffer(),
        K.buffer()
    });

    float scale = 1.0f / std::sqrt(static_cast<float>(HD));
    struct { int B, H, S_q, S_kv, HD; float scale; } constants = {(int)B, (int)H, (int)S_q, (int)S_kv, (int)HD, scale};

    cmdBuff
        .bindPipeline(scoresPipelineCached)
        .setPushConstants(0, sizeof(constants), &constants)
        .bindDescSets({scoresCachedDescSet})
        .dispatch0(CEIL_DIV(B*H, 8), CEIL_DIV(S_q, 8))
        .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / scores.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ));
}

void MultiHeadAttentionNode::applyCausalMaskCached(CommandBuffer& cmdBuff, Tensor& scores,
                                                    uint32_t B, uint32_t H, uint32_t S_q, uint32_t S_kv, uint32_t cache_len)
{
    // Use member descriptor set (OK because this function is called only once per layer per forward pass)
    maskCachedDescSet.write({scores.buffer()});

    int constants[] = {(int)B, (int)H, (int)S_q, (int)S_kv, (int)cache_len};

    cmdBuff
        .bindPipeline(maskPipelineCached)
        .setPushConstants(0, sizeof(constants), constants)
        .bindDescSets({maskCachedDescSet})
        .dispatch0(CEIL_DIV(B*H*S_q*S_kv, 256))
        .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / scores.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ));
}

void MultiHeadAttentionNode::computeSoftmaxCached(CommandBuffer& cmdBuff, IntermediateTensors& tensors,
                                                   uint32_t B, uint32_t H, uint32_t S_q, uint32_t S_kv)
{
    softmaxDescSet.write({
        tensors.attn_weights.buffer(),
        tensors.scores.buffer()
    });

    int constants[] = {(int)(B * H * S_q), (int)S_kv};

    cmdBuff
        .bindPipeline(softmaxPipeline)
        .setPushConstants(0, sizeof(constants), constants)
        .bindDescSets({softmaxDescSet})
        .dispatch0(CEIL_DIV(B * H * S_q, 64))
        .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / tensors.attn_weights.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ));
}

void MultiHeadAttentionNode::computeWeightedSumCached(CommandBuffer& cmdBuff, IntermediateTensors& tensors,
                                                       uint32_t B, uint32_t H, uint32_t S_q, uint32_t S_kv, uint32_t HD)
{
    // Use member descriptor set (OK because this function is called only once per layer per forward pass)
    weightedSumCachedDescSet.write({
        tensors.context.buffer(),
        tensors.attn_weights.buffer(),
        tensors.V_full.buffer()
    });

    int constants[] = {(int)B, (int)H, (int)S_q, (int)S_kv, (int)HD};

    cmdBuff
        .bindPipeline(weightedSumPipelineCached)
        .setPushConstants(0, sizeof(constants), constants)
        .bindDescSets({weightedSumCachedDescSet})
        .dispatch0(CEIL_DIV(B*H, 8), CEIL_DIV(S_q, 8))
        .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / tensors.context.buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ));
}

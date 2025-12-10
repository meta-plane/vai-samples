#include "transformer.h"
#include "../../core/globalContext.h"
#include "../attention/attentionNode.h"
#include "../../core/error.h"
#include <cmath>
#include <unordered_map>

using namespace vk;

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))


// ============================================================================
// LayerNormNode: output = scale * (x - mean) / sqrt(var + eps) + shift
// ============================================================================

const char* src_layer_norm = R"(
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer Output { float y[]; };      // [B*S*D]
layout(set = 0, binding = 1) buffer Input { float x[]; };       // [B*S*D]
layout(set = 0, binding = 2) buffer Scale { float scale[]; };   // [D]
layout(set = 0, binding = 3) buffer Shift { float shift[]; };   // [D]

layout(push_constant) uniform PushConstants {
    int num_rows;  // B * S
    int D;         // d_model
    float eps;
};

void main() {
    int row = int(gl_GlobalInvocationID.x);
    if (row >= num_rows) return;

    int offset = row * D;

    // Compute mean
    float sum = 0.0;
    for (int i = 0; i < D; ++i) {
        sum += x[offset + i];
    }
    float mean = sum / float(D);

    // Compute variance
    float var_sum = 0.0;
    for (int i = 0; i < D; ++i) {
        float diff = x[offset + i] - mean;
        var_sum += diff * diff;
    }
    float variance = var_sum / float(D);

    // Normalize and apply scale/shift
    float inv_std = 1.0 / sqrt(variance + eps);
    for (int i = 0; i < D; ++i) {
        float norm_val = (x[offset + i] - mean) * inv_std;
        y[offset + i] = scale[i] * norm_val + shift[i];
    }
}
)";

LayerNormNode::LayerNormNode(uint32_t normalized_shape, float eps)
    : normalized_shape(normalized_shape), eps(eps)
{
    addSlot("in0", NodeSlot::input);
    addSlot("scale", NodeSlot::input);  // Learnable parameter
    addSlot("shift", NodeSlot::input);  // Learnable parameter
    addSlot("out0", NodeSlot::output);

    layerNormPipeline = requestPipeline(src_layer_norm);
    layerNormDescSet = layerNormPipeline.descSetLayout(0).newDescSet(gDestSetPool);
}

void LayerNormNode::prepare()
{
    Tensor& input = (*this)["in0"];
    _ASSERT(input.validShape());
    _ASSERT(input.shape().size() == 3);  // [B, S, D]

    uint32_t B = input.shape()[0];
    uint32_t S = input.shape()[1];
    uint32_t D = input.shape()[2];
    _ASSERT(D == normalized_shape);

    // Initialize scale and shift if not set
    Tensor& scale = (*this)["scale"];
    Tensor& shift = (*this)["shift"];

    if (!scale.validShape()) {
        scale = Tensor(normalized_shape);
        // Initialize scale to 1.0
        std::vector<float> scale_data(normalized_shape, 1.0f);
        scale.set(scale_data);
    }

    if (!shift.validShape()) {
        shift = Tensor(normalized_shape);
        // Initialize shift to 0.0
        std::vector<float> shift_data(normalized_shape, 0.0f);
        shift.set(shift_data);
    }

    (*this)["out0"] = Tensor(B, S, D);
}

void LayerNormNode::run(CommandBuffer cmdBuff)
{
    Tensor& input = (*this)["in0"];
    Tensor& scale = (*this)["scale"];
    Tensor& shift = (*this)["shift"];
    Tensor& output = (*this)["out0"];

    uint32_t B = input.shape()[0];
    uint32_t S = input.shape()[1];
    uint32_t D = input.shape()[2];
    uint32_t num_rows = B * S;

    layerNormDescSet.write({
        output.buffer(),
        input.buffer(),
        scale.buffer(),
        shift.buffer()
    });

    struct { int num_rows, D; float eps; } constants = {(int)num_rows, (int)D, eps};

    cmdBuff
        .bindPipeline(layerNormPipeline)
        .setPushConstants(0, sizeof(constants), &constants)
        .bindDescSets({layerNormDescSet})
        .dispatch0(CEIL_DIV(num_rows, 256))
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / output.buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}

// ============================================================================
// GELUNode: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// ============================================================================

const char* src_gelu = R"(
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer Output { float y[]; };
layout(set = 0, binding = 1) buffer Input { float x[]; };

layout(push_constant) uniform PushConstants {
    int N;  // Total number of elements
};

void main() {
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= N) return;

    float val = x[idx];

    // GELU approximation using tanh
    // sqrt(2/pi) ≈ 0.7978845608
    const float sqrt_2_over_pi = 0.7978845608;
    const float coeff = 0.044715;

    float inner = sqrt_2_over_pi * (val + coeff * val * val * val);
    float gelu = 0.5 * val * (1.0 + tanh(inner));

    y[idx] = gelu;
}
)";

GELUNode::GELUNode()
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);

    geluPipeline = requestPipeline(src_gelu);
    geluDescSet = geluPipeline.descSetLayout(0).newDescSet(gDestSetPool);
}

void GELUNode::prepare()
{
    Tensor& input = (*this)["in0"];
    _ASSERT(input.validShape());

    // Output has same shape as input
    (*this)["out0"] = Tensor(input.shape());
}

void GELUNode::run(CommandBuffer cmdBuff)
{
    Tensor& input = (*this)["in0"];
    Tensor& output = (*this)["out0"];

    uint32_t N = input.numElements();

    geluDescSet.write({
        output.buffer(),
        input.buffer()
    });

    int constants[] = {(int)N};

    cmdBuff
        .bindPipeline(geluPipeline)
        .setPushConstants(0, sizeof(constants), constants)
        .bindDescSets({geluDescSet})
        .dispatch0(CEIL_DIV(N, 256))
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / output.buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}

// ============================================================================
// FeedForwardNode: Linear(d -> 4d) -> GELU -> Linear(4d -> d)
// ============================================================================

const char* src_linear_ff = R"(
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

const char* src_linear_gemv = R"(
#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_ballot : enable

// Workgroup size = subgroup size (32 for NVIDIA, 64 for AMD)
layout(local_size_x = 32, local_size_y = 1) in;

layout(set = 0, binding = 0) buffer Output { float y[]; };    // [M*N]
layout(set = 0, binding = 1) buffer Input { float x[]; };     // [M*K]
layout(set = 0, binding = 2) buffer Weight { float w[]; };    // [N*K]
layout(set = 0, binding = 3) buffer Bias { float b[]; };      // [N]

layout(push_constant) uniform PushConstants {
    int M;   // batch_size * seq_len (usually 1 in KV cache)
    int K;   // input features
    int N;   // output features
};

void main() {
    uint tid = gl_LocalInvocationID.x;         // Thread ID within workgroup [0, 31]
    uint row_idx = gl_WorkGroupID.x;           // Which row (M dimension)
    uint col_idx = gl_WorkGroupID.y;           // Which output feature (N dimension)

    if (row_idx >= M || col_idx >= N) return;

    // Phase 1: Each thread computes partial dot product over strided K
    float partial_sum = 0.0;
    for (uint k = tid; k < K; k += 32) {
        partial_sum += x[row_idx * K + k] * w[col_idx * K + k];
    }

    // Phase 2: Subgroup reduction (hardware-accelerated!)
    // All 32 threads in the subgroup sum their values in parallel
    float total_sum = subgroupAdd(partial_sum);

    // Phase 3: First thread (subgroup representative) writes result
    if (subgroupElect()) {
        y[row_idx * N + col_idx] = total_sum + b[col_idx];
    }
}
)";

FeedForwardNode::FeedForwardNode(uint32_t d_model)
    : d_model(d_model), hidden_dim(4 * d_model)
{
    addSlot("in0", NodeSlot::input);
    addSlot("weight1", NodeSlot::input);  // [4*d_model, d_model] (learnable parameter)
    addSlot("weight2", NodeSlot::input);  // [d_model, 4*d_model] (learnable parameter)
    addSlot("bias1", NodeSlot::input);    // [4*d_model] - NEW: bias parameter
    addSlot("bias2", NodeSlot::input);    // [d_model] - NEW: bias parameter
    addSlot("out0", NodeSlot::output);

    linear1Pipeline = requestPipeline(src_linear_ff);
    linear1PipelineGEMV = requestPipeline(src_linear_gemv);
    geluPipeline = requestPipeline(src_gelu);
    linear2Pipeline = requestPipeline(src_linear_ff);
    linear2PipelineGEMV = requestPipeline(src_linear_gemv);

    linear1DescSet = linear1Pipeline.descSetLayout(0).newDescSet(gDestSetPool);
    linear1DescSetGEMV = linear1PipelineGEMV.descSetLayout(0).newDescSet(gDestSetPool);
    geluDescSet = geluPipeline.descSetLayout(0).newDescSet(gDestSetPool);
    linear2DescSet = linear2Pipeline.descSetLayout(0).newDescSet(gDestSetPool);
    linear2DescSetGEMV = linear2PipelineGEMV.descSetLayout(0).newDescSet(gDestSetPool);
}

void FeedForwardNode::prepare()
{
    Tensor& input = (*this)["in0"];
    _ASSERT(input.validShape());
    _ASSERT(input.shape().size() == 3);  // [B, S, D]

    uint32_t B = input.shape()[0];
    uint32_t S = input.shape()[1];
    uint32_t D = input.shape()[2];
    _ASSERT(D == d_model);

    // Initialize weights if not set
    Tensor& weight1 = (*this)["weight1"];
    Tensor& weight2 = (*this)["weight2"];

    if (!weight1.validShape()) {
        weight1 = Tensor(hidden_dim, d_model);
    }

    if (!weight2.validShape()) {
        weight2 = Tensor(d_model, hidden_dim);
    }

    // Initialize biases if not set
    Tensor& bias1 = (*this)["bias1"];
    Tensor& bias2 = (*this)["bias2"];

    if (!bias1.validShape()) {
        bias1 = Tensor(hidden_dim);
    }

    if (!bias2.validShape()) {
        bias2 = Tensor(d_model);
    }

    (*this)["out0"] = Tensor(B, S, D);
}

// ==================== FeedForwardNode Helper Functions ====================

void FeedForwardNode::runLinear1(CommandBuffer& cmdBuff, const Tensor& input, Tensor& hidden,
                                   const Tensor& weight1, const Tensor& bias1,
                                   uint32_t B, uint32_t S, uint32_t D, uint32_t H)
{
    uint32_t M = B * S;
    static bool first_call = true;

    if (M == 1) {
        // GEMV path: Use optimized kernel for M=1
        if (first_call) {
            printf("[DEBUG] FeedForwardNode::runLinear1 using GEMV path (M=%u, K=%u, N=%u)\n", M, D, H);
            first_call = false;
        }
        linear1DescSetGEMV.write({
            hidden.buffer(),
            input.buffer(),
            weight1.buffer(),
            bias1.buffer()
        });

        int constants[] = {(int)M, (int)D, (int)H};  // M, K, N

        cmdBuff
            .bindPipeline(linear1PipelineGEMV)
            .setPushConstants(0, sizeof(constants), constants)
            .bindDescSets({linear1DescSetGEMV})
            .dispatch0(M, H)  // Dispatch M×H workgroups (each uses 32 threads with subgroup operations)
            .barrier(
                (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
                / hidden.buffer()
                / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
            );
    } else {
        // GEMM path: Use naive kernel for M>1 (fallback)
        linear1DescSet.write({
            hidden.buffer(),
            input.buffer(),
            weight1.buffer(),
            bias1.buffer()
        });

        int constants[] = {(int)B, (int)S, (int)D, (int)H};

        cmdBuff
            .bindPipeline(linear1Pipeline)
            .setPushConstants(0, sizeof(constants), constants)
            .bindDescSets({linear1DescSet})
            .dispatch0(CEIL_DIV(M, 16), CEIL_DIV(H, 16))
            .barrier(
                (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
                / hidden.buffer()
                / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
            );
    }
}

void FeedForwardNode::runGELU(CommandBuffer& cmdBuff, const Tensor& hidden, Tensor& gelu_out,
                               uint32_t B, uint32_t S, uint32_t H)
{
    geluDescSet.write({
        gelu_out.buffer(),
        hidden.buffer()
    });

    int constants[] = {(int)(B * S * H)};

    cmdBuff
        .bindPipeline(geluPipeline)
        .setPushConstants(0, sizeof(constants), constants)
        .bindDescSets({geluDescSet})
        .dispatch0(CEIL_DIV(B * S * H, 256))
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / gelu_out.buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}

void FeedForwardNode::runLinear2(CommandBuffer& cmdBuff, const Tensor& gelu_out, Tensor& output,
                                   const Tensor& weight2, const Tensor& bias2,
                                   uint32_t B, uint32_t S, uint32_t H, uint32_t D)
{
    uint32_t M = B * S;

    if (M == 1) {
        // GEMV path: Use optimized kernel for M=1
        linear2DescSetGEMV.write({
            output.buffer(),
            gelu_out.buffer(),
            weight2.buffer(),
            bias2.buffer()
        });

        int constants[] = {(int)M, (int)H, (int)D};  // M, K, N

        cmdBuff
            .bindPipeline(linear2PipelineGEMV)
            .setPushConstants(0, sizeof(constants), constants)
            .bindDescSets({linear2DescSetGEMV})
            .dispatch0(M, D)  // Dispatch M×D workgroups (each uses 32 threads with subgroup operations)
            .barrier(
                (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
                / output.buffer()
                / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
            );
    } else {
        // GEMM path: Use naive kernel for M>1 (fallback)
        linear2DescSet.write({
            output.buffer(),
            gelu_out.buffer(),
            weight2.buffer(),
            bias2.buffer()
        });

        int constants[] = {(int)B, (int)S, (int)H, (int)D};

        cmdBuff
            .bindPipeline(linear2Pipeline)
            .setPushConstants(0, sizeof(constants), constants)
            .bindDescSets({linear2DescSet})
            .dispatch0(CEIL_DIV(M, 16), CEIL_DIV(D, 16))
            .barrier(
                (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
                / output.buffer()
                / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
            );
    }
}

// ==================== FeedForwardNode Main Run ====================

void FeedForwardNode::run(CommandBuffer cmdBuff)
{
    Tensor& input = (*this)["in0"];
    Tensor& weight1 = (*this)["weight1"];
    Tensor& weight2 = (*this)["weight2"];
    Tensor& bias1 = (*this)["bias1"];
    Tensor& bias2 = (*this)["bias2"];
    Tensor& output = (*this)["out0"];

    uint32_t B = input.shape()[0];
    uint32_t S = input.shape()[1];
    uint32_t D = d_model;
    uint32_t H = hidden_dim;

    // Allocate temporary buffers
    BufferPool& pool = BufferPool::get();
    Tensor hidden(B, S, H);
    hidden.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*S*H*sizeof(float)));

    Tensor gelu_out(B, S, H);
    gelu_out.bindBuffer(pool.requestBuffer(netGlobalDevice, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, B*S*H*sizeof(float)));

    // Execute feedforward pipeline
    runLinear1(cmdBuff, input, hidden, weight1, bias1, B, S, D, H);
    runGELU(cmdBuff, hidden, gelu_out, B, S, H);
    runLinear2(cmdBuff, gelu_out, output, weight2, bias2, B, S, H, D);
}

// ============================================================================
// IdentityNode: Pass-through node for fan-out
// ============================================================================

IdentityNode::IdentityNode()
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
}

void IdentityNode::prepare()
{
    // Output tensor is the same as input tensor (no computation needed)
    Tensor& input = (*this)["in0"];
    Tensor& output = (*this)["out0"];
    output = input;  // Share the same tensor
}

void IdentityNode::run(CommandBuffer cmdBuff)
{
    // No computation needed - input and output share the same buffer
}

// ============================================================================
// AddNode: Element-wise addition for residual connections
// ============================================================================

const char* src_add = R"(
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer Output { float y[]; };   // output
layout(set = 0, binding = 1) buffer Input0 { float a[]; };   // in0 (residual)
layout(set = 0, binding = 2) buffer Input1 { float b[]; };   // in1 (main path)

layout(push_constant) uniform PushConstants {
    int N;  // Total number of elements
};

void main() {
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= N) return;

    y[idx] = a[idx] + b[idx];
}
)";

AddNode::AddNode()
{
    addSlot("in0", NodeSlot::input);   // Residual connection
    addSlot("in1", NodeSlot::input);   // Main path
    addSlot("out0", NodeSlot::output);

    addPipeline = requestPipeline(src_add);
    addDescSet = addPipeline.descSetLayout(0).newDescSet(gDestSetPool);
}

void AddNode::prepare()
{
    Tensor& in0 = (*this)["in0"];
    Tensor& in1 = (*this)["in1"];
    _ASSERT(in0.validShape());
    _ASSERT(in1.validShape());
    _ASSERT(in0.shape() == in1.shape());

    (*this)["out0"] = Tensor(in0.shape());
}

void AddNode::run(CommandBuffer cmdBuff)
{
    Tensor& in0 = (*this)["in0"];
    Tensor& in1 = (*this)["in1"];
    Tensor& output = (*this)["out0"];

    uint32_t N = in0.numElements();

    addDescSet.write({
        output.buffer(),
        in0.buffer(),
        in1.buffer()
    });

    int constants[] = {(int)N};

    cmdBuff
        .bindPipeline(addPipeline)
        .setPushConstants(0, sizeof(constants), constants)
        .bindDescSets({addDescSet})
        .dispatch0(CEIL_DIV(N, 256))
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / output.buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}

// ============================================================================
// TransformerBlock: NodeGroup with residual connections
// ============================================================================

TransformerBlock::TransformerBlock(uint32_t d_model, uint32_t num_heads)
    : NodeGroup(),
      d_model(d_model),
      num_heads(num_heads),
      norm1(d_model),
      attention(d_model, d_model, num_heads),
      norm2(d_model),
      feedforward(d_model)
{
    // Build graph: x = x + Attention(LayerNorm(x))
    //              x = x + FeedForward(LayerNorm(x))
    //
    // Graph structure:
    // input --> inputRouter --> norm1 --> attention --> add1 --> norm2 --> feedforward --> add2 --> output
    //              |                                     ^         |                        ^
    //              +-------------------------------------+         +------------------------+
    //                   (first residual skip)                    (second residual skip)

    // Fan out input to main path and first residual
    inputRouter - norm1;            // Main path: input -> norm1
    inputRouter - "in0" / add1;     // First residual: input -> add1.in0

    // First sub-path: norm1 -> attention -> add1.in1 (main path)
    norm1 - attention - "in1" / add1;

    // Second sub-path: add1 -> norm2 -> feedforward -> add2.in1 (main path)
    add1 - norm2 - feedforward - "in1" / add2;

    // Second residual connection: add1.out0 -> add2.in0
    add1 - "in0" / add2;

    // Define external slots
    defineSlot("in0", inputRouter.slot("in0"));  // External input goes to inputRouter
    defineSlot("out0", add2.slot("out0"));       // Final output comes from add2
}

Tensor& TransformerBlock::operator[](const std::string& name)
{
    // Provide access to internal weights
    if (name == "norm1_scale") return norm1["scale"];
    if (name == "norm1_shift") return norm1["shift"];
    if (name == "attn_wq") return attention["W_query"];
    if (name == "attn_wk") return attention["W_key"];
    if (name == "attn_wv") return attention["W_value"];
    if (name == "attn_wout") return attention["W_out"];
    if (name == "attn_bq") return attention["B_query"];
    if (name == "attn_bk") return attention["B_key"];
    if (name == "attn_bv") return attention["B_value"];
    if (name == "attn_bout") return attention["B_out"];
    if (name == "norm2_scale") return norm2["scale"];
    if (name == "norm2_shift") return norm2["shift"];
    if (name == "ff_w1") return feedforward["weight1"];
    if (name == "ff_w2") return feedforward["weight2"];
    if (name == "ff_b1") return feedforward["bias1"];
    if (name == "ff_b2") return feedforward["bias2"];

    throw std::runtime_error("No such weight in TransformerBlock: " + name);
}

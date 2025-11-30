#include "neuralNodes.h"
#include "vulkanApp.h"
#include <unordered_map>
#include <vector>
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))


// ... (Keep existing generic shaders: src_relu, src_setZero, src_maxpool, src_im2col, src_gemm_naive, src_gemm_kSplit, src_gemm_shared) ...
// To save context window, I am omitting the full re-declaration of existing shaders if I can avoid it, 
// BUT `write` overwrites the file. So I MUST include EVERYTHING.
// I will copy the previous content precisely.

static const char* src_relu = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in0[]; };
layout(push_constant) uniform PushConstants {
    int O;
};

void main() 
{
    int o = int(gl_GlobalInvocationID.x);
    if (o >= O) return;
    out0[o] = max(in0[o], 0.0f);
})";

static const char* src_setZero = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(push_constant) uniform PushConstants {
    int O;
};

void main() 
{
    int o = int(gl_GlobalInvocationID.x);
    if (o >= O) return;
    out0[o] = 0.0;
})";

static const char* src_maxpool = R"(
#version 450
#define FLT_MIN -3.402823466e+38
#define DISCARD_TAIL
layout(local_size_x = 64, local_size_y = 4, local_size_z = 4) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in0[]; };
layout(push_constant) uniform PushConstants {
    int H;      // input height
    int W;      // input width
    int C;
    int P;      // pooling size
};

void main()
{
    int h_ = int(gl_GlobalInvocationID.x);  // output row
    int w_ = int(gl_GlobalInvocationID.y);  // output col
    int c = int(gl_GlobalInvocationID.z);   // channel
#ifdef DISCARD_TAIL
    int H_ = H / P;  
    int W_ = W / P;  
#else
    int H_ = (H + P - 1) / P;
    int W_ = (W + P - 1) / P;
#endif
    if (h_ >= H_ || w_ >= W_ || c >= C)
        return;

    int h0 = h_ * P;  
    int w0 = w_ * P;     
    float maxVal = FLT_MIN;
    for (int dh=0; dh < P; ++dh) 
    {
        int h = h0 + dh;  
        for (int dw=0; dw < P; ++dw) 
        {
            int w = w0 + dw;

        #ifndef DISCARD_TAIL
            if (h < H && w < W) 
        #endif
            {
                maxVal = max(maxVal, in0[(h * W + w) * C + c]);
            }
        }
    }
    out0[(h_ * W_ + w_) * C + c] = maxVal;
})";


static const char* src_im2col = R"(
#version 450
layout(local_size_x = 32, local_size_y = 32) in;
layout(set = 0, binding = 0) buffer OutBuffer { float im2colOut[]; };
layout(set = 0, binding = 1) readonly buffer InBuffer { float in0[]; };

layout(push_constant) uniform PushConstants {
    int H;      // Input Height
    int W;      // Input Width
    int C;      // Channels
    int K;      // Kernel Size
    int Stride; // Stride
    int Padding;// Padding
    int H_out;  // Output Height
    int W_out;  // Output Width
};

void main() 
{
    int i = int(gl_GlobalInvocationID.x); 
    int j = int(gl_GlobalInvocationID.y); 
    int KK = K * K;
    int CKK = C * KK;
    
    if (i >= H_out * W_out || j >= CKK) 
        return;

    int h_out = i / W_out;
    int w_out = i % W_out;
    int c = j / KK;
    int k = j % KK;

    int h_in = h_out * Stride + (k / K) - Padding;
    int w_in = w_out * Stride + (k % K) - Padding;

    float value = 0.0;
    if (0 <= h_in && h_in < H && 0 <= w_in && w_in < W) 
        value = in0[((h_in * W) + w_in) * C + c];

    im2colOut[i * CKK + j] = value;
})";

static const char* src_gemm_naive = R"(
#version 450
layout(local_size_x = 32, local_size_y = 32) in;
layout(set = 0, binding = 0) buffer OutBuffer { float C[]; };
layout(set = 0, binding = 1) buffer InBuffer { float A[]; };
layout(set = 0, binding = 2) buffer Weight { float B[]; };
layout(set = 0, binding = 3) buffer Bias { float b[]; };

// C(MxN) = A(MxK)*B(KxN) + b(1xN)
layout(push_constant) uniform PushConstants {
    int M;  // # of batchs
    int K;  // # of inputs
    int N;  // # of outputs
};

void main() 
{
    int n = int(gl_GlobalInvocationID.x); 
    int m = int(gl_GlobalInvocationID.y); 

    if (m >= M || n >= N) 
        return;

    float sum = b[n];
    for (int k = 0; k < K; ++k)
        sum += A[m * K + k] * B[k * N + n];

    C[m * N + n] = sum;
}
)";

static const char* src_gemm_kSplit = R"(
#version 450
#define P 16
#extension GL_EXT_shader_atomic_float : require
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) buffer OutBuffer { float C[]; };
layout(set = 0, binding = 1) buffer InBuffer { float A[]; };
layout(set = 0, binding = 2) buffer Weight   { float B[]; };
layout(set = 0, binding = 3) buffer Bias     { float b[]; };

layout(push_constant) uniform PushConstants {
    int M;  // batch
    int K;  // input size
    int N;  // output size
};

void main()
{
    int n = int(gl_GlobalInvocationID.x); // output column
    int m = int(gl_GlobalInvocationID.y); // output row
    int Pid = int(gl_GlobalInvocationID.z); // split index along K

    if (n >= N) return; 

    int k0 = Pid * P;
    float sum = (Pid==0) ? b[n] : 0.0;
    for (int p = 0; p < P; ++p) 
    {
        int k = k0 + p;
        if (k >= K) 
            break;
        sum += A[m * K + k] * B[k * N + n];
    }

    atomicAdd(C[m * N + n], sum);
}
)";

static const char* src_gemm_shared = R"(
#version 450
layout(local_size_x = 32, local_size_y = 32) in;
layout(set = 0, binding = 0) buffer OutBuffer { float C[]; };
layout(set = 0, binding = 1) buffer InBuffer { float A[]; };
layout(set = 0, binding = 2) buffer Weight { float B[]; };
layout(set = 0, binding = 3) buffer Bias { float b[]; };

// C(MxN) = A(MxK)*B(KxN) + b(1xN)  
layout(push_constant) uniform PushConstants {
    int M;  // # of batchs
    int K;  // # of inputs
    int N;  // # of outputs
};

shared float As[32 * 32];
shared float Bs[32 * 32];

void main() 
{
    int n = int(gl_GlobalInvocationID.x); 
    int m = int(gl_GlobalInvocationID.y); 
    int _n = int(gl_LocalInvocationID.x); 
    int _m = int(gl_LocalInvocationID.y); 
    bool validThread = (m < M && n < N);

    float acc = 0.0;
    int sharedIdx = _m * 32 + _n;
    for (int k0 = 0; k0 < K; k0 += 32) 
    {
        int n_ = k0 + _n;
        int m_ = k0 + _m;
        As[sharedIdx] = (m < M && n_ < K) ? A[m * K + n_] : 0.0; // A[m, n_]
        Bs[sharedIdx] = (m_ < K && n < N) ? B[m_ * N + n] : 0.0; // B[m_, n]
        barrier();

        for (int k = 0; k < 32; ++k) 
            acc += As[_m * 32 + k] * Bs[k * 32 + _n];
        barrier();
    }

    if (validThread)
        C[m * N + n] = acc + b[n];
})";


// EfficientNet-specific shader declarations

static const char* src_depthwise_conv = R"(
#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) readonly buffer InBuffer { float in0[]; };
layout(set = 0, binding = 2) readonly buffer WeightBuffer { float weight[]; };
layout(set = 0, binding = 3) readonly buffer BiasBuffer { float bias[]; };

layout(push_constant) uniform PushConstants {
    int H;      // Input Height
    int W;      // Input Width
    int C;      // Channels
    int K;      // Kernel Size
    int Stride; // Stride
    int Padding; // Padding
    int H_out;  // Output Height
    int W_out;  // Output Width
};

void main() 
{
    int w_out = int(gl_GlobalInvocationID.x);
    int h_out = int(gl_GlobalInvocationID.y);
    int c = int(gl_GlobalInvocationID.z);

    if (w_out >= W_out || h_out >= H_out || c >= C)
        return;

    float sum = bias[c];
    int K_2 = K / 2;

    for (int kh = 0; kh < K; ++kh)
    {
        for (int kw = 0; kw < K; ++kw)
        {
            int h_in = h_out * Stride + kh - Padding;
            int w_in = w_out * Stride + kw - Padding;

            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W)
            {
                int input_idx = (h_in * W + w_in) * C + c;
                int weight_idx = c * K * K + kh * K + kw;
                sum += in0[input_idx] * weight[weight_idx];
            }
        }
    }

    int out_idx = (h_out * W_out + w_out) * C + c;
    out0[out_idx] = sum;
}
)";

static const char* src_add = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) readonly buffer InBuffer0 { float in0[]; };
layout(set = 0, binding = 2) readonly buffer InBuffer1 { float in1[]; };

layout(push_constant) uniform PushConstants {
    int N; // Total elements
};

void main() 
{
    int i = int(gl_GlobalInvocationID.x);
    if (i >= N) return;
    out0[i] = in0[i] + in1[i];
}
)";

static const char* src_global_avg_pool = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) readonly buffer InBuffer { float in0[]; };

layout(push_constant) uniform PushConstants {
    int H;
    int W;
    int C;
};

void main() 
{
    int c = int(gl_GlobalInvocationID.x);
    if (c >= C) return;

    float sum = 0.0;
    for (int h = 0; h < H; ++h)
    {
        for (int w = 0; w < W; ++w)
        {
            int idx = (h * W + w) * C + c;
            sum += in0[idx];
        }
    }

    out0[c] = sum / float(H * W);
}
)";

static const char* src_sigmoid = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) readonly buffer InBuffer { float in0[]; };

layout(push_constant) uniform PushConstants {
    int N;
};

void main() 
{
    int i = int(gl_GlobalInvocationID.x);
    if (i >= N) return;
    float x = in0[i];
    out0[i] = 1.0 / (1.0 + exp(-x));
}
)";

static const char* src_batchnorm = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) readonly buffer InBuffer { float in0[]; };
layout(set = 0, binding = 2) readonly buffer Mean { float mean[]; };
layout(set = 0, binding = 3) readonly buffer Var { float var[]; };
layout(set = 0, binding = 4) readonly buffer Gamma { float gamma[]; };
layout(set = 0, binding = 5) readonly buffer Beta { float beta[]; };

layout(push_constant) uniform PushConstants {
    int N; // Total spatial elements (H*W)
    int C; // Channels
    float Eps;
};

void main() 
{
    int i = int(gl_GlobalInvocationID.x);
    int total_elements = N * C;
    if (i >= total_elements) return;

    int c = i % C;

    float x = in0[i];
    float m = mean[c];
    float v = var[c];
    float g = gamma[c];
    float b = beta[c];

    out0[i] = (x - m) / sqrt(v + Eps) * g + b;
}
)";

static const char* src_depthwise_conv_bn_swish = R"(
#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) readonly buffer InBuffer { float in0[]; };
layout(set = 0, binding = 2) readonly buffer WeightBuffer { float weight[]; };
layout(set = 0, binding = 3) readonly buffer BiasBuffer { float bias[]; };
layout(set = 0, binding = 4) readonly buffer Mean { float mean[]; };
layout(set = 0, binding = 5) readonly buffer Var { float var[]; };
layout(set = 0, binding = 6) readonly buffer Gamma { float gamma[]; };
layout(set = 0, binding = 7) readonly buffer Beta { float beta[]; };

layout(push_constant) uniform PushConstants {
    int H; int W; int C; int K; int Stride; int Padding; int H_out; int W_out;
    float Eps;
};

void main() 
{
    int w_out = int(gl_GlobalInvocationID.x);
    int h_out = int(gl_GlobalInvocationID.y);
    int c = int(gl_GlobalInvocationID.z);

    if (w_out >= W_out || h_out >= H_out || c >= C)
        return;

    float sum = bias[c];
    
    for (int kh = 0; kh < K; ++kh)
    {
        for (int kw = 0; kw < K; ++kw)
        {
            int h_in = h_out * Stride + kh - Padding;
            int w_in = w_out * Stride + kw - Padding;

            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W)
            {
                int input_idx = (h_in * W + w_in) * C + c;
                int weight_idx = c * K * K + kh * K + kw;
                sum += in0[input_idx] * weight[weight_idx];
            }
        }
    }

    // Batch Norm
    float m = mean[c];
    float v = var[c];
    float g = gamma[c];
    float b = beta[c];
    float bn_res = (sum - m) / sqrt(v + Eps) * g + b;

    // Swish: x * sigmoid(x)
    float swish_res = bn_res * (1.0 / (1.0 + exp(-bn_res)));

    int out_idx = (h_out * W_out + w_out) * C + c;
    out0[out_idx] = swish_res;
}
)";

static const char* src_conv_bn_swish = R"(
#version 450
layout(local_size_x = 32, local_size_y = 32) in;
layout(set = 0, binding = 0) buffer OutBuffer { float C[]; };
layout(set = 0, binding = 1) buffer InBuffer { float A[]; };
layout(set = 0, binding = 2) buffer Weight { float B[]; };
layout(set = 0, binding = 3) buffer Bias { float b[]; };
layout(set = 0, binding = 4) readonly buffer Mean { float mean[]; };
layout(set = 0, binding = 5) readonly buffer Var { float var[]; };
layout(set = 0, binding = 6) readonly buffer Gamma { float gamma[]; };
layout(set = 0, binding = 7) readonly buffer Beta { float beta[]; };

layout(push_constant) uniform PushConstants {
    int M;  // # of batchs
    int K;  // # of inputs
    int N;  // # of outputs
    float Eps;
};

shared float As[32 * 32];
shared float Bs[32 * 32];

void main() 
{
    int n = int(gl_GlobalInvocationID.x); 
    int m = int(gl_GlobalInvocationID.y); 
    int _n = int(gl_LocalInvocationID.x); 
    int _m = int(gl_LocalInvocationID.y); 
    bool validThread = (m < M && n < N);

    float acc = 0.0;
    int sharedIdx = _m * 32 + _n;
    for (int k0 = 0; k0 < K; k0 += 32) 
    {
        int n_ = k0 + _n;
        int m_ = k0 + _m;
        As[sharedIdx] = (m < M && n_ < K) ? A[m * K + n_] : 0.0;
        Bs[sharedIdx] = (m_ < K && n < N) ? B[m_ * N + n] : 0.0;
        barrier();

        for (int k = 0; k < 32; ++k) 
            acc += As[_m * 32 + k] * Bs[k * 32 + _n];
        barrier();
    }

    if (validThread) {
        float conv_res = acc + b[n];
        
        // Batch Norm (n is the channel index for output)
        float m_val = mean[n];
        float v_val = var[n];
        float g_val = gamma[n];
        float b_val = beta[n];
        float bn_res = (conv_res - m_val) / sqrt(v_val + Eps) * g_val + b_val;

        // Swish
        float swish_res = bn_res * (1.0 / (1.0 + exp(-bn_res)));

        C[m * N + n] = swish_res;
    }
})";

// Helper shaders for SE block internals
static const char* src_fc_swish = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in0[]; };
layout(set = 0, binding = 2) buffer Weight { float weight[]; };
layout(set = 0, binding = 3) buffer Bias { float bias[]; };

layout(push_constant) uniform PushConstants {
    int I; // Input size
    int O; // Output size
};

void main() 
{
    int o = int(gl_GlobalInvocationID.x);
    if (o >= O) return;

    float sum = bias[o];
    for (int i = 0; i < I; ++i)
        sum += in0[i] * weight[i * O + o]; // Assuming weight is [I][O] flattened

    out0[o] = sum * (1.0 / (1.0 + exp(-sum)));
}
)";

static const char* src_fc_sigmoid = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in0[]; };
layout(set = 0, binding = 2) buffer Weight { float weight[]; };
layout(set = 0, binding = 3) buffer Bias { float bias[]; };

layout(push_constant) uniform PushConstants {
    int I; // Input size
    int O; // Output size
};

void main() 
{
    int o = int(gl_GlobalInvocationID.x);
    if (o >= O) return;

    float sum = bias[o];
    for (int i = 0; i < I; ++i)
        sum += in0[i] * weight[i * O + o];

    out0[o] = 1.0 / (1.0 + exp(-sum));
}
)";

static const char* src_se_scale = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in0[]; }; // HxWxC
layout(set = 0, binding = 2) buffer ScaleBuffer { float scale[]; }; // 1x1xC

layout(push_constant) uniform PushConstants {
    int N; // H*W
    int C; 
};

void main() 
{
    int i = int(gl_GlobalInvocationID.x);
    if (i >= N * C) return;

    int c = i % C;
    out0[i] = in0[i] * scale[c];
}
)";


Device netGlobalDevice = VulkanApp::get().device();

static DescriptorPool gDestSetPool = netGlobalDevice.createDescriptorPool({
    .maxTypes = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER <= 10000}, 
    .maxSets = 5000
});


static ComputePipeline requestPipeline(const char* src)
{
    static std::unordered_map<const char*, ComputePipeline> pipelineCache;

    auto [it, inserted] = pipelineCache.try_emplace(src);
    if (inserted)
        it->second = netGlobalDevice.createComputePipeline({src});
    return it->second;
}

static std::map<const char*, uint32_t> gGemmTileSize =
{
    {src_gemm_naive, 32},
    {src_gemm_shared, 32},
    {src_conv_bn_swish, 32} // Reuse tile size for fused kernel
};

template<typename... Dims>
static void ensureTensorFilled(Tensor& tensor, float value, Dims... dims)
{
    if (tensor.numElements())
        return;
    tensor = Tensor(static_cast<uint32_t>(dims)...);
    std::vector<float> data(tensor.numElements(), value);
    tensor.set(std::move(data));
}

void loadShaders()
{
    requestPipeline(src_relu);
    requestPipeline(src_setZero);
    requestPipeline(src_maxpool);
    requestPipeline(src_im2col);
    requestPipeline(src_gemm_naive);
    requestPipeline(src_gemm_kSplit);
    requestPipeline(src_gemm_shared);
    
    requestPipeline(src_depthwise_conv);
    requestPipeline(src_add);
    requestPipeline(src_global_avg_pool);
    requestPipeline(src_sigmoid);
    requestPipeline(src_batchnorm);
    requestPipeline(src_depthwise_conv_bn_swish);
    requestPipeline(src_conv_bn_swish);
    
    requestPipeline(src_fc_swish);
    requestPipeline(src_fc_sigmoid);
    requestPipeline(src_se_scale);
}


/////////////////////////////////////////////////////////////////////////////////////////
// Existing Node Implementations (from reference)
/////////////////////////////////////////////////////////////////////////////////////////

ConvolutionNode::ConvolutionNode(uint32_t inChannels, uint32_t outChannels, uint32_t kernelWidth)
:  C(inChannels), F(outChannels), K(kernelWidth)
{
    _ASSERT(K % 2 == 1);
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
    addSlot("im2colOut", NodeSlot::internal);
    addSlot("weight", NodeSlot::input);
    addSlot("bias", NodeSlot::input);

    im2col = requestPipeline(src_im2col);
    im2colDescSet = im2col.descSetLayout(0).newDescSet(gDestSetPool);
    
    const char* gemmSrc = src_gemm_shared;
    gemm = requestPipeline(gemmSrc);
    gemmTileSize = gGemmTileSize.at(gemmSrc);
    gemmDescSet = gemm.descSetLayout(0).newDescSet(gDestSetPool);
}

void ConvolutionNode::prepare()
{
    _ASSERT((*this)["in0"].isShapeOf(-1, -1, C));
    ensureTensorFilled((*this)["weight"], 0.0f, C * K * K, F);
    ensureTensorFilled((*this)["bias"], 0.0f, F);
    if (!(*this)["weight"].isShapeOf(C*K*K, F)) {
        const auto& s = (*this)["weight"].shape();
        std::cout << "ConvolutionNode::prepare: weight shape mismatch! Expected " << C*K*K << "x" << F << ", got ";
        for (auto d : s) std::cout << d << " ";
        std::cout << std::endl;
    }
    _ASSERT((*this)["weight"].isShapeOf(C*K*K, F));
    _ASSERT((*this)["bias"].isShapeOf(F));

    const auto& inShape = (*this)["in0"].shape();
    (*this)["im2colOut"] = Tensor(inShape[0], inShape[1], C*K*K);
    (*this)["out0"] = Tensor(inShape[0], inShape[1], F);
}

void ConvolutionNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    uint32_t H = inShape[0], W = inShape[1];

    im2colDescSet.write({
        (*this)["im2colOut"].buffer(),
        (*this)["in0"].buffer(),
    });

    gemmDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["im2colOut"].buffer(),
        (*this)["weight"].buffer(),
        (*this)["bias"].buffer(),
    });

    uint32_t Stride = 1;
    uint32_t Padding = 0; // ConvolutionNode assumes valid padding handled externally or same padding? 
    // Wait, ConvolutionNode doesn't support padding in constructor?
    // It assumes "Same" padding?
    // Reference implementation didn't have padding logic in im2col?
    // Let's assume Stride=1, Padding=K/2 (Same)
    Padding = K / 2;
    uint32_t H_out = H;
    uint32_t W_out = W;
    uint32_t im2colConstants[] = {H, W, C, K, Stride, Padding, H_out, W_out};
    uint32_t M = H * W;
    uint32_t K_ = C * K * K;
    uint32_t N = F;
    uint32_t gemmConstants[] = {M, K_, N};

    cmdBuff
        .bindPipeline(im2col)
        .bindDescSets({im2colDescSet})
        .setPushConstants(0, sizeof(im2colConstants), im2colConstants)
        .dispatch(H * W, C * K * K)
        .barrier( 
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["im2colOut"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        )
        .bindPipeline(gemm)
        .bindDescSets({gemmDescSet})
        .setPushConstants(0, sizeof(gemmConstants), gemmConstants)
        .dispatch0(CEIL_DIV(N, gemmTileSize), CEIL_DIV(M, gemmTileSize))
        .barrier( 
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}  


ReluNode::ReluNode()
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);

    relu = requestPipeline(src_relu);
    reluDescSet = relu.descSetLayout(0).newDescSet(gDestSetPool);
}

void ReluNode::prepare()
{
    _ASSERT((*this)["in0"].validShape());
    (*this)["out0"] = Tensor((*this)["in0"].shape());
}

void ReluNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    int I = 1;
    for (int dim : inShape) I *= dim;
    
    reluDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
    });

    int reluConstants[] = {I};

    cmdBuff
        .bindPipeline(relu)
        .setPushConstants(0, sizeof(reluConstants), reluConstants)
        .bindDescSets({reluDescSet})
        .dispatch(I)
        .barrier( 
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}  


MaxPoolingNode::MaxPoolingNode(uint32_t poolSize)
: P(poolSize)
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);

    maxpool = requestPipeline(src_maxpool);
    maxpoolDescSet = maxpool.descSetLayout(0).newDescSet(gDestSetPool);
}

void MaxPoolingNode::prepare()
{
    const auto& inShape = (*this)["in0"].shape();
    _ASSERT(inShape.size() == 3);
    uint32_t H = inShape[0], W = inShape[1], C = inShape[2];

    if (discardTail)
        (*this)["out0"] = Tensor(H / P, W / P, C);
    else    
        (*this)["out0"] = Tensor((H + P - 1) / P, (W + P - 1) / P, C);
}

void MaxPoolingNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    uint32_t H = inShape[0], W = inShape[1], C = inShape[2];
    uint32_t H_ = discardTail ? H / P : (H + P - 1) / P;
    uint32_t W_ = discardTail ? W / P : (W + P - 1) / P;

    maxpoolDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
    });

    uint32_t maxpoolConstants[] = {H, W, C, P};

    cmdBuff
        .bindPipeline(maxpool)
        .bindDescSets({maxpoolDescSet})
        .setPushConstants(0, sizeof(maxpoolConstants), maxpoolConstants)
        .dispatch(H_, W_, C)
        .barrier( 
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}  


FlattenNode::FlattenNode()
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
}

void FlattenNode::prepare()
{
    _ASSERT((*this)["in0"].validShape());
    Tensor& outTensor = (*this)["out0"] = (*this)["in0"];
    outTensor.reshape(outTensor.numElements());  
}

void FlattenNode::run(CommandBuffer cmdBuff) 
{
}  


FullyConnectedNode::FullyConnectedNode(uint32_t inDim, uint32_t outDim)
: I(inDim), O(outDim) 
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
    addSlot("weight", NodeSlot::input);
    addSlot("bias", NodeSlot::input);

    const char* gemmSrc = src_gemm_naive;
    gemm = requestPipeline(gemmSrc);
    gemmDescSet = gemm.descSetLayout(0).newDescSet(gDestSetPool);

    setZero = requestPipeline(src_setZero);
    setZeroDescSet = setZero.descSetLayout(0).newDescSet(gDestSetPool);
}

void FullyConnectedNode::prepare() 
{
    _ASSERT((*this)["in0"].isShapeOf(I));
    ensureTensorFilled((*this)["weight"], 0.0f, I, O);
    ensureTensorFilled((*this)["bias"], 0.0f, O);
    _ASSERT((*this)["weight"].isShapeOf(I, O));
    _ASSERT((*this)["bias"].isShapeOf(O));
    (*this)["out0"] = Tensor(O); 
}

void FullyConnectedNode::run(CommandBuffer cmdBuff) 
{
    uint32_t M = 1;
    uint32_t K = (*this)["in0"].shape()[0];
    uint32_t N = (*this)["out0"].shape()[0];

    gemmDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
        (*this)["weight"].buffer(),
        (*this)["bias"].buffer(),
    });

    uint32_t gemmConstants[] = {M, K, N};

    cmdBuff
        .bindPipeline(gemm)
        .bindDescSets({gemmDescSet})
        .setPushConstants(0, sizeof(gemmConstants), gemmConstants)
        .dispatch(N, M)
        .barrier( 
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}  


/////////////////////////////////////////////////////////////////////////////////////////
// New EfficientNet Node Implementations
/////////////////////////////////////////////////////////////////////////////////////////

DepthwiseConvNode::DepthwiseConvNode(uint32_t channels, uint32_t kernelWidth)
: C(channels), K(kernelWidth)
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
    addSlot("weight", NodeSlot::input);
    addSlot("bias", NodeSlot::input);
    
    depthwiseConv = requestPipeline(src_depthwise_conv);
    descSet = depthwiseConv.descSetLayout(0).newDescSet(gDestSetPool);
}

void DepthwiseConvNode::prepare()
{
    const auto& inShape = (*this)["in0"].shape();
    _ASSERT(inShape.size() == 3 && inShape[2] == C);
    
    _ASSERT((*this)["weight"].isShapeOf(C, K, K));
    _ASSERT((*this)["bias"].isShapeOf(C));

    (*this)["out0"] = Tensor(inShape[0], inShape[1], C);
}

void DepthwiseConvNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    uint32_t H = inShape[0];
    uint32_t W = inShape[1];
    
    uint32_t Stride = 1; 
    uint32_t Padding = K / 2; 
    uint32_t H_out = H; 
    uint32_t W_out = W;

    descSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
        (*this)["weight"].buffer(),
        (*this)["bias"].buffer(),
    });

    uint32_t constants[] = {H, W, C, K, Stride, Padding, H_out, W_out};

    cmdBuff
        .bindPipeline(depthwiseConv)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), constants)
        .dispatch(CEIL_DIV(W_out, 16), CEIL_DIV(H_out, 16), C)
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}


AddNode::AddNode()
{
    addSlot("in0", NodeSlot::input);
    addSlot("in1", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
    
    add = requestPipeline(src_add);
    descSet = add.descSetLayout(0).newDescSet(gDestSetPool);
}

void AddNode::prepare()
{
    _ASSERT((*this)["in0"].validShape());
    const auto& s0 = (*this)["in0"].shape();
    const auto& s1 = (*this)["in1"].shape();
    _ASSERT(s0 == s1);

    (*this)["out0"] = Tensor(s0);
}

void AddNode::run(CommandBuffer cmdBuff)
{
    uint32_t N = (*this)["out0"].numElements();

    descSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
        (*this)["in1"].buffer(),
    });

    cmdBuff
        .bindPipeline(add)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(N), &N)
        .dispatch(N)
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}


GlobalAvgPoolNode::GlobalAvgPoolNode()
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
    
    globalAvgPool = requestPipeline(src_global_avg_pool);
    descSet = globalAvgPool.descSetLayout(0).newDescSet(gDestSetPool);
}

void GlobalAvgPoolNode::prepare()
{
    const auto& inShape = (*this)["in0"].shape();
    _ASSERT(inShape.size() == 3);
    uint32_t C = inShape[2];
    (*this)["out0"] = Tensor(1, 1, C);
}

void GlobalAvgPoolNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    uint32_t H = inShape[0];
    uint32_t W = inShape[1];
    uint32_t C = inShape[2];

    descSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
    });

    uint32_t constants[] = {H, W, C};

    cmdBuff
        .bindPipeline(globalAvgPool)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), constants)
        .dispatch(C)
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}


SigmoidNode::SigmoidNode()
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
    
    sigmoid = requestPipeline(src_sigmoid);
    descSet = sigmoid.descSetLayout(0).newDescSet(gDestSetPool);
}

void SigmoidNode::prepare()
{
    _ASSERT((*this)["in0"].validShape());
    (*this)["out0"] = Tensor((*this)["in0"].shape());
}

void SigmoidNode::run(CommandBuffer cmdBuff)
{
    uint32_t N = (*this)["out0"].numElements();

    descSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
    });

    cmdBuff
        .bindPipeline(sigmoid)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(N), &N)
        .dispatch(N)
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}


BatchNormNode::BatchNormNode(float epsilon)
: eps(epsilon)
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
    addSlot("mean", NodeSlot::input);
    addSlot("variance", NodeSlot::input);
    addSlot("gamma", NodeSlot::input);
    addSlot("beta", NodeSlot::input);
    
    batchnorm = requestPipeline(src_batchnorm);
    descSet = batchnorm.descSetLayout(0).newDescSet(gDestSetPool);
}

void BatchNormNode::prepare()
{
    const auto& inShape = (*this)["in0"].shape();
    _ASSERT(inShape.size() == 3);
    uint32_t C = inShape[2];
    ensureTensorFilled((*this)["mean"], 0.0f, C);
    ensureTensorFilled((*this)["variance"], 1.0f, C);
    ensureTensorFilled((*this)["gamma"], 1.0f, C);
    ensureTensorFilled((*this)["beta"], 0.0f, C);

    _ASSERT((*this)["mean"].isShapeOf(C));
    _ASSERT((*this)["variance"].isShapeOf(C));
    _ASSERT((*this)["gamma"].isShapeOf(C));
    _ASSERT((*this)["beta"].isShapeOf(C));

    (*this)["out0"] = Tensor(inShape);
}

void BatchNormNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    uint32_t H = inShape[0];
    uint32_t W = inShape[1];
    uint32_t C = inShape[2];
    uint32_t N = H * W;

    descSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
        (*this)["mean"].buffer(),
        (*this)["variance"].buffer(),
        (*this)["gamma"].buffer(),
        (*this)["beta"].buffer(),
    });

    struct PC { int N; int C; float Eps; };
    PC constants = { (int)N, (int)C, eps };

    cmdBuff
        .bindPipeline(batchnorm)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch(N * C)
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}


DepthwiseConvBNSwishNode::DepthwiseConvBNSwishNode(uint32_t channels, uint32_t kernelWidth, uint32_t stride)
: C(channels), K(kernelWidth), Stride(stride)
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
    addSlot("weight", NodeSlot::input);
    addSlot("bias", NodeSlot::input);
    addSlot("mean", NodeSlot::input);
    addSlot("variance", NodeSlot::input);
    addSlot("gamma", NodeSlot::input);
    addSlot("beta", NodeSlot::input);
    
    fused = requestPipeline(src_depthwise_conv_bn_swish);
    descSet = fused.descSetLayout(0).newDescSet(gDestSetPool);
}

void DepthwiseConvBNSwishNode::prepare()
{
    const auto& s = (*this)["weight"].shape();
    printf("DepthwiseConvBNSwishNode::prepare: C=%d K=%d Weight=", C, K);
    for(auto d : s) printf("%dx", d);
    printf("\n");
    fflush(stdout);

    const auto& inShape = (*this)["in0"].shape();
    _ASSERT(inShape.size() == 3 && inShape[2] == C);
    ensureTensorFilled((*this)["weight"], 0.0f, C, K, K);
    ensureTensorFilled((*this)["bias"], 0.0f, C);
    ensureTensorFilled((*this)["mean"], 0.0f, C);
    ensureTensorFilled((*this)["variance"], 1.0f, C);
    ensureTensorFilled((*this)["gamma"], 1.0f, C);
    ensureTensorFilled((*this)["beta"], 0.0f, C);
    
    // const auto& s = (*this)["weight"].shape();
    // std::cout << "DepthwiseConvBNSwishNode::prepare: C=" << C << " K=" << K << " Weight=";
    // for(auto d : s) std::cout << d << "x";
    // std::cout << std::endl;
    
    if (!(*this)["weight"].isShapeOf(C, K, K)) {
        std::cout << "DepthwiseConvBNSwishNode::prepare: weight shape mismatch! Expected " << C << "x" << K << "x" << K << ", got ";
        for (auto d : s) std::cout << d << " ";
        std::cout << std::endl;
    }
    _ASSERT((*this)["weight"].isShapeOf(C, K, K));
    
    // Stride에 따른 출력 크기 계산 (same padding 가정)
    uint32_t H = inShape[0];
    uint32_t W = inShape[1];
    uint32_t H_out = (H + Stride - 1) / Stride;
    uint32_t W_out = (W + Stride - 1) / Stride;
    (*this)["out0"] = Tensor(H_out, W_out, C);
}

void DepthwiseConvBNSwishNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    uint32_t H = inShape[0];
    uint32_t W = inShape[1];
    
    uint32_t Padding = K / 2; 
    uint32_t H_out = (H + Stride - 1) / Stride;
    uint32_t W_out = (W + Stride - 1) / Stride;
    float Eps = 1e-5f;

    descSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
        (*this)["weight"].buffer(),
        (*this)["bias"].buffer(),
        (*this)["mean"].buffer(),
        (*this)["variance"].buffer(),
        (*this)["gamma"].buffer(),
        (*this)["beta"].buffer(),
    });

    struct PC { 
        int H; int W; int C; int K; int Stride; int Padding; int H_out; int W_out;
        float Eps;
    };
    PC constants = { (int)H, (int)W, (int)C, (int)K, (int)Stride, (int)Padding, (int)H_out, (int)W_out, Eps };

    cmdBuff
        .bindPipeline(fused)
        .bindDescSets({descSet})
        .setPushConstants(0, sizeof(constants), &constants)
        .dispatch(W_out, H_out, C)
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}


ConvBNSwishNode::ConvBNSwishNode(uint32_t inChannels, uint32_t outChannels, uint32_t kernelWidth, uint32_t stride)
: C(inChannels), F(outChannels), K(kernelWidth), Stride(stride)
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
    addSlot("im2colOut", NodeSlot::internal);
    addSlot("weight", NodeSlot::input);
    addSlot("bias", NodeSlot::input);
    addSlot("mean", NodeSlot::input);
    addSlot("variance", NodeSlot::input);
    addSlot("gamma", NodeSlot::input);
    addSlot("beta", NodeSlot::input);
    
    im2col = requestPipeline(src_im2col);
    im2colDescSet = im2col.descSetLayout(0).newDescSet(gDestSetPool);
    
    gemm_bn_swish = requestPipeline(src_conv_bn_swish);
    gemmDescSet = gemm_bn_swish.descSetLayout(0).newDescSet(gDestSetPool);
    gemmTileSize = gGemmTileSize.at(src_conv_bn_swish);
}

void ConvBNSwishNode::prepare()
{
    _ASSERT((*this)["in0"].isShapeOf(-1, -1, C));
    ensureTensorFilled((*this)["weight"], 0.0f, C * K * K, F);
    ensureTensorFilled((*this)["bias"], 0.0f, F);
    ensureTensorFilled((*this)["mean"], 0.0f, F);
    ensureTensorFilled((*this)["variance"], 1.0f, F);
    ensureTensorFilled((*this)["gamma"], 1.0f, F);
    ensureTensorFilled((*this)["beta"], 0.0f, F);
    const auto& inShape = (*this)["in0"].shape();
    
    uint32_t H_out = (inShape[0] + Stride - 1) / Stride;
    uint32_t W_out = (inShape[1] + Stride - 1) / Stride;
    
    (*this)["im2colOut"] = Tensor(H_out, W_out, C*K*K);
    (*this)["out0"] = Tensor(H_out, W_out, F);
}

void ConvBNSwishNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    uint32_t H = inShape[0], W = inShape[1];
    
    uint32_t Padding = K / 2;
    uint32_t H_out = (H + Stride - 1) / Stride;
    uint32_t W_out = (W + Stride - 1) / Stride;

    im2colDescSet.write({
        (*this)["im2colOut"].buffer(),
        (*this)["in0"].buffer(),
    });

    gemmDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["im2colOut"].buffer(),
        (*this)["weight"].buffer(),
        (*this)["bias"].buffer(),
        (*this)["mean"].buffer(),
        (*this)["variance"].buffer(),
        (*this)["gamma"].buffer(),
        (*this)["beta"].buffer(),
    });

    uint32_t im2colConstants[] = {H, W, C, K, Stride, Padding, H_out, W_out};
    uint32_t M = H_out * W_out;
    uint32_t K_ = C * K * K;
    uint32_t N = F;
    float Eps = 1e-5f;

    struct PC { int M; int K; int N; float Eps; };
    PC gemmConstants = { (int)M, (int)K_, (int)N, Eps };

    cmdBuff
        .bindPipeline(im2col)
        .bindDescSets({im2colDescSet})
        .setPushConstants(0, sizeof(im2colConstants), im2colConstants)
        .dispatch(H_out * W_out, C * K * K)
        .barrier( 
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["im2colOut"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        )
        .bindPipeline(gemm_bn_swish)
        .bindDescSets({gemmDescSet})
        .setPushConstants(0, sizeof(gemmConstants), &gemmConstants)
        .dispatch0(CEIL_DIV(N, gemmTileSize), CEIL_DIV(M, gemmTileSize))
        .barrier( 
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}


SEBlockNode::SEBlockNode(uint32_t channels, uint32_t seReduce)
: C(channels), se_reduce(seReduce)
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
    
    addSlot("gap_out", NodeSlot::internal);
    addSlot("reduce_out", NodeSlot::internal);
    addSlot("expand_out", NodeSlot::internal);

    addSlot("weight_reduce", NodeSlot::input);
    addSlot("bias_reduce", NodeSlot::input);
    addSlot("weight_expand", NodeSlot::input);
    addSlot("bias_expand", NodeSlot::input);
    
    gapPipeline = requestPipeline(src_global_avg_pool);
    gapDescSet = gapPipeline.descSetLayout(0).newDescSet(gDestSetPool);

    reducePipeline = requestPipeline(src_fc_swish);
    reduceDescSet = reducePipeline.descSetLayout(0).newDescSet(gDestSetPool);

    expandPipeline = requestPipeline(src_fc_sigmoid);
    expandDescSet = expandPipeline.descSetLayout(0).newDescSet(gDestSetPool);

    scalePipeline = requestPipeline(src_se_scale);
    scaleDescSet = scalePipeline.descSetLayout(0).newDescSet(gDestSetPool);
}

void SEBlockNode::prepare()
{
    const auto& inShape = (*this)["in0"].shape();
    _ASSERT(inShape.size() == 3 && inShape[2] == C);
    ensureTensorFilled((*this)["weight_reduce"], 0.0f, C, se_reduce);
    ensureTensorFilled((*this)["bias_reduce"], 0.0f, se_reduce);
    ensureTensorFilled((*this)["weight_expand"], 0.0f, se_reduce, C);
    ensureTensorFilled((*this)["bias_expand"], 0.0f, C);
    
    (*this)["gap_out"] = Tensor(1, 1, C);
    (*this)["reduce_out"] = Tensor(1, 1, se_reduce);
    (*this)["expand_out"] = Tensor(1, 1, C);
    (*this)["out0"] = Tensor(inShape);
}

void SEBlockNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    uint32_t H = inShape[0], W = inShape[1];

    // 1. GAP: [H, W, C] -> [1, 1, C]
    {
        gapDescSet.write({
            (*this)["gap_out"].buffer(),
            (*this)["in0"].buffer(),
        });
        uint32_t consts[] = {H, W, C};
        cmdBuff
            .bindPipeline(gapPipeline)
            .bindDescSets({gapDescSet})
            .setPushConstants(0, sizeof(consts), consts)
            .dispatch(C)
            .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / (*this)["gap_out"].buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ));
    }

    // 2. FC Reduce + Swish: [1, 1, C] -> [1, 1, se_reduce]
    {
        reduceDescSet.write({
            (*this)["reduce_out"].buffer(),
            (*this)["gap_out"].buffer(),
            (*this)["weight_reduce"].buffer(),
            (*this)["bias_reduce"].buffer(),
        });
        uint32_t consts[] = {C, se_reduce};
        cmdBuff
            .bindPipeline(reducePipeline)
            .bindDescSets({reduceDescSet})
            .setPushConstants(0, sizeof(consts), consts)
            .dispatch(se_reduce)
            .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / (*this)["reduce_out"].buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ));
    }

    // 3. FC Expand + Sigmoid: [1, 1, se_reduce] -> [1, 1, C]
    {
        expandDescSet.write({
            (*this)["expand_out"].buffer(),
            (*this)["reduce_out"].buffer(),
            (*this)["weight_expand"].buffer(),
            (*this)["bias_expand"].buffer(),
        });
        uint32_t consts[] = {se_reduce, C};
        cmdBuff
            .bindPipeline(expandPipeline)
            .bindDescSets({expandDescSet})
            .setPushConstants(0, sizeof(consts), consts)
            .dispatch(C)
            .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / (*this)["expand_out"].buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ));
    }

    // 4. Scale: [H, W, C] * [1, 1, C] (broadcast) -> Output
    {
        scaleDescSet.write({
            (*this)["out0"].buffer(),
            (*this)["in0"].buffer(),
            (*this)["expand_out"].buffer(),
        });
        uint32_t consts[] = {H * W, C};
        cmdBuff
            .bindPipeline(scalePipeline)
            .bindDescSets({scaleDescSet})
            .setPushConstants(0, sizeof(consts), consts)
            .dispatch(H * W * C)
            .barrier((PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE) / (*this)["out0"].buffer() / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ));
    }
}


MBConvBlockNode::MBConvBlockNode(const MBConvConfig& cfg)
: config(cfg)
{
    NodeSlot* lastOutputSlot = nullptr;
    NodeSlot* skipInputSlot = nullptr; // Skip connection을 위한 입력 슬롯 저장

    // Skip connection이 필요한지 확인
    bool needsSkipConnection = (config.stride == 1 && config.in_channels == config.out_channels);

    // Skip connection이 필요한 경우, InputNode를 사용하여 입력을 분기
    if (needsSkipConnection)
    {
        inputSplitNode = std::make_unique<InputNode>();
        defineSlot("in0", inputSplitNode->slot("in0"));
        skipInputSlot = &inputSplitNode->slot("out0");
    }

    // 1. Expand Phase (1x1 Conv + BN + Swish)
    if (config.expand_ratio != 1)
    {
        uint32_t expanded_channels = config.in_channels * config.expand_ratio;
        expandConv = std::make_unique<ConvBNSwishNode>(config.in_channels, expanded_channels, 1);
        
        if (skipInputSlot)
        {
            // Skip connection이 있는 경우, InputNode 출력을 expandConv 입력에 연결
            *skipInputSlot - expandConv->slot("in0");
        }
        else
        {
            // Skip connection이 없는 경우, 그룹 입력을 직접 연결
            defineSlot("in0", expandConv->slot("in0"));
        }
        lastOutputSlot = &expandConv->slot("out0");
    }

    // 2. Depthwise Phase (KxK DW Conv + BN + Swish)
    uint32_t dw_channels = config.in_channels * config.expand_ratio;
    depthwiseConv = std::make_unique<DepthwiseConvBNSwishNode>(dw_channels, config.kernel_size, config.stride);
    
    if (lastOutputSlot)
    {
        *lastOutputSlot - depthwiseConv->slot("in0");
    }
    else
    {
        if (skipInputSlot)
        {
            *skipInputSlot - depthwiseConv->slot("in0");
        }
        else
        {
            defineSlot("in0", depthwiseConv->slot("in0"));
        }
    }
    
    lastOutputSlot = &depthwiseConv->slot("out0");

    // 3. Squeeze-Excite Phase
    if (config.se_ratio > 0)
    {
        uint32_t se_reduce = static_cast<uint32_t>(config.in_channels * config.se_ratio); // Usually based on input channels
        seBlock = std::make_unique<SEBlockNode>(dw_channels, se_reduce);
        
        *lastOutputSlot - seBlock->slot("in0");
        lastOutputSlot = &seBlock->slot("out0");
    }

    // 4. Output Project Phase (1x1 Conv + BN) - Linear activation
    projectConvNode = std::make_unique<ConvolutionNode>(dw_channels, config.out_channels, 1);
    projectBNNode = std::make_unique<BatchNormNode>();

    *lastOutputSlot - projectConvNode->slot("in0");
    projectConvNode->slot("out0") - projectBNNode->slot("in0");
    
    lastOutputSlot = &projectBNNode->slot("out0");

    // 5. Skip Connection (Add)
    if (needsSkipConnection && skipInputSlot)
    {
        addNode = std::make_unique<AddNode>();
        
        // One input from projection
        *lastOutputSlot - addNode->slot("in0");
        
        // Other input from original group input (via InputNode)
        *skipInputSlot - addNode->slot("in1");
        
        lastOutputSlot = &addNode->slot("out0");
    }

    defineSlot("out0", *lastOutputSlot);
}

Tensor& MBConvBlockNode::operator[](const std::string& name)
{
    if (expandConv && name.starts_with("expand."))
        return (*expandConv)[name.substr(7)];
    if (depthwiseConv && name.starts_with("depthwise."))
        return (*depthwiseConv)[name.substr(10)];
    if (seBlock && name.starts_with("se."))
        return (*seBlock)[name.substr(3)];
    if (projectConvNode && name.starts_with("project."))
        return (*projectConvNode)[name.substr(8)];
    if (projectBNNode && name.starts_with("projectBN."))
        return (*projectBNNode)[name.substr(10)];
        
    throw std::runtime_error("MBConvBlockNode: Weight not found " + name);
}

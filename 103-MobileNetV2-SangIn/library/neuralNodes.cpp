#include "neuralNodes.h"
#include <unordered_map>
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))


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

static const char* src_copy = R"(
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
    out0[o] = in0[o];
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
layout(local_size_x = 64, local_size_y = 16) in;

layout(set = 0, binding = 0) writeonly buffer OutBuffer { float im2colOut[]; };
layout(set = 0, binding = 1) readonly buffer InBuffer  { float in0[]; };

layout(push_constant) uniform PushConstants {
    int H;      // input height
    int W;      // input width
    int C;      // input channels
    int K;      // kernel size
    int S;      // stride
    int P;      // padding
    int H_out;  // output height
    int W_out;  // output width
};

void main() 
{
    int i = int(gl_GlobalInvocationID.x);  // 0 .. H_out*W_out - 1
    int j = int(gl_GlobalInvocationID.y);  // 0 .. C*K*K      - 1

    int KK  = K * K;
    int CKK = C * KK;

    if (i >= H_out * W_out || j >= CKK)
        return;

    // output 좌표
    int h_out = i / W_out;
    int w_out = i % W_out;

    // 채널 / kernel index
    int c  = j / KK;
    int k  = j % KK;
    int kh = k / K;
    int kw = k % K;

    // input 좌표 (stride / padding 적용)
    int h_in = h_out * S + kh - P;
    int w_in = w_out * S + kw - P;

    float value = 0.0;
    if (0 <= h_in && h_in < H && 0 <= w_in && w_in < W)
        value = in0[( (h_in * W) + w_in ) * C + c];

    int out_idx = i * CKK + j;
    im2colOut[out_idx] = value;
}
)";

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

static const char* src_gemm_kSplit2 = R"(
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

shared float sA;

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

        if (gl_LocalInvocationIndex.x == 0) 
            sA = A[m * K + k];
        barrier();

        sum += sA * B[k * N + n];
        barrier();
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

static const char* src_gemm_multiOut1d = R"(
#version 450
#define BN 64
#define BM 64
#define BK 16
#define TM 4
#define TRANSPOSE_A 0
// assert BN == BM == TM*BK
// Each workgroup computes a BMxBN results while it consists of (BM/TM)xBN threads.

layout(local_size_x = BN*BM/TM) in;     // (BM/TM) x BN
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

shared float As[BM * BK]; // BMxBK
shared float Bs[BK * BN]; // BKxBN

void main() 
{
    int tileCol = int(gl_WorkGroupID.x); 
    int tileRow = int(gl_WorkGroupID.y); 
    int t = int(gl_LocalInvocationID.x);

    int bk = t % BK, d_bk = t / BK;
    int bn = t % BN, d_bn = t / BN; 
    /*
    * The matrix row index m is computed in two different contexts:
    *   m = tileRow * BM + d_bk;            // Used when loading A to ensure coalesced global memory access.
    *   m = tileRow * BM + d_bn * TM + tm;  // Used when accessing C multiple times within a single thread.
    */
    int m = tileRow * BM + d_bk;            // d_bk : 0 ~ (BM*BN)/(TM*BK)-1 = BN-1 => assert BN == BM (Since 0 <= m - tileRow * BM < BM)
    int n = tileCol * BN + bn;

    float result[TM];   for (int tm = 0; tm < TM; ++tm) result[tm] = 0.0;
#if TRANSPOSE_A
    float regM[TM];
    int t_t = bk * BM + d_bk; // transposed index of t
#endif

    for (int k_ = 0; k_ < K; k_ += BK)
    {
        int k = k_ + bk;
    #if TRANSPOSE_A
        As[t_t] = (m < M) && (k < K) ? A[m * K + k] : 0.0;
    #else
        As[t] = (m < M) && (k < K) ? A[m * K + k] : 0.0;   
    #endif
        
        k = k_ + d_bn;                      // d_bn : 0 ~ (BM*BN)/(TM*BN)-1 = BM/TM-1 => assert BM/TM == BK (Since 0 <= m_ - k_ < BK)
        Bs[t] = (k < K) && (n < N) ? B[k * N + n] : 0.0;  
        barrier();

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
            float regN = Bs[dotIdx * BN + bn];

        #if TRANSPOSE_A
            for (int tm = 0; tm < TM; ++tm)
                regM[tm] = As[dotIdx * BM + (d_bn * TM + tm)];
            for (int tm = 0; tm < TM; ++tm)
                result[tm] += regM[tm] * regN;
        #else
            for (int tm = 0; tm < TM; ++tm)
                result[tm] += As[(d_bn * TM + tm) * BK + dotIdx] * regN;
        #endif
        }
        barrier();
    }

    if (n >= N)     // Return only after barrier(); skipping it may cause deadlock
        return;

    float bias = b[n];
    int m0 = tileRow * BM + d_bn * TM;

    for (int tm = 0; tm < TM; ++tm)
    {
        m = m0 + tm;
        if (m >= M)
            break;
        C[m * N + n] = result[tm] + bias;
    }
})";

static const char* src_gemm_multiOut2d = R"(
#version 450
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
/*
* assert: T/BK, T/BN is natural number
* assert: BM/(T/BK) = (BK*TM*TN)/BN is natural number
* assert: BK/(T/BN) = (BK*TM*TN)/BM is natural number
*/

layout(local_size_x = (BM*BN)/(TM*TN)) in; 
layout(set = 0, binding = 0) buffer OutBuffer { float C[]; };
layout(set = 0, binding = 1) buffer InBuffer { float A[]; };
layout(set = 0, binding = 2) buffer Weight   { float B[]; };
layout(set = 0, binding = 3) buffer Bias     { float b[]; };

// C(MxN) = A(MxK)*B(KxN) + b(1xN)  
layout(push_constant) uniform PushConstants {
    int M;  // # of rows (batches)
    int K;  // # of inputs
    int N;  // # of outputs
};

shared float As[BM * BK]; // BMxBK
shared float Bs[BK * BN]; // BKxBN

void main() 
{
    int tileCol = int(gl_WorkGroupID.x);
    int tileRow = int(gl_WorkGroupID.y);
    int t = int(gl_LocalInvocationID.x);
    int T = int(gl_WorkGroupSize.x);        // 64 or 256 (== (BM*BN)/(TM*TN))

    // A/B 로드용 로컬 인덱스
    int innerRowA = t / BK;
    int innerColA = t % BK;
    int rowStrideA = T / BK;    // 8 or 32

    int innerRowB = t / BN;
    int innerColB = t % BN;
    int rowStrideB = T / BN;    // 1(64/64) or 2(256/128)

    // 이 스레드가 계산할 결과 블록 (TMxTN)
    int t_n = t % (BN / TN);
    int t_m = t / (BN / TN);

    // 결과와 레지스터 캐시
    float threadResults[TM * TN];
    for (int i = 0; i < TM*TN; ++i) threadResults[i] = 0.0;
    float regM[TM];
    float regN[TN];

    int m0 = tileRow * BM + innerRowA;
    int n0 = tileCol * BN + innerColB;
    int t_t = innerColA * BM + innerRowA; // transposed index of t

    for (int k0 = 0; k0 < K; k0 += BK) 
    {
        int k = k0 + innerColA;
        for (int bm = 0; bm < BM; bm += rowStrideA)  // 8(64/8) or 4(128/32)
        {
            int m = m0 + bm;
            /*
            * bm * BK + t 
            * == bm * BK + (innerRowA * BK + innerColA)
            * == (bm + innerRowA) * BK + innerColA
            * transpose => 
            * innerColA * BM + (bm + innerRowA)
            */
            As[bm + t_t] = (m < M && k < K) ? A[m * K + k] : 0.0;

        }
        
        int n = n0;
        for (int bk = 0; bk < BK; bk += rowStrideB) // 8(8/1) or 4(8/2)
        {
            k = k0 + innerRowB + bk; 
            Bs[bk * BN + t] = (k < K && n < N) ? B[k * N + n] : 0.0;
        }
        barrier();

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) 
        {
            for (int tm = 0; tm < TM; ++tm)
                regM[tm] = As[dotIdx * BM + (t_m * TM + tm)];

            for (int tn = 0; tn < TN; ++tn)
                regN[tn] = Bs[dotIdx * BN + (t_n * TN + tn)];

            for (int tm = 0; tm < TM; ++tm)
                for (int tn = 0; tn < TN; ++tn)
                    threadResults[tm*TN + tn] += regM[tm] * regN[tn];
        }
        barrier();
    }

    m0 = tileRow * BM + t_m * TM;
    n0 = tileCol * BN + t_n * TN;

    for (int tn = 0; tn < TN; ++tn) 
    {
        int n = n0 + tn;
        if (n >= N)
            break;

        float bias = b[n];
        for (int tm = 0; tm < TM; ++tm) 
        {
            int m = m0 + tm;
            if (m >= M)
                break; 
            C[m * N + n] = threadResults[tm*TN + tn] + bias;
        }
    }
})";


/////////////////////////////////////////////////////////////////////////////////////////
// MobileNetV2-specific shaders
/////////////////////////////////////////////////////////////////////////////////////////
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
                int k = kh * K + kw;
                sum += in0[input_idx] * weight[k * C + c]; // [K*K, C] row-major 가정
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

static const char* src_relu6 = R"(
#version 450
layout(local_size_x = 64) in;

layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in0[]; };

layout(push_constant) uniform PushConstants {
    int O;   // total number of elements
};

void main()
{
    int o = int(gl_GlobalInvocationID.x);
    if (o >= O) return;

    float x = in0[o];
    // ReLU6(x) = min(max(x, 0.0), 6.0)
    out0[o] = min(max(x, 0.0), 6.0);
}
)";

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

Device netGlobalDevice = VulkanApp::get().device();

/*
Descriptor : 셰이더가 접근하는 소스(GPU 메모리)의 주소/정보를 담은 구조체
 - 셰이더는 Descriptor를 통해 GPU 메모리에 접근

DescriptorSet : Descriptor들의 집합
 - 셰이더는 DescriptorSet 단위로 GPU 메모리에 접근

DescriptorPool : DescriptorSet을 대량으로 생성하기 위한 GPU-side 메모리 풀
 - DescriptorSet을 생성/해제할 때 DescriptorPool을 사용
*/
static DescriptorPool gDestSetPool = netGlobalDevice.createDescriptorPool({
    .maxTypes = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER <= 1000},
    .maxSets = 500
});


static ComputePipeline requestPipeline(const char* src)
{
    static std::unordered_map<const char*, ComputePipeline> pipelineCache;

    auto [it, inserted] = pipelineCache.try_emplace(src); // key가 없을 때만 생성하고 삽입
    if (inserted)
        it->second = netGlobalDevice.createComputePipeline({src});
    return it->second;
}

static std::map<const char*, uint32_t> gGemmTileSize =
{
    {src_gemm_naive, 32},
    {src_gemm_shared, 32},
    {src_gemm_multiOut1d, 64},
    {src_gemm_multiOut2d, 128}
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
    // src_xxx는 GPU 연산을 GLSL로 적어 놓은 shader source code
    //     - CUDA의 .cu 파일, Metal의 .metal 파일 등과 동일한 역할을 하는 GPU kernel code
    //     - Vulkan은 GLSL/HLSL -> SPIR-V로 컴파일 필요
    
    // pipeline = GPU kernel + 실행 환경(layout/descriptor)을 패키징한 것
    //     - 모든 필요한 GPU kernel을 미리 메모리에 올려두는 초기화 과정

    requestPipeline(src_relu);
    requestPipeline(src_setZero);
    requestPipeline(src_maxpool);
    requestPipeline(src_im2col);
    requestPipeline(src_gemm_naive);
    requestPipeline(src_gemm_kSplit);
    requestPipeline(src_gemm_shared);
    requestPipeline(src_gemm_multiOut1d);
    requestPipeline(src_gemm_multiOut2d);

    // MobileNetV2-specific shaders
    requestPipeline(src_depthwise_conv);
    requestPipeline(src_add);
    requestPipeline(src_global_avg_pool);
    requestPipeline(src_batchnorm);
    requestPipeline(src_relu6);
    requestPipeline(src_softmax);
}

/////////////////////////////////////////////////////////////////////////////////////////
// ConvolutionNode
/////////////////////////////////////////////////////////////////////////////////////////
ConvolutionNode::ConvolutionNode(uint32_t inChannels, uint32_t outChannels, uint32_t kernelSize, uint32_t stride, uint32_t padding)
:  C(inChannels), F(outChannels), K(kernelSize), S(stride), P(padding)
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

    // setname
    setName("ConvolutionNode");
}

void ConvolutionNode::prepare()
{
    _ASSERT((*this)["in0"].isShapeOf(-1, -1, C));
    _ASSERT((*this)["weight"].isShapeOf(C * K * K, F));

    auto& bias = (*this)["bias"];
    if (bias.shape().empty()) {
        bias = Tensor(F);
        std::vector<float> zeros(F, 0.0f);
        bias.set(zeros);
    }
    _ASSERT((*this)["bias"].isShapeOf(F));

    const auto& inShape = (*this)["in0"].shape(); // (H, W, C)
    uint32_t H = inShape[0];
    uint32_t W = inShape[1];

    uint32_t H_out = (H + 2 * P - K) / S + 1;
    uint32_t W_out = (W + 2 * P - K) / S + 1;

    (*this)["im2colOut"] = Tensor(H_out, W_out, C * K * K);
    (*this)["out0"] = Tensor(H_out, W_out, F);
}

void ConvolutionNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape  = (*this)["in0"].shape();       // (H, W, C)
    const auto& colShape = (*this)["im2colOut"].shape(); // (H_out, W_out, C*K*K)
    const auto& outShape = (*this)["out0"].shape();      // (H_out, W_out, F)

    uint32_t H = inShape[0];
    uint32_t W = inShape[1];

    uint32_t H_out = outShape[0];
    uint32_t W_out = outShape[1];

    // im2col
    im2colDescSet.write({
        (*this)["im2colOut"].buffer(),
        (*this)["in0"].buffer(),
    });

    uint32_t im2colConstants[] = {
        H, W, C, K,
        S, P,
        H_out, W_out
    };

    gemmDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["im2colOut"].buffer(),
        (*this)["weight"].buffer(),
        (*this)["bias"].buffer(),
    });

    uint32_t M = H_out * W_out;  // batch (= positions)
    uint32_t _K = C * K * K;     // input dim
    uint32_t N = F;              // output dim
    uint32_t gemmConstants[] = {M, _K, N};


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

/////////////////////////////////////////////////////////////////////////////////////////
// DepthwiseConvNode
/////////////////////////////////////////////////////////////////////////////////////////
DepthwiseConvNode::DepthwiseConvNode(uint32_t channels, uint32_t kernelWidth, uint32_t stride)
: C(channels), K(kernelWidth), S(stride)
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
    addSlot("weight", NodeSlot::input);
    addSlot("bias", NodeSlot::input);
    
    depthwiseConv = requestPipeline(src_depthwise_conv);
    depthwiseConvDescSet = depthwiseConv.descSetLayout(0).newDescSet(gDestSetPool);

    // setname
    setName("DepthwiseConvNode");
}

void DepthwiseConvNode::prepare()
{
    const auto& inShape = (*this)["in0"].shape(); // (H, W, C)
    _ASSERT(inShape.size() == 3 && inShape[2] == C);

    _ASSERT((*this)["weight"].isShapeOf(K * K, C));

    auto& bias = (*this)["bias"];
    if (bias.shape().empty()) {
        bias = Tensor(C);
        std::vector<float> zeros(C, 0.0f);
        bias.set(zeros);
    }
    _ASSERT((*this)["bias"].isShapeOf(C));

    uint32_t H = inShape[0];
    uint32_t W = inShape[1];

    uint32_t padding = K / 2;

    uint32_t H_out = (H + 2 * padding - K) / S + 1;
    uint32_t W_out = (W + 2 * padding - K) / S + 1;

    (*this)["out0"] = Tensor(H_out, W_out, C);
}

void DepthwiseConvNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    const auto& outShape = (*this)["out0"].shape();

    uint32_t H = inShape[0];
    uint32_t W = inShape[1];

    uint32_t H_out = outShape[0];
    uint32_t W_out = outShape[1];

    uint32_t Stride = S;
    uint32_t Padding = K / 2;
    
    depthwiseConvDescSet.write({ // binding 순서대로 내부 버퍼 주소 매핑
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
        (*this)["weight"].buffer(),
        (*this)["bias"].buffer(),
    });

    uint32_t constants[] = {H, W, C, K, Stride, Padding, H_out, W_out};

    cmdBuff
        .bindPipeline(depthwiseConv)
        .bindDescSets({depthwiseConvDescSet})
        .setPushConstants(0, sizeof(constants), constants)
        .dispatch(CEIL_DIV(W_out, 16), CEIL_DIV(H_out, 16), C)
        .barrier( 
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}

/////////////////////////////////////////////////////////////////////////////////////////
// PointwiseConvNode
/////////////////////////////////////////////////////////////////////////////////////////
PointwiseConvNode::PointwiseConvNode(uint32_t inChannels, uint32_t outChannels)
: C(inChannels), F(outChannels)
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
    addSlot("weight", NodeSlot::input);
    addSlot("bias", NodeSlot::input);

    const char* gemmSrc = src_gemm_shared;

    gemm = requestPipeline(gemmSrc);
    gemmTileSize = gGemmTileSize.at(gemmSrc);
    gemmDescSet = gemm.descSetLayout(0).newDescSet(gDestSetPool);

    _ASSERT(gemmTileSize == 32);

    // setname
    setName("PointwiseConvNode");
}

void PointwiseConvNode::prepare()
{
    _ASSERT((*this)["in0"].isShapeOf(-1, -1, C));
    _ASSERT((*this)["weight"].isShapeOf(C, F));

    auto& bias = (*this)["bias"];
    if (bias.shape().empty()) {
        bias = Tensor(F);
        std::vector<float> zeros(F, 0.0f);
        bias.set(zeros);
    }
    _ASSERT((*this)["bias"].isShapeOf(F));

    const auto& inShape = (*this)["in0"].shape();
    (*this)["out0"] = Tensor(inShape[0], inShape[1], F);
}

void PointwiseConvNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    uint32_t H = inShape[0];
    uint32_t W = inShape[1];

    gemmDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
        (*this)["weight"].buffer(),
        (*this)["bias"].buffer(),
    });

    uint32_t M = H * W;    // Num of batches
    uint32_t K = C;        // Input size
    uint32_t N = F;        // Output size
    uint32_t gemmConstants[] = {M, K, N};

    cmdBuff
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

/////////////////////////////////////////////////////////////////////////////////////////
// AddNode
/////////////////////////////////////////////////////////////////////////////////////////
AddNode::AddNode()
{
    addSlot("in0", NodeSlot::input);
    addSlot("in1", NodeSlot::input);
    addSlot("out0", NodeSlot::output);

    add = requestPipeline(src_add);
    addDescSet = add.descSetLayout(0).newDescSet(gDestSetPool);

    // setname
    setName("AddNode");
}

void AddNode::prepare()
{
    _ASSERT((*this)["in0"].validShape());
    _ASSERT((*this)["in1"].validShape());
    _ASSERT((*this)["in0"].shape() == (*this)["in1"].shape());

    (*this)["out0"] = Tensor((*this)["in0"].shape()); // output shape = input shape
}

void AddNode::run(CommandBuffer cmdBuff)
{
    const auto& N = (*this)["out0"].numElements(); 
    
    addDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
        (*this)["in1"].buffer(),
    });

    cmdBuff
        .bindPipeline(add)
        .bindDescSets({addDescSet})
        .setPushConstants(0, sizeof(N), &N)
        .dispatch(N)
        .barrier( 
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}

/////////////////////////////////////////////////////////////////////////////////////////
// BatchNormNode
/////////////////////////////////////////////////////////////////////////////////////////
BatchNormNode::BatchNormNode(float epsilon)
: eps(epsilon)
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
    addSlot("mean", NodeSlot::input);
    addSlot("variance", NodeSlot::input);
    addSlot("gamma", NodeSlot::input);
    addSlot("beta", NodeSlot::input);

    batchNorm = requestPipeline(src_batchnorm);
    batchNormDescSet = batchNorm.descSetLayout(0).newDescSet(gDestSetPool);

    // setname
    setName("BatchNormNode");
}

void BatchNormNode::prepare()
{
    const auto& inShape = (*this)["in0"].shape();
    _ASSERT(inShape.size() == 3);

    uint32_t C = inShape[2];
    ensureTensorFilled((*this)["mean"], 0.0f, C); // if not filled, fill with default values
    ensureTensorFilled((*this)["variance"], 1.0f, C);
    ensureTensorFilled((*this)["gamma"], 1.0f, C);
    ensureTensorFilled((*this)["beta"], 0.0f, C);

    _ASSERT((*this)["mean"].isShapeOf(C));
    _ASSERT((*this)["variance"].isShapeOf(C));
    _ASSERT((*this)["gamma"].isShapeOf(C));
    _ASSERT((*this)["beta"].isShapeOf(C));

    (*this)["out0"] = Tensor((*this)["in0"].shape());
}

void BatchNormNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    uint32_t H = inShape[0];
    uint32_t W = inShape[1];
    uint32_t C = inShape[2];

    batchNormDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
        (*this)["mean"].buffer(),
        (*this)["variance"].buffer(),
        (*this)["gamma"].buffer(),
        (*this)["beta"].buffer(),
    });

    uint32_t batchnormConstants[] = {H, W, C};
    float epsilon = eps;

    cmdBuff
        .bindPipeline(batchNorm)
        .bindDescSets({batchNormDescSet})
        .setPushConstants(0, sizeof(batchnormConstants), batchnormConstants)
        .setPushConstants(sizeof(batchnormConstants), sizeof(epsilon), &epsilon)
        .dispatch(H * W, C)
        .barrier( 
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}

/////////////////////////////////////////////////////////////////////////////////////////
// ReluNode
/////////////////////////////////////////////////////////////////////////////////////////
ReluNode::ReluNode()
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);

    relu = requestPipeline(src_relu);
    reluDescSet = relu.descSetLayout(0).newDescSet(gDestSetPool);

    // setname
    setName("ReluNode");
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

/////////////////////////////////////////////////////////////////////////////////////////
// Relu6Node
/////////////////////////////////////////////////////////////////////////////////////////
Relu6Node::Relu6Node()
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);

    relu6 = requestPipeline(src_relu6);
    relu6DescSet = relu6.descSetLayout(0).newDescSet(gDestSetPool);

    // setname
    setName("Relu6Node");
}

void Relu6Node::prepare()
{
    _ASSERT((*this)["in0"].validShape());
    (*this)["out0"] = Tensor((*this)["in0"].shape());
}

void Relu6Node::run(CommandBuffer cmdBuff) {
    const auto& inShape = (*this)["in0"].shape();
    int I = 1;
    for (int dim : inShape) I *= dim;

    relu6DescSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
    });

    int constants[] = {I};
    cmdBuff
        .bindPipeline(relu6)
        .setPushConstants(0, sizeof(constants), constants)
        .bindDescSets({relu6DescSet})
        .dispatch(I)
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}

/////////////////////////////////////////////////////////////////////////////////////////
// MaxPoolingNode
/////////////////////////////////////////////////////////////////////////////////////////
MaxPoolingNode::MaxPoolingNode(uint32_t poolSize)
: P(poolSize)
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);

    maxpool = requestPipeline(src_maxpool);
    maxpoolDescSet = maxpool.descSetLayout(0).newDescSet(gDestSetPool);

    // setname
    setName("MaxPoolingNode");
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

/////////////////////////////////////////////////////////////////////////////////////////
// GlobalAvgPoolNode
/////////////////////////////////////////////////////////////////////////////////////////
GlobalAvgPoolNode::GlobalAvgPoolNode()
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);

    globalAvgPool = requestPipeline(src_global_avg_pool);
    globalAvgPoolDescSet = globalAvgPool.descSetLayout(0).newDescSet(gDestSetPool);

    // setname
    setName("GlobalAvgPoolNode");
}

void GlobalAvgPoolNode::prepare()
{
    const auto& inShape = (*this)["in0"].shape();
    _ASSERT(inShape.size() == 3);
    uint32_t C = inShape[2];

    (*this)["out0"] = Tensor(1, 1, C); // (H, W, C) -> (1, 1, C)
}

void GlobalAvgPoolNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    uint32_t H = inShape[0]; 
    uint32_t W = inShape[1];
    uint32_t C = inShape[2];

    globalAvgPoolDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
    });

    uint32_t constants[] = {H, W, C};

    cmdBuff
        .bindPipeline(globalAvgPool)
        .bindDescSets({globalAvgPoolDescSet})
        .setPushConstants(0, sizeof(constants), constants) // set constants to shader
        .dispatch(C)
        .barrier( 
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}

/////////////////////////////////////////////////////////////////////////////////////////
// FlattenNode
/////////////////////////////////////////////////////////////////////////////////////////
FlattenNode::FlattenNode()
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);

    // setname
    setName("FlattenNode");
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

/////////////////////////////////////////////////////////////////////////////////////////
// SoftmaxNode
/////////////////////////////////////////////////////////////////////////////////////////
SoftmaxNode::SoftmaxNode()
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);

    softmax = requestPipeline(src_softmax);
    softmaxDescSet = softmax.descSetLayout(0).newDescSet(gDestSetPool);

    // setname
    setName("SoftmaxNode");
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
        .bindPipeline(softmax)
        .setPushConstants(0, sizeof(constants), constants)
        .bindDescSets({softmaxDescSet})
        .dispatch0(CEIL_DIV(num_rows, 64))
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / output.buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}

/////////////////////////////////////////////////////////////////////////////////////////
// FullyConnectedNode
/////////////////////////////////////////////////////////////////////////////////////////
FullyConnectedNode::FullyConnectedNode(uint32_t inDim, uint32_t outDim)
: I(inDim), O(outDim) 
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
    addSlot("weight", NodeSlot::input);
    addSlot("bias", NodeSlot::input);

    /*const char* gemmSrc = src_gemm_naive;
    gemmTileSize = gGemmTileSize.at(gemmSrc);*/

    const char* gemmSrc = src_gemm_kSplit;

    gemm = requestPipeline(gemmSrc);
    gemmDescSet = gemm.descSetLayout(0).newDescSet(gDestSetPool);

	setZero = requestPipeline(src_setZero);
    setZeroDescSet = setZero.descSetLayout(0).newDescSet(gDestSetPool);

    // setname
    setName("FullyConnectedNode");
}

void FullyConnectedNode::prepare() 
{
    _ASSERT((*this)["in0"].isShapeOf(I));
    _ASSERT((*this)["weight"].isShapeOf(I, O));

    auto& bias = (*this)["bias"];
    if (bias.shape().empty()) {
        bias = Tensor(O);
        std::vector<float> zeros(O, 0.0f);
        bias.set(zeros);
    }
    _ASSERT((*this)["bias"].isShapeOf(O));
    (*this)["out0"] = Tensor(O);
}

void FullyConnectedNode::run(CommandBuffer cmdBuff) 
{
    uint32_t M = 1;
    uint32_t K = (*this)["in0"].shape()[0];
    uint32_t N = (*this)["out0"].shape()[0];
    
    setZeroDescSet.write({
        (*this)["out0"].buffer(),
    });
    gemmDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
        (*this)["weight"].buffer(),
        (*this)["bias"].buffer(),
    });

	uint32_t setZeroConstants[] = { N };
    uint32_t gemmConstants[] = {M, K, N};

    cmdBuff
		.bindPipeline(setZero)
		.bindDescSets({setZeroDescSet})
		.setPushConstants(0, sizeof(setZeroConstants), setZeroConstants)
		.dispatch(N)
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
        )

        .bindPipeline(gemm)
        .bindDescSets({gemmDescSet})
        .setPushConstants(0, sizeof(gemmConstants), gemmConstants)
        .dispatch0(CEIL_DIV(N, 32), M, CEIL_DIV(K, 16))
        //.dispatch0(CEIL_DIV(N, gemmTileSize), CEIL_DIV(M, gemmTileSize))
        .barrier( 
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );

}  

/////////////////////////////////////////////////////////////////////////////////////////
// Node Groups
/////////////////////////////////////////////////////////////////////////////////////////
ConvBNReLU6::ConvBNReLU6(uint32_t inChannels,
                         uint32_t outChannels,
                         uint32_t kernel,
                         uint32_t stride,
                         uint32_t padding)
{
    // 1) primitive Node 생성
    conv = std::make_unique<ConvolutionNode>(inChannels,
                                             outChannels,
                                             kernel,
                                             stride,
                                             padding);

    bn   = std::make_unique<BatchNormNode>();
    relu = std::make_unique<Relu6Node>();  
    
    // 2) 내부 그래프 연결
    // conv(out) -> bn(in) -> relu(in) -> relu(out) …
    defineSlot("in0",  conv->slot("in0"));
    conv->slot("out0") - bn->slot("in0");
    bn->slot("out0") - relu->slot("in0");
    defineSlot("out0", relu->slot("out0"));
}

Tensor& ConvBNReLU6::operator[](const std::string& slotName)
{
    // ----------------------------
    // ConvolutionNode weights
    // ----------------------------
    if (slotName == "conv.weight")
        return conv->slot("weight").getValueRef();

    if (slotName == "conv.bias")
        return conv->slot("bias").getValueRef();

    // ----------------------------
    // BatchNormNode parameters
    // ----------------------------
    if (slotName == "bn.mean")
        return bn->slot("mean").getValueRef();

    if (slotName == "bn.variance")
        return bn->slot("variance").getValueRef();

    if (slotName == "bn.gamma")
        return bn->slot("gamma").getValueRef();

    if (slotName == "bn.beta")
        return bn->slot("beta").getValueRef();

    // Unhandled name
    throw std::runtime_error(
        ">> ConvBNReLU6: Tensor not found for name: " + slotName
    );
}

PWConvBNReLU6::PWConvBNReLU6(uint32_t inChannels,
                             uint32_t outChannels)
{
    // 1) primitive Node 생성
    pointwiseConv = std::make_unique<PointwiseConvNode>(inChannels,
                                                 outChannels);

    bn     = std::make_unique<BatchNormNode>();
    relu   = std::make_unique<Relu6Node>();  
    
    // 2) 내부 그래프 연결
    // pwConv(out) -> bn(in) -> relu(in) -> relu(out) …
    defineSlot("in0", pointwiseConv->slot("in0"));
    pointwiseConv->slot("out0") - bn->slot("in0");
    bn->slot("out0") - relu->slot("in0");
    defineSlot("out0", relu->slot("out0"));
}

Tensor& PWConvBNReLU6::operator[](const std::string& slotName)
{
    // ----------------------------
    // PointwiseConvNode weights
    // ----------------------------
    if (slotName == "pointwiseConv.weight")
        return pointwiseConv->slot("weight").getValueRef();

    if (slotName == "pointwiseConv.bias")
        return pointwiseConv->slot("bias").getValueRef();

    // ----------------------------
    // BatchNormNode parameters
    // ----------------------------
    if (slotName == "bn.mean")
        return bn->slot("mean").getValueRef();

    if (slotName == "bn.variance")
        return bn->slot("variance").getValueRef();

    if (slotName == "bn.gamma")
        return bn->slot("gamma").getValueRef();

    if (slotName == "bn.beta")
        return bn->slot("beta").getValueRef();

    // Unhandled name
    throw std::runtime_error(
        ">> PWConvBNReLU6: Tensor not found for name: " + slotName
    );
}

PWConvBN::PWConvBN(uint32_t inChannels, uint32_t outChannels)
{
    // 1) primitive Node 생성
    pointwiseConv = std::make_unique<PointwiseConvNode>(inChannels, outChannels);
    bn            = std::make_unique<BatchNormNode>();
    
    // 2) 내부 그래프 연결
    // pwConv(out) -> bn(in) -> bn(out) …
    defineSlot("in0", pointwiseConv->slot("in0"));
    pointwiseConv->slot("out0") - bn->slot("in0");
    defineSlot("out0", bn->slot("out0"));
}

Tensor& PWConvBN::operator[](const std::string& slotName)
{
    // ----------------------------
    // PointwiseConvNode weights
    // ----------------------------
    if (slotName == "pointwiseConv.weight")
        return pointwiseConv->slot("weight").getValueRef();

    if (slotName == "pointwiseConv.bias")
        return pointwiseConv->slot("bias").getValueRef();

    // ----------------------------
    // BatchNormNode parameters
    // ----------------------------
    if (slotName == "bn.mean")
        return bn->slot("mean").getValueRef();

    if (slotName == "bn.variance")
        return bn->slot("variance").getValueRef();

    if (slotName == "bn.gamma")
        return bn->slot("gamma").getValueRef();

    if (slotName == "bn.beta")
        return bn->slot("beta").getValueRef();

    // Unhandled name
    throw std::runtime_error(
        ">> PWConvBN: Tensor not found for name: " + slotName
    );
}

DWConvBNReLU6::DWConvBNReLU6(uint32_t channels,
                             uint32_t kernel,
                             uint32_t stride)
{
    // 1) primitive Node 생성
    depthwiseConv = std::make_unique<DepthwiseConvNode>(channels,
                                                        kernel,
                                                        stride);

    bn     = std::make_unique<BatchNormNode>();
    relu   = std::make_unique<Relu6Node>();  
    
    // 2) 내부 그래프 연결
    // dwConv(out) -> bn(in) -> relu(in) -> relu(out) …
    defineSlot("in0", depthwiseConv->slot("in0"));
    depthwiseConv->slot("out0") - bn->slot("in0");
    bn->slot("out0") - relu->slot("in0");
    defineSlot("out0", relu->slot("out0"));
}

Tensor& DWConvBNReLU6::operator[](const std::string& slotName)
{
    // ----------------------------
    // DepthwiseConvNode weights
    // ----------------------------
    if (slotName == "depthwiseConv.weight")
        return depthwiseConv->slot("weight").getValueRef();

    if (slotName == "depthwiseConv.bias")
        return depthwiseConv->slot("bias").getValueRef();

    // ----------------------------
    // BatchNormNode parameters
    // ----------------------------
    if (slotName == "bn.mean")
        return bn->slot("mean").getValueRef();

    if (slotName == "bn.variance")
        return bn->slot("variance").getValueRef();

    if (slotName == "bn.gamma")
        return bn->slot("gamma").getValueRef();

    if (slotName == "bn.beta")
        return bn->slot("beta").getValueRef();

    // Unhandled name
    throw std::runtime_error(
        ">> DWConvBNReLU6: Tensor not found for name: " + slotName
    );
}

InvertedResidualBlock::InvertedResidualBlock(uint32_t inChannels,
                                             uint32_t outChannels,
                                             uint32_t expansionFactor,
                                             uint32_t stride)
{
    const uint32_t expandedChannels = inChannels * expansionFactor;
    useResidual = ((stride == 1) && (inChannels == outChannels));

    NodeSlot* cur = nullptr;
    NodeSlot* skipOut = nullptr;

    if (useResidual)
    {
        inputSplit = std::make_unique<InputNode>();
        defineSlot("in0", inputSplit->slot("in0"));

        cur = &inputSplit->slot("out0");
        skipOut = cur;
    }

    // 1) expand
    if (expansionFactor != 1)
    {
        pwConvBNReLU6 = std::make_unique<PWConvBNReLU6>(inChannels, expandedChannels);

        if (useResidual)
            *cur - pwConvBNReLU6->slot("in0");
        else
            defineSlot("in0", pwConvBNReLU6->slot("in0"));

        cur = &pwConvBNReLU6->slot("out0");
    }

    // 2) depthwise (여기가 expansionFactor==1인 경우의 첫 연산이 될 수 있음)
    const uint32_t dwInChannels = (expansionFactor != 1) ? expandedChannels : inChannels;
    dwConvBNReLU6 = std::make_unique<DWConvBNReLU6>(dwInChannels, 3, stride);

    if (cur)
    {
        *cur - dwConvBNReLU6->slot("in0");
    }
    else
    {
        // useResidual == false 이면서 expand도 없는 케이스
        defineSlot("in0", dwConvBNReLU6->slot("in0"));
    }

    cur = &dwConvBNReLU6->slot("out0");

    // 3) project
    pwConvBN = std::make_unique<PWConvBN>(dwInChannels, outChannels);
    *cur - pwConvBN->slot("in0");
    cur = &pwConvBN->slot("out0");

    // 4) residual add
    if (useResidual)
    {
        add = std::make_unique<AddNode>();
        *skipOut - add->slot("in0");
        *cur     - add->slot("in1");
        defineSlot("out0", add->slot("out0"));
    }
    else
    {
        defineSlot("out0", *cur);
    }
}

Tensor& InvertedResidualBlock::operator[](const std::string& slotName)
{
    if (slotName.rfind("pwConvBNReLU6.", 0) == 0) {
        return (*pwConvBNReLU6)[slotName.substr(14)];
    }

    if (slotName.rfind("dwConvBNReLU6.", 0) == 0) {
        return (*dwConvBNReLU6)[slotName.substr(14)];
    }

    if (slotName.rfind("pwConvBN.", 0) == 0) {
        return (*pwConvBN)[slotName.substr(9)];
    }

    throw std::runtime_error("InvertedResidualBlock: Tensor not found: " + slotName);
}

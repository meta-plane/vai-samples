#include "neuralNodes.h"
#include <unordered_map>
#include <iostream>
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

static const char* src_batchnorm1d = R"(
#version 450
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out_data[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in_data[]; };
layout(set = 0, binding = 2) buffer MeanBuffer { float mean[]; };
layout(set = 0, binding = 3) buffer VarBuffer { float var[]; };
layout(set = 0, binding = 4) buffer GammaBuffer { float gamma[]; };
layout(set = 0, binding = 5) buffer BetaBuffer { float beta[]; };

layout(push_constant) uniform PushConstants {
    uint C;       // channels (outer dimension)
    uint N;       // num_points (inner dimension)
    float eps;    // epsilon (1e-5)
};

void main() 
{
    uint idx = gl_GlobalInvocationID.x;
    uint total = C * N;
    if (idx >= total) return;
    
    uint c = idx / N;  // channel index (outer)
    uint n = idx % N;  // point index (inner)
    
    // BatchNorm: out = gamma * (in - mean) / sqrt(var + eps) + beta
    // Layout: [C, N] - data[c * N + n]
    float x = in_data[idx];
    float normalized = (x - mean[c]) / sqrt(var[c] + eps);
    out_data[idx] = gamma[c] * normalized + beta[c];
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
layout(local_size_x = 32, local_size_y = 4, local_size_z = 4) in;
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

static const char* src_maxpool1d = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in0[]; };
layout(push_constant) uniform PushConstants {
    int C; // number of channels (outer dimension)
    int N; // number of points (inner dimension)
};

void main() {
    int c = int(gl_GlobalInvocationID.x);
    if (c >= C) return;

    float m = -3.4e38;
    for (int n = 0; n < N; ++n) {
        float v = in0[c * N + n];  // [C, N] layout
        m = max(m, v);
    }
    out0[c] = m;
})";

static const char* src_im2col = R"(
#version 450
layout(local_size_x = 64, local_size_y = 16) in;
layout(set = 0, binding = 0) writeonly buffer OutBuffer { float im2colOut[]; };
layout(set = 0, binding = 1) readonly buffer InBuffer { float in0[]; };
layout(push_constant) uniform PushConstants {
    int H;
    int W;
    int C;
    int K;
};

void main() 
{
    int i = int(gl_GlobalInvocationID.x); 
    int j = int(gl_GlobalInvocationID.y); 
    int KK = K * K;
    int CKK = C * KK;
    if (i >= H * W || j >= CKK) 
        return;

    int h = i / W;          // image center row
    int w = i % W;          // image center col
    int c = j / KK;         // image channel
    int K_2 = K / 2;
    int k = j % KK;

    float value = 0.0;
    h += k / K - K_2;  
    w += k % K - K_2;   
    if (0 <= h && h < H && 0 <= w && w < W) 
        value = in0[((h * W) + w) * C + c];

    im2colOut[i * CKK + j] = value;
})";

static const char* src_gemm_naive = R"(
#version 450
layout(local_size_x = 32, local_size_y = 32) in;
layout(set = 0, binding = 0) buffer OutBuffer { float C[]; };
layout(set = 0, binding = 1) buffer InBuffer { float A[]; };
layout(set = 0, binding = 2) buffer Weight { float B[]; };
layout(set = 0, binding = 3) buffer Bias { float b[]; };

// C(NxM) = B(NxK)*A(KxM) + b(Nx1)
// PyTorch: [OutCh, InCh] @ [InCh, Points] = [OutCh, Points]
layout(push_constant) uniform PushConstants {
    int N;  // # of output channels (outer dimension)
    int K;  // # of input channels
    int M;  // # of points (inner dimension)
};

void main() 
{
    int m = int(gl_GlobalInvocationID.x);  // point index (inner)
    int n = int(gl_GlobalInvocationID.y);  // output channel (outer)

    if (n >= N || m >= M) 
        return;

    float sum = b[n];
    for (int k = 0; k < K; ++k)
        sum += B[n * K + k] * A[k * M + m];

    C[n * M + m] = sum;
}
)";

//static const char* gemm_srcCode_vec = R"(
//#version 450
//layout(local_size_x = 32, local_size_y = 32) in;
//layout(set = 0, binding = 0) buffer OutBuffer { vec4 C[]; };
//layout(set = 0, binding = 1) buffer InBuffer { vec4 A[]; };
//layout(set = 0, binding = 2) buffer Weight { vec4 B[]; };
//layout(set = 0, binding = 3) buffer Bias { float b[]; };
//
//// C(MxN) = A(MxK)*B(KxN) + b(1xN)
//layout(push_constant) uniform PushConstants {
//    int M;  // # of batchs
//    int K;  // # of inputs
//    int N;  // # of outputs
//};
//
//void main() 
//{
//    int n = int(gl_GlobalInvocationID.x); 
//    int m = int(gl_GlobalInvocationID.y); 
//
//    if (m >= M || n >= N) 
//        return;
//
//    float sum = b[n];
//    for (int k = 0; k < K; ++k)
//        sum += A[m * K + k] * B[k * N + n];
//
//    C[m * N + n] = sum;
//})";

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

// C(NxM) = B(NxK)*A(KxM) + b(Nx1)
// PyTorch: [OutCh, InCh] @ [InCh, Points] = [OutCh, Points]
layout(push_constant) uniform PushConstants {
    int N;  // # of output channels (rows in output)
    int K;  // # of input channels
    int M;  // # of points (cols in output)
};

shared float Bs[BN * BK]; // BNxBK for B tiles
shared float As[BK * BM]; // BKxBM for A tiles

void main() 
{
    int tileCol = int(gl_WorkGroupID.x);  // M direction (points)
    int tileRow = int(gl_WorkGroupID.y);  // N direction (output channels)
    int t = int(gl_LocalInvocationID.x);
    int T = int(gl_WorkGroupSize.x);        // 64 or 256 (== (BM*BN)/(TM*TN))

    // B loading indices (B is [N, K])
    int innerRowB = t / BK;
    int innerColB = t % BK;
    int rowStrideB = T / BK;    // 8 or 32

    // A loading indices (A is [K, M])
    int innerRowA = t / BM;
    int innerColA = t % BM;
    int rowStrideA = T / BM;    // 1(64/64) or 2(256/128)

    // Thread result block (TNxTM)
    int t_m = t % (BM / TM);
    int t_n = t / (BM / TM);

    // Result and register cache
    float threadResults[TN * TM];
    for (int i = 0; i < TN*TM; ++i) threadResults[i] = 0.0;
    float regN[TN];
    float regM[TM];

    int n0 = tileRow * BN + innerRowB;
    int m0 = tileCol * BM + innerColA;
    int t_t = innerColB * BN + innerRowB; // transposed index of t

    for (int k0 = 0; k0 < K; k0 += BK) 
    {
        // Load B tile [BN, BK] from B[N, K]
        int k = k0 + innerColB;
        for (int bn = 0; bn < BN; bn += rowStrideB)
        {
            int n = n0 + bn;
            Bs[bn + t_t] = (n < N && k < K) ? B[n * K + k] : 0.0;
        }
        
        // Load A tile [BK, BM] from A[K, M]
        int m = m0;
        for (int bk = 0; bk < BK; bk += rowStrideA)
        {
            k = k0 + innerRowA + bk; 
            As[bk * BM + t] = (k < K && m < M) ? A[k * M + m] : 0.0;
        }
        barrier();

        // Compute
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) 
        {
            for (int tn = 0; tn < TN; ++tn)
                regN[tn] = Bs[dotIdx * BN + (t_n * TN + tn)];

            for (int tm = 0; tm < TM; ++tm)
                regM[tm] = As[dotIdx * BM + (t_m * TM + tm)];

            for (int tn = 0; tn < TN; ++tn)
                for (int tm = 0; tm < TM; ++tm)
                    threadResults[tn*TM + tm] += regN[tn] * regM[tm];
        }
        barrier();
    }

    n0 = tileRow * BN + t_n * TN;
    m0 = tileCol * BM + t_m * TM;

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
            C[n * M + m] = threadResults[tn*TM + tm] + bias;
        }
    }
})";


// Broadcast shader: repeat [C, 1] to [C, N]
static const char* src_broadcast = R"(
#version 450
layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer In0 { float global_feature[]; };  // [C, 1]
layout(binding = 1) writeonly buffer Out0 { float output_data[]; };   // [C, N]

layout(push_constant) uniform PushConstants {
    uint C;  // channels (outer dimension)
    uint N;  // points (inner dimension)
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= C * N) return;
    
    uint c = idx / N;  // channel (outer)
    uint n = idx % N;  // point (inner)
    output_data[idx] = global_feature[c];  // idx = c * N + n
}
)";

// Concat shader: concatenate [C1, N] + [C2, N] → [C1+C2, N]
static const char* src_concat = R"(
#version 450
layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer In0 { float input0[]; };   // [C1, N]
layout(binding = 1) readonly buffer In1 { float input1[]; };   // [C2, N]
layout(binding = 2) writeonly buffer Out0 { float output_data[]; };  // [C1+C2, N]

layout(push_constant) uniform PushConstants {
    uint C1;  // channels in first input
    uint C2;  // channels in second input
    uint N;   // points
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = (C1 + C2) * N;
    if (idx >= total) return;
    
    uint c = idx / N;  // channel (outer)
    uint n = idx % N;  // point (inner)
    
    if (c < C1) {
        output_data[idx] = input0[c * N + n];
    } else {
        output_data[idx] = input1[(c - C1) * N + n];
    }
}
)";

// Slice shader: extract channels [start, end) from [C_in, N] → [slice_size, N]
static const char* src_slice = R"(
#version 450
layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer In { float input_data[]; };   // [C_in, N]
layout(binding = 1) writeonly buffer Out { float output_data[]; }; // [slice_size, N]

layout(push_constant) uniform PushConstants {
    uint C_in;         // input channels (outer dimension)
    uint N;            // number of points (inner dimension)
    uint start_ch;     // start channel index
    uint slice_size;   // number of channels to slice (end - start)
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = slice_size * N;
    if (idx >= total) return;
    
    uint c_out = idx / N;      // output channel index (outer)
    uint n = idx % N;          // point index (inner)
    
    uint c_in = start_ch + c_out;
    uint in_idx = c_in * N + n;
    output_data[idx] = input_data[in_idx];
}
)";

// Matrix multiplication shader: C[M, N] = B[M, K] @ A[K, N]
// No bias addition (pure matrix multiplication)
// For TNet: transformation matrix [3,3] or [64,64]
static const char* src_matmul = R"(
#version 450
layout(local_size_x = 16, local_size_y = 16) in;
layout(set = 0, binding = 0) buffer OutBuffer { float C[]; };  // [M, N]
layout(set = 0, binding = 1) buffer InBuffer  { float A[]; };  // [K, N]
layout(set = 0, binding = 2) buffer Weight    { float B[]; };  // [M, K]

layout(push_constant) uniform PushConstants {
    int M;  // # of output dim (rows in output)
    int K;  // # of intermediate dim
    int N;  // # of points (cols in output)
};

void main() 
{
    int n = int(gl_GlobalInvocationID.x);  // point index (inner, col)
    int m = int(gl_GlobalInvocationID.y);  // output dim (outer, row)

    if (m >= M || n >= N) 
        return;

    float sum = 0.0;
    for (int k = 0; k < K; ++k)
        sum += B[m * K + k] * A[k * N + n];

    C[m * N + n] = sum;
}
)";


Device netGlobalDevice = VulkanApp::get().device();

static DescriptorPool gDestSetPool = netGlobalDevice.createDescriptorPool({
    .maxTypes = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER <= 20000}, 
    .maxSets = 100000
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
    {src_gemm_multiOut1d, 64},
    {src_gemm_multiOut2d, 128}
};

void loadShaders()
{
    requestPipeline(src_relu);
    requestPipeline(src_batchnorm1d);
    // requestPipeline(src_copy);
    requestPipeline(src_setZero);
    requestPipeline(src_maxpool);
    requestPipeline(src_im2col);
    requestPipeline(src_gemm_naive);
    requestPipeline(src_gemm_kSplit);
    // requestPipeline(src_gemm_kSplit2);
    requestPipeline(src_gemm_shared);
    requestPipeline(src_gemm_multiOut1d);
    requestPipeline(src_gemm_multiOut2d);

}

/////////////////////////////////////////////////////////////////////////////////////////
// ConvolutionNode
/////////////////////////////////////////////////////////////////////////////////////////
ConvolutionNode::ConvolutionNode(uint32_t inChannels, uint32_t outChannels, uint32_t kernelWidth)
:  C(inChannels), F(outChannels), K(kernelWidth)
{
    setName("ConvolutionNode");
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
    _ASSERT((*this)["weight"].validShape());
    _ASSERT((*this)["weight"].isShapeOf(C*K*K, F));
    _ASSERT((*this)["bias"].validShape());
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

    uint32_t im2colConstants[] = {H, W, C, K};
    // uint32_t gemmConstants[] = {H * W, C * K * K, F};

    uint32_t M = H * W;         // N
    uint32_t K_ = C * K * K;    // I
    uint32_t N = F;             // O
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
        // .dispatch(F, H * W)
        .dispatch0(CEIL_DIV(N, gemmTileSize), CEIL_DIV(M, gemmTileSize))
        .barrier( 
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}  


/////////////////////////////////////////////////////////////////////////////////////////
// PointWiseMLPNode (Conv1x1 + BatchNorm1D + ReLU)
/////////////////////////////////////////////////////////////////////////////////////////
PointWiseMLPNode::PointWiseMLPNode(uint32_t inDim, uint32_t outDim)
: Cin(inDim)
, Cout(outDim)
{
    setName("PointWiseMLPNode");
    // Slots for full MLP chain
    addSlot("in0", NodeSlot::input);      // [N, Cin]
    addSlot("out0", NodeSlot::output);    // [N, Cout]
    
    // Conv weights
    addSlot("weight", NodeSlot::input);   // [Cin, Cout]
    addSlot("bias", NodeSlot::input);     // [Cout]
    
    // BatchNorm parameters
    addSlot("bn_mean", NodeSlot::input);  // [Cout]
    addSlot("bn_var", NodeSlot::input);   // [Cout]
    addSlot("bn_gamma", NodeSlot::input); // [Cout]
    addSlot("bn_beta", NodeSlot::input);  // [Cout]
    
    // Intermediate tensors for pipeline stages
    addSlot("conv_out", NodeSlot::internal);  // After Conv
    addSlot("bn_out", NodeSlot::internal);    // After BatchNorm

    // Create pipelines for each stage
    gemm = requestPipeline(src_gemm_shared);
    gemmDesc = gemm.descSetLayout(0).newDescSet(gDestSetPool);
    
    batchnorm = requestPipeline(src_batchnorm1d);
    batchnormDesc = batchnorm.descSetLayout(0).newDescSet(gDestSetPool);
    
    relu = requestPipeline(src_relu);
    reluDesc = relu.descSetLayout(0).newDescSet(gDestSetPool);
}

void PointWiseMLPNode::prepare()
{
    const auto& inShape = (*this)["in0"].shape();
    _ASSERT(inShape.size() == 2);
    _ASSERT(inShape[0] == Cin);  // [Cin, N] layout - channels is outer dimension
    _ASSERT((*this)["weight"].validShape());
    _ASSERT((*this)["weight"].isShapeOf(Cout, Cin));  // PyTorch format: [Cout, Cin]
    _ASSERT((*this)["bias"].validShape());
    _ASSERT((*this)["bias"].isShapeOf(Cout));

    (*this)["out0"] = Tensor(Cout, inShape[1]);      // [Cout, N]
    (*this)["conv_out"] = Tensor(Cout, inShape[1]);  // [Cout, N]
    (*this)["bn_out"] = Tensor(Cout, inShape[1]);    // [Cout, N]
    
    // Set default BatchNorm parameters if not provided
    if (!(*this)["bn_mean"].validShape()) {
        std::vector<float> zeros(Cout, 0.0f);
        (*this)["bn_mean"] = Tensor(Cout).set(zeros);
    }
    if (!(*this)["bn_var"].validShape()) {
        std::vector<float> ones(Cout, 1.0f);
        (*this)["bn_var"] = Tensor(Cout).set(ones);
    }
    if (!(*this)["bn_gamma"].validShape()) {
        std::vector<float> ones(Cout, 1.0f);
        (*this)["bn_gamma"] = Tensor(Cout).set(ones);
    }
    if (!(*this)["bn_beta"].validShape()) {
        std::vector<float> zeros(Cout, 0.0f);
        (*this)["bn_beta"] = Tensor(Cout).set(zeros);
    }
}

void PointWiseMLPNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    uint32_t Cin_actual = inShape[0];  // input channels (outer)
    uint32_t M = inShape[1];           // points (inner)

    // Stage 1: Conv1x1 (GEMM)
    // C(NxM) = B(NxK)*A(KxM) + b
    gemmDesc.write({
        (*this)["conv_out"].buffer(),
        (*this)["in0"].buffer(),
        (*this)["weight"].buffer(),
        (*this)["bias"].buffer(),
    });

    uint32_t pc_gemm[] = {
        Cout,         // N (output channels, outer)
        Cin_actual,   // K (input channels)
        M             // M (points, inner)
    };
    cmdBuff
        .bindPipeline(gemm)
        .bindDescSets({gemmDesc})
        .setPushConstants(0, sizeof(pc_gemm), pc_gemm)
        .dispatch0(CEIL_DIV(M, 32), CEIL_DIV(Cout, 32))  // x=M, y=N
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["conv_out"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );

    // Stage 2: BatchNorm1D
    batchnormDesc.write({
        (*this)["bn_out"].buffer(),
        (*this)["conv_out"].buffer(),
        (*this)["bn_mean"].buffer(),
        (*this)["bn_var"].buffer(),
        (*this)["bn_gamma"].buffer(),
        (*this)["bn_beta"].buffer()
    });

    float eps = 1e-5f;
    uint32_t pc_bn[] = { Cout, M };  // [C, N] layout
    cmdBuff
        .bindPipeline(batchnorm)
        .bindDescSets({batchnormDesc})
        .setPushConstants(0, sizeof(pc_bn), pc_bn)
        .setPushConstants(sizeof(pc_bn), sizeof(float), &eps)
        .dispatch0(CEIL_DIV(Cout * M, 256))
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["bn_out"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );

    // Stage 3: ReLU
    reluDesc.write({
        (*this)["out0"].buffer(),
        (*this)["bn_out"].buffer()
    });

    int total = Cout * M;
    cmdBuff
        .bindPipeline(relu)
        .setPushConstants(0, sizeof(int), &total)
        .bindDescSets({reluDesc})
        .dispatch(total)
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}


/////////////////////////////////////////////////////////////////////////////////////////
// PointWiseConvNode (Conv1x1 + BatchNorm1D, NO ReLU)
// Used for the last conv layer in PointNet encoder
/////////////////////////////////////////////////////////////////////////////////////////
PointWiseConvNode::PointWiseConvNode(uint32_t inDim, uint32_t outDim)
: Cin(inDim)
, Cout(outDim)
{
    setName("PointWiseConvNode");
    // Slots for Conv + BN (no ReLU)
    addSlot("in0", NodeSlot::input);      // [N, Cin]
    addSlot("out0", NodeSlot::output);    // [N, Cout]
    
    // Conv weights
    addSlot("weight", NodeSlot::input);   // [Cin, Cout]
    addSlot("bias", NodeSlot::input);     // [Cout]
    
    // BatchNorm parameters
    addSlot("bn_mean", NodeSlot::input);  // [Cout]
    addSlot("bn_var", NodeSlot::input);   // [Cout]
    addSlot("bn_gamma", NodeSlot::input); // [Cout]
    addSlot("bn_beta", NodeSlot::input);  // [Cout]
    
    // Intermediate tensor for Conv output
    addSlot("conv_out", NodeSlot::internal);  // After Conv

    // Create pipelines for Conv and BatchNorm only
    gemm = requestPipeline(src_gemm_shared);
    gemmDesc = gemm.descSetLayout(0).newDescSet(gDestSetPool);
    
    batchnorm = requestPipeline(src_batchnorm1d);
    batchnormDesc = batchnorm.descSetLayout(0).newDescSet(gDestSetPool);
}

void PointWiseConvNode::prepare()
{
    const auto& inShape = (*this)["in0"].shape();
    _ASSERT(inShape.size() == 2);
    _ASSERT(inShape[0] == Cin);  // [Cin, N] layout
    _ASSERT((*this)["weight"].validShape());
    _ASSERT((*this)["weight"].isShapeOf(Cout, Cin));  // PyTorch format: [Cout, Cin]
    _ASSERT((*this)["bias"].validShape());
    _ASSERT((*this)["bias"].isShapeOf(Cout));

    (*this)["out0"] = Tensor(Cout, inShape[1]);      // [Cout, N]
    (*this)["conv_out"] = Tensor(Cout, inShape[1]);  // [Cout, N]
    
    // Set default BatchNorm parameters if not provided
    if (!(*this)["bn_mean"].validShape()) {
        std::vector<float> zeros(Cout, 0.0f);
        (*this)["bn_mean"] = Tensor(Cout).set(zeros);
    }
    if (!(*this)["bn_var"].validShape()) {
        std::vector<float> ones(Cout, 1.0f);
        (*this)["bn_var"] = Tensor(Cout).set(ones);
    }
    if (!(*this)["bn_gamma"].validShape()) {
        std::vector<float> ones(Cout, 1.0f);
        (*this)["bn_gamma"] = Tensor(Cout).set(ones);
    }
    if (!(*this)["bn_beta"].validShape()) {
        std::vector<float> zeros(Cout, 0.0f);
        (*this)["bn_beta"] = Tensor(Cout).set(zeros);
    }
}

void PointWiseConvNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    uint32_t N = inShape[0];

    // Stage 1: Conv1x1 (GEMM)
    gemmDesc.write({
        (*this)["conv_out"].buffer(),
        (*this)["in0"].buffer(),
        (*this)["weight"].buffer(),
        (*this)["bias"].buffer(),
    });

    uint32_t pc_gemm[] = {
        N,        // M
        Cin,      // K
        Cout      // N
    };
    cmdBuff
        .bindPipeline(gemm)
        .bindDescSets({gemmDesc})
        .setPushConstants(0, sizeof(pc_gemm), pc_gemm)
        .dispatch0(CEIL_DIV(Cout, 32), CEIL_DIV(N, 32))
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["conv_out"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );

    // Stage 2: BatchNorm (output to out0 directly, no ReLU after)
    batchnormDesc.write({
        (*this)["out0"].buffer(),       // Output directly to out0
        (*this)["conv_out"].buffer(),
        (*this)["bn_mean"].buffer(),
        (*this)["bn_var"].buffer(),
        (*this)["bn_gamma"].buffer(),
        (*this)["bn_beta"].buffer()
    });

    uint32_t pc_bn[] = { N, Cout };
    cmdBuff
        .bindPipeline(batchnorm)
        .bindDescSets({batchnormDesc})
        .setPushConstants(0, sizeof(pc_bn), pc_bn)
        .dispatch0(CEIL_DIV(N * Cout, 256))  // FIX: dispatch for all N*C elements
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
    
    // No ReLU stage - output is directly from BatchNorm
}  


/////////////////////////////////////////////////////////////////////////////////////////
// ReluNode
/////////////////////////////////////////////////////////////////////////////////////////
ReluNode::ReluNode()
{
    setName("ReluNode");
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


/////////////////////////////////////////////////////////////////////////////////////////
// BatchNorm1DNode
/////////////////////////////////////////////////////////////////////////////////////////
BatchNorm1DNode::BatchNorm1DNode(uint32_t channels)
: C(channels)
{
    setName("BatchNorm1DNode");
    addSlot("in0", NodeSlot::input);        // [N, C]
    addSlot("out0", NodeSlot::output);      // [N, C]
    addSlot("mean", NodeSlot::input);       // [C] running_mean
    addSlot("var", NodeSlot::input);        // [C] running_var
    addSlot("gamma", NodeSlot::input);      // [C] weight (scale)
    addSlot("beta", NodeSlot::input);       // [C] bias (shift)

    batchnorm = requestPipeline(src_batchnorm1d);
    batchnormDesc = batchnorm.descSetLayout(0).newDescSet(gDestSetPool);
}

void BatchNorm1DNode::prepare()
{
    const auto& inShape = (*this)["in0"].shape();
    _ASSERT(inShape.size() == 2);
    _ASSERT(inShape[0] == C);  // [C, N] layout - channels is outer dimension
    _ASSERT((*this)["mean"].validShape());
    _ASSERT((*this)["mean"].isShapeOf(C));
    _ASSERT((*this)["var"].validShape());
    _ASSERT((*this)["var"].isShapeOf(C));
    _ASSERT((*this)["gamma"].validShape());
    _ASSERT((*this)["gamma"].isShapeOf(C));
    _ASSERT((*this)["beta"].validShape());
    _ASSERT((*this)["beta"].isShapeOf(C));

    (*this)["out0"] = Tensor(inShape);
}

void BatchNorm1DNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    uint32_t C_actual = inShape[0];  // channels (outer)
    uint32_t N = inShape[1];         // points (inner)
    uint32_t total = C_actual * N;
    
    batchnormDesc.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
        (*this)["mean"].buffer(),
        (*this)["var"].buffer(),
        (*this)["gamma"].buffer(),
        (*this)["beta"].buffer(),
    });
    
    struct { uint32_t C, N; float eps; } pc = {C_actual, N, 1e-5f};
    
    cmdBuff
        .bindPipeline(batchnorm)
        .bindDescSets({batchnormDesc})
        .setPushConstants(0, sizeof(pc), &pc)
        .dispatch(total)  // dispatch(numThreads) auto-calculates work groups
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}


/////////////////////////////////////////////////////////////////////////////////////////
// MaxPooling1DNode
/////////////////////////////////////////////////////////////////////////////////////////
MaxPooling1DNode::MaxPooling1DNode()
{
    setName("MaxPooling1DNode");
    addSlot("in0", NodeSlot::input);   // [N, C]
    addSlot("out0", NodeSlot::output); // [C]

    maxpool = requestPipeline(src_maxpool1d);
    desc = maxpool.descSetLayout(0).newDescSet(gDestSetPool);}

void MaxPooling1DNode::prepare()
{
    const auto& inShape = (*this)["in0"].shape();
    _ASSERT(inShape.size() == 2);
    uint32_t C = inShape[0];  // [C, N] layout - channels is outer dimension
    (*this)["out0"] = Tensor(C);
}

void MaxPooling1DNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    uint32_t C = inShape[0];  // channels (outer)
    uint32_t N = inShape[1];  // points (inner)

    desc.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
    });

    uint32_t pc[] = {C, N};

    cmdBuff
        .bindPipeline(maxpool)
        .bindDescSets({desc})
        .setPushConstants(0, sizeof(pc), pc)
        .dispatch0(CEIL_DIV(C, 32))
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
    setName("MaxPoolingNode");
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



/////////////////////////////////////////////////////////////////////////////////////////
// FlattenNode
/////////////////////////////////////////////////////////////////////////////////////////
FlattenNode::FlattenNode()
{
    setName("FlattenNode");
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


/////////////////////////////////////////////////////////////////////////////////////////
// FullyConnectedNode
/////////////////////////////////////////////////////////////////////////////////////////
FullyConnectedNode::FullyConnectedNode(uint32_t inDim, uint32_t outDim)
: I(inDim), O(outDim) 
{
    setName("FullyConnectedNode");
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
    addSlot("weight", NodeSlot::input);
    addSlot("bias", NodeSlot::input);

    const char* gemmSrc = src_gemm_naive;
    //gemmTileSize = gGemmTileSize.at(gemmSrc);

    //const char* gemmSrc = src_gemm_kSplit;

    gemm = requestPipeline(gemmSrc);
    gemmDescSet = gemm.descSetLayout(0).newDescSet(gDestSetPool);

	setZero = requestPipeline(src_setZero);
    setZeroDescSet = setZero.descSetLayout(0).newDescSet(gDestSetPool);
}

void FullyConnectedNode::prepare()
{
    _ASSERT((*this)["in0"].isShapeOf(I));
    _ASSERT((*this)["weight"].validShape());
    _ASSERT((*this)["weight"].isShapeOf(I, O));
    _ASSERT((*this)["bias"].validShape());
    _ASSERT((*this)["bias"].isShapeOf(O));
    (*this)["out0"] = Tensor(O); 
}

void FullyConnectedNode::run(CommandBuffer cmdBuff) 
{
    // FC on 1D vector: out[O] = weight[O, I] @ in[I] + bias[O]
    // Using GEMM: C(NxM) = B(NxK)*A(KxM) + b
    uint32_t N = (*this)["out0"].shape()[0];  // output size
    uint32_t K = (*this)["in0"].shape()[0];   // input size
    uint32_t M = 1;  // single point
    
    gemmDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
        (*this)["weight"].buffer(),
        (*this)["bias"].buffer(),
    });

    uint32_t gemmConstants[] = {N, K, M};

    cmdBuff
        .bindPipeline(gemm)
        .bindDescSets({gemmDescSet})
        .setPushConstants(0, sizeof(gemmConstants), gemmConstants)
        .dispatch0(CEIL_DIV(M, 32), CEIL_DIV(N, 32))  // x=M, y=N
        .barrier( 
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );

}


//==============================================================================
// BroadcastNode Implementation
//==============================================================================

BroadcastNode::BroadcastNode()
{
    setName("BroadcastNode");
    addSlot("in0", NodeSlot::input);   // [1, C]
    addSlot("in1", NodeSlot::input);   // [N, C] (for shape reference)
    addSlot("out0", NodeSlot::output); // [N, C]
    
    broadcast = requestPipeline(src_broadcast);
    broadcastDescSet = broadcast.descSetLayout(0).newDescSet(gDestSetPool);
}

void BroadcastNode::prepare()
{
    Tensor& in0 = (*this)["in0"];  // global feature [C, 1]
    Tensor& in1 = (*this)["in1"];  // point features [C, N] for shape reference
    
    _ASSERT(in0.validShape() && in1.validShape());
    _ASSERT(in0.shape().size() == 2);
    _ASSERT(in1.shape().size() == 2);
    _ASSERT(in0.shape()[1] == 1);  // global feature should have N=1
    _ASSERT(in0.shape()[0] == in1.shape()[0]); // same channel dimension
    
    // Output shape: [C, N]
    (*this)["out0"] = Tensor(in0.shape()[0], in1.shape()[1]);
}

void BroadcastNode::run(CommandBuffer cmdBuff)
{
    Tensor& in0 = (*this)["in0"];   // [C, 1]
    Tensor& in1 = (*this)["in1"];   // [C, N]
    Tensor& out0 = (*this)["out0"]; // [C, N]
    
    uint32_t C = in0.shape()[0];  // channels (outer)
    uint32_t N = in1.shape()[1];  // points (inner)
    
    broadcastDescSet.write({
        in0.buffer(),
        out0.buffer()
    });
    
    uint32_t pc[] = {C, N};
    
    cmdBuff
        .bindPipeline(broadcast)
        .bindDescSets({broadcastDescSet})
        .setPushConstants(0, sizeof(pc), pc)
        .dispatch0((N * C + 255) / 256, 1, 1)
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / out0.buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}


//==============================================================================
// ConcatNode Implementation
//==============================================================================

ConcatNode::ConcatNode()
{
    setName("ConcatNode");
    addSlot("in0", NodeSlot::input);   // [N, C1]
    addSlot("in1", NodeSlot::input);   // [N, C2]
    addSlot("out0", NodeSlot::output); // [N, C1+C2]
    
    concat = requestPipeline(src_concat);
    concatDescSet = concat.descSetLayout(0).newDescSet(gDestSetPool);
}

void ConcatNode::prepare()
{
    Tensor& in0 = (*this)["in0"];
    Tensor& in1 = (*this)["in1"];
    
    _ASSERT(in0.validShape() && in1.validShape());
    _ASSERT(in0.shape().size() == 2);
    _ASSERT(in1.shape().size() == 2);
    _ASSERT(in0.shape()[1] == in1.shape()[1]); // same N
    
    uint32_t C1 = in0.shape()[0];  // channels in first input (outer)
    uint32_t C2 = in1.shape()[0];  // channels in second input
    uint32_t N = in0.shape()[1];   // points (inner)
    
    // Output shape: [C1+C2, N]
    (*this)["out0"] = Tensor(C1 + C2, N);
}

void ConcatNode::run(CommandBuffer cmdBuff)
{
    Tensor& in0 = (*this)["in0"];   // [C1, N]
    Tensor& in1 = (*this)["in1"];   // [C2, N]
    Tensor& out0 = (*this)["out0"]; // [C1+C2, N]
    
    uint32_t C1 = in0.shape()[0];  // channels (outer)
    uint32_t C2 = in1.shape()[0];
    uint32_t N = in0.shape()[1];   // points (inner)
    
    concatDescSet.write({
        in0.buffer(),
        in1.buffer(),
        out0.buffer()
    });
    
    uint32_t pc[] = {C1, C2, N};
    
    cmdBuff
        .bindPipeline(concat)
        .bindDescSets({concatDescSet})
        .setPushConstants(0, sizeof(pc), pc)
        .dispatch0((N * (C1 + C2) + 255) / 256, 1, 1)
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / out0.buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}

// ============================================================================
// SliceNode - Extract channel slice from tensor
// ============================================================================

SliceNode::SliceNode(uint32_t start, uint32_t end)
: start_channel(start), end_channel(end)
{
    setName("SliceNode");
    addSlot("in0", NodeSlot::input);   // [N, C_in]
    addSlot("out0", NodeSlot::output); // [N, slice_size]
    
    slice = requestPipeline(src_slice);
    sliceDescSet = slice.descSetLayout(0).newDescSet(gDestSetPool);
}

void SliceNode::prepare()
{
    Tensor& in0 = (*this)["in0"];
    
    _ASSERT(in0.validShape());
    _ASSERT(in0.shape().size() == 2);
    
    uint32_t C_in = in0.shape()[0];  // input channels (outer)
    uint32_t N = in0.shape()[1];     // points (inner)
    
    _ASSERT(start_channel < end_channel);
    _ASSERT(end_channel <= C_in);
    
    uint32_t slice_size = end_channel - start_channel;
    // Output shape: [slice_size, N]
    (*this)["out0"] = Tensor(slice_size, N);
}

void SliceNode::run(CommandBuffer cmdBuff)
{
    Tensor& in0 = (*this)["in0"];   // [C_in, N]
    Tensor& out0 = (*this)["out0"]; // [slice_size, N]
    
    uint32_t C_in = in0.shape()[0];  // input channels (outer)
    uint32_t N = in0.shape()[1];     // points (inner)
    uint32_t slice_size = end_channel - start_channel;
    
    sliceDescSet.write({
        in0.buffer(),
        out0.buffer()
    });
    
    uint32_t pc[] = {C_in, N, start_channel, slice_size};
    
    cmdBuff
        .bindPipeline(slice)
        .bindDescSets({sliceDescSet})
        .setPushConstants(0, sizeof(pc), pc)
        .dispatch0((slice_size * N + 255) / 256, 1, 1)
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / out0.buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}

ReShapeNode::ReShapeNode()
{
    setName("ReShapeNode");
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
}

ReShapeNode::ReShapeNode(std::vector<uint32_t> shape)
: targetShape(shape)
{
    setName("ReShapeNode");
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
}

void ReShapeNode::setTargetShape(std::vector<uint32_t> shape)
{
    targetShape = shape;
}

void ReShapeNode::prepare()
{
    _ASSERT((*this)["in0"].validShape());
    Tensor& outTensor = (*this)["out0"] = (*this)["in0"];
    
    // If target shape is specified, reshape to it
    if (!targetShape.empty()) {
        // Verify total elements match
        uint32_t totalElements = 1;
        for (auto dim : targetShape) {
            totalElements *= dim;
        }
        _ASSERT(outTensor.numElements() == totalElements);
        
        // Apply reshape
        if (targetShape.size() == 1) {
            outTensor.reshape(targetShape[0]);
        } else if (targetShape.size() == 2) {
            outTensor.reshape(targetShape[0], targetShape[1]);
        } else if (targetShape.size() == 3) {
            outTensor.reshape(targetShape[0], targetShape[1], targetShape[2]);
        } else if (targetShape.size() == 4) {
            outTensor.reshape(targetShape[0], targetShape[1], targetShape[2], targetShape[3]);
        }
    }
}

void ReShapeNode::run(CommandBuffer cmdBuff) 
{
    // No-op: reshape is just a metadata change, no GPU work needed
}


//==============================================================================
// MatMulNode Implementation
//==============================================================================

MatMulNode::MatMulNode()
{
    setName("MatMulNode");
    addSlot("in0", NodeSlot::input);   // [N, K]
    addSlot("in1", NodeSlot::input);   // [K, M]
    addSlot("out0", NodeSlot::output); // [N, M]
    
    matmul = requestPipeline(src_matmul);
    matmulDescSet = matmul.descSetLayout(0).newDescSet(gDestSetPool);
}

void MatMulNode::prepare()
{
    Tensor& in0 = (*this)["in0"];  // [K, N] - point features
    Tensor& in1 = (*this)["in1"];  // [M, K] - transformation matrix
    
    _ASSERT(in0.validShape() && in1.validShape());
    _ASSERT(in0.shape().size() == 2);
    _ASSERT(in1.shape().size() == 2);
    _ASSERT(in0.shape()[0] == in1.shape()[1]); // K must match
    
    uint32_t K = in0.shape()[0];   // intermediate dimension
    uint32_t N = in0.shape()[1];   // points (inner)
    uint32_t M = in1.shape()[0];   // output dimension (outer)
    
    // Output shape: [M, N]
    (*this)["out0"] = Tensor(M, N);
}

void MatMulNode::run(CommandBuffer cmdBuff)
{
    Tensor& in0 = (*this)["in0"];   // [K, N]
    Tensor& in1 = (*this)["in1"];   // [M, K]
    Tensor& out0 = (*this)["out0"]; // [M, N]
    
    uint32_t K = in0.shape()[0];  // intermediate dimension
    uint32_t N = in0.shape()[1];  // points (inner)
    uint32_t M = in1.shape()[0];  // output dimension (outer)
    
    matmulDescSet.write({
        out0.buffer(),
        in0.buffer(),
        in1.buffer()
    });
    
    uint32_t pc[] = {M, K, N};
    
    cmdBuff
        .bindPipeline(matmul)
        .bindDescSets({matmulDescSet})
        .setPushConstants(0, sizeof(pc), pc)
        .dispatch0((N + 15) / 16, (M + 15) / 16, 1)  // x=N, y=M
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / out0.buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}

//==============================================================================
// AddIdentityNode Implementation
//==============================================================================

static const char* src_add_identity = R"(
#version 450
layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer OutBuffer { float out_data[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in_data[]; };

layout(push_constant) uniform PushConstants {
    uint K;  // Matrix dimension (K x K)
};

void main() 
{
    uint idx = gl_GlobalInvocationID.x;
    uint total = K * K;
    if (idx >= total) return;
    
    uint row = idx / K;
    uint col = idx % K;
    
    // Add identity matrix: out[i][j] = in[i][j] + (i == j ? 1.0 : 0.0)
    float identity = (row == col) ? 1.0 : 0.0;
    out_data[idx] = in_data[idx] + identity;
}
)";

AddIdentityNode::AddIdentityNode()
{
    setName("AddIdentityNode");
    addSlot("in0", NodeSlot::input);   // [K, K]
    addSlot("out0", NodeSlot::output); // [K, K]
    
    addIdentity = requestPipeline(src_add_identity);
    addIdentityDescSet = addIdentity.descSetLayout(0).newDescSet(gDestSetPool);
}

void AddIdentityNode::prepare()
{
    Tensor& in0 = (*this)["in0"];
    
    _ASSERT(in0.validShape());
    _ASSERT(in0.shape().size() == 2);
    _ASSERT(in0.shape()[0] == in0.shape()[1]); // Must be square matrix
    
    K = in0.shape()[0];
    
    // Output shape: same as input [K, K]
    (*this)["out0"] = Tensor(K, K);
}

void AddIdentityNode::run(CommandBuffer cmdBuff)
{
    Tensor& in0 = (*this)["in0"];
    Tensor& out0 = (*this)["out0"];
    
    addIdentityDescSet.write({
        out0.buffer(),
        in0.buffer()
    });
    
    uint32_t pc[] = {K};
    
    cmdBuff
        .bindPipeline(addIdentity)
        .bindDescSets({addIdentityDescSet})
        .setPushConstants(0, sizeof(pc), pc)
        .dispatch0(CEIL_DIV(K * K, 256), 1, 1)
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / out0.buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}

// IdentityNode - Pass-through node for signal splitting
// Allows one input to fan out to multiple outputs without computation
IdentityNode::IdentityNode()
{
    setName("IdentityNode");
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);  // First output
    addSlot("out1", NodeSlot::output);  // Second output for splitting
}

void IdentityNode::prepare()
{
    Tensor& in0 = (*this)["in0"];
    
    _ASSERT(in0.validShape());
    
    // Both outputs are identical copies (same shape and data)
    (*this)["out0"] = in0;  // Reference to input (no copy)
    (*this)["out1"] = in0;  // Reference to input (no copy)
}

void IdentityNode::run(CommandBuffer cmdBuff)
{
    // No computation needed - outputs reference the same buffer as input
    // This is a logical node for graph connectivity only
}

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

static const char* src_lrn = R"(
#version 450
// Local Response Normalization across channels (GoogLeNet style)
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in0[]; };
layout(push_constant) uniform PushConstants {
    int C;
    int H;
    int W;
    int size;    // window size across channels (e.g., 5)
    float alpha; // usually 1e-4
    float beta;  // usually 0.75
    float k;     // usually 1.0
};

void main()
{
    int idx = int(gl_GlobalInvocationID.x);
    int total = C * H * W;
    if (idx >= total) return;

    int c = idx / (H * W);
    int hw = idx - c * H * W;
    int h = hw / W;
    int w = hw - h * W;

    int halfWindow = size / 2;
    float sumSq = 0.0;
    for (int cc = max(0, c - halfWindow); cc <= min(C - 1, c + halfWindow); ++cc)
    {
        float v = in0[(cc * H + h) * W + w];
        sumSq += v * v;
    }

    float norm = pow(k + alpha * sumSq, beta);
    out0[idx] = in0[idx] / norm;
}
)";

static const char* src_maxpool = R"(
#version 450
#define FLT_MIN -3.402823466e+38
layout(local_size_x = 64, local_size_y = 4, local_size_z = 4) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in0[]; };
layout(push_constant) uniform PushConstants {
    int C;
    int H;      // input height
    int W;      // input width
    int P;      // pooling window
    int S;      // stride
    int pad;    // symmetric padding
};

void main()
{
    int h_ = int(gl_GlobalInvocationID.x);  // output row
    int w_ = int(gl_GlobalInvocationID.y);  // output col
    int c = int(gl_GlobalInvocationID.z);   // channel
    int H_ = (H + 2 * pad - P + S - 1) / S + 1;
    int W_ = (W + 2 * pad - P + S - 1) / S + 1;
    if (h_ >= H_ || w_ >= W_ || c >= C)
        return;

    int h0 = h_ * S - pad;  
    int w0 = w_ * S - pad;     
    float maxVal = FLT_MIN;
    for (int dh=0; dh < P; ++dh) 
    {
        int h = h0 + dh;  
        for (int dw=0; dw < P; ++dw) 
        {
            int w = w0 + dw;

            if (0 <= h && h < H && 0 <= w && w < W) 
            {
                maxVal = max(maxVal, in0[(c * H + h) * W + w]);
            }
        }
    }
    out0[(c * H_ + h_) * W_ + w_] = maxVal;
})";

static const char* src_im2col = R"(
#version 450
layout(local_size_x = 64, local_size_y = 16) in;
layout(set = 0, binding = 0) writeonly buffer OutBuffer { float im2colOut[]; };
layout(set = 0, binding = 1) readonly buffer InBuffer { float in0[]; };
layout(push_constant) uniform PushConstants {
    int C;
    int H;
    int W;
    int K;
    int S;
    int pad;
};

void main() 
{
    int i = int(gl_GlobalInvocationID.x); 
    int j = int(gl_GlobalInvocationID.y); 
    int KK = K * K;
    int CKK = C * KK;
    int H_ = (H + 2 * pad - K) / S + 1;
    int W_ = (W + 2 * pad - K) / S + 1;
    if (i >= H_ * W_ || j >= CKK) 
        return;

    int h_ = i / W_;
    int w_ = i % W_;
    int c = j / KK;
    int k = j % KK;

    float value = 0.0;
    int h = h_ * S + k / K - pad;  
    int w = w_ * S + k % K - pad;   
    if (0 <= h && h < H && 0 <= w && w < W) 
        value = in0[(c * H + h) * W + w];

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

// Variant for convolution: write output as channel-major (c,h,w) => index = n * M + m,
// where M = H_out * W_out, n = out channel.
static const char* src_gemm_shared_conv = R"(
#version 450
layout(local_size_x = 32, local_size_y = 32) in;
layout(set = 0, binding = 0) buffer OutBuffer { float C[]; };
layout(set = 0, binding = 1) buffer InBuffer { float A[]; };
layout(set = 0, binding = 2) buffer Weight { float B[]; };
layout(set = 0, binding = 3) buffer Bias { float b[]; };

// C(NxM) in channel-major order (n = out channel, m = spatial idx)
layout(push_constant) uniform PushConstants {
    int M;  // H_out * W_out
    int K;  // input size (C * K * K)
    int N;  // output channels
    int useRelu; // 0: none, 1: relu
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
    {
        float val = acc + b[n];
        if (useRelu == 1) val = max(0.0, val);
        C[n * M + m] = val;
    }
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


static const char* src_gemm_conv_optimized = R"(
#version 450
#define BM 64
#define BN 64
#define BK 8
#define TM 4
#define TN 4

layout(local_size_x = 256) in; // (BM*BN)/(TM*TN) = 4096 / 16 = 256

layout(set = 0, binding = 0) buffer OutBuffer { float C[]; };
layout(set = 0, binding = 1) buffer InBuffer { float A[]; };
layout(set = 0, binding = 2) buffer Weight   { float B[]; };
layout(set = 0, binding = 3) buffer Bias     { float b[]; };

// C(NxM) in channel-major order (n = out channel, m = spatial idx)
// A(MxK), B(KxN)
layout(push_constant) uniform PushConstants {
    int M;  // H_out * W_out
    int K;  // input size (C * K * K)
    int N;  // output channels
    int useRelu; // 0: none, 1: relu
};

shared float As[BM * BK]; // 64*8
shared float Bs[BK * BN]; // 8*64

void main() 
{
    int tileCol = int(gl_WorkGroupID.x); // Tile index along N (Output Channels)
    int tileRow = int(gl_WorkGroupID.y); // Tile index along M (Spatial)
    int t = int(gl_LocalInvocationID.x); // Thread index (0..255)

    int rowStrideA = 256 / BK; // 32
    int rowStrideB = 256 / BN; // 4

    int innerRowA = t / BK; // 0..31
    int innerColA = t % BK; // 0..7
    
    int innerRowB = t / BN; // 0..3
    int innerColB = t % BN; // 0..63

    // Thread results (TMxTN = 4x4 = 16 registers)
    float threadResults[TM * TN];
    for (int i = 0; i < TM*TN; ++i) threadResults[i] = 0.0;

    float regM[TM];
    float regN[TN];

    // Global indices for the top-left of the tile
    int m0 = tileRow * BM;
    int n0 = tileCol * BN;

    // Thread's computation tile within the workgroup tile
    // t ranges 0..255. 
    // t_m = t / (BN/TN) = t / (64/4) = t / 16. (0..15)
    // t_n = t % 16. (0..15)
    int t_m = t / 16;
    int t_n = t % 16;

    for (int k0 = 0; k0 < K; k0 += BK) 
    {
        // Load As (BMxBK = 64x8 = 512 floats)
        // Threads: 256. Loads per thread: 2.
        for (int i = 0; i < 2; ++i)
        {
            int r = innerRowA + i * 32; // 0..63
            int c = innerColA;          // 0..7
            int globalM = m0 + r;
            int globalK = k0 + c;
            
            As[c * BM + r] = (globalM < M && globalK < K) ? A[globalM * K + globalK] : 0.0;
        }

        // Load Bs (BKxBN = 8x64 = 512 floats)
        // Threads: 256. Loads per thread: 2.
        for (int i = 0; i < 2; ++i)
        {
            int r = innerRowB + i * 4; // 0..7
            int c = innerColB;         // 0..63
            int globalK = k0 + r;
            int globalN = n0 + c;

            Bs[r * BN + c] = (globalK < K && globalN < N) ? B[globalK * N + globalN] : 0.0;
        }

        barrier();

        // Compute
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
            // Load A values for this dotIdx into registers
            for (int i = 0; i < TM; ++i)
            {
                regM[i] = As[dotIdx * BM + (t_m * TM + i)];
            }

            // Load B values for this dotIdx into registers
            for (int i = 0; i < TN; ++i)
            {
                regN[i] = Bs[dotIdx * BN + (t_n * TN + i)];
            }

            // Outer product
            for (int i = 0; i < TM; ++i)
            {
                for (int j = 0; j < TN; ++j)
                {
                    threadResults[i * TN + j] += regM[i] * regN[j];
                }
            }
        }

        barrier();
    }

    // Write results
    int globalM_base = m0 + t_m * TM;
    int globalN_base = n0 + t_n * TN;

    for (int j = 0; j < TN; ++j)
    {
        int n = globalN_base + j;
        if (n >= N) continue;
        
        float biasVal = b[n];

        for (int i = 0; i < TM; ++i)
        {
            int m = globalM_base + i;
            if (m >= M) continue;

            float val = threadResults[i * TN + j] + biasVal;
            if (useRelu == 1) val = max(0.0, val);

            C[n * M + m] = val;
        }
    }
})";


static const char* src_avgpool = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in0[]; };
layout(push_constant) uniform PushConstants {
    int C;
    int H;
    int W;
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
            sum += in0[(c * H + h) * W + w];
        }
    }
    out0[c] = sum / float(H * W);
})";

Device netGlobalDevice = VulkanApp::get().device();

static DescriptorPool gDestSetPool = netGlobalDevice.createDescriptorPool({
    .maxTypes = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER <= 2000000}, 
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
    {src_gemm_shared_conv, 32},
    {src_gemm_conv_optimized, 64}, // Tuned to 64
    {src_gemm_multiOut1d, 64},
    {src_gemm_multiOut2d, 128}
};

void loadShaders()
{
    requestPipeline(src_relu);
    // requestPipeline(src_copy);
    requestPipeline(src_setZero);
    requestPipeline(src_lrn);
    requestPipeline(src_maxpool);
    requestPipeline(src_im2col);
    requestPipeline(src_gemm_naive);
    requestPipeline(src_gemm_shared);      // used by Conv/FC
    requestPipeline(src_gemm_conv_optimized); // New optimized shader
    requestPipeline(src_gemm_multiOut1d);  // optional alternative
    requestPipeline(src_gemm_multiOut2d);  // optional alternative

}

/////////////////////////////////////////////////////////////////////////////////////////
// ConvolutionNode
/////////////////////////////////////////////////////////////////////////////////////////
ConvolutionNode::ConvolutionNode(uint32_t inChannels, uint32_t outChannels, uint32_t kernelWidth, uint32_t stride, uint32_t padding, bool useRelu)
:  C(inChannels), F(outChannels), K(kernelWidth), S(stride), padding(padding), useRelu(useRelu)
{
    _ASSERT(K % 2 == 1);
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
    addSlot("im2colOut", NodeSlot::internal);
    addSlot("weight", NodeSlot::input);
    addSlot("bias", NodeSlot::input);

    im2col = requestPipeline(src_im2col);
    im2colDescSet = im2col.descSetLayout(0).newDescSet(gDestSetPool);
    
    const char* gemmSrc = src_gemm_conv_optimized;

    gemm = requestPipeline(gemmSrc);
    gemmTileSize = gGemmTileSize.at(gemmSrc);
    gemmDescSet = gemm.descSetLayout(0).newDescSet(gDestSetPool);
}

void ConvolutionNode::prepare()
{
    _ASSERT((*this)["in0"].isShapeOf(C, -1, -1));
    _ASSERT((*this)["weight"].isShapeOf(C*K*K, F));
    _ASSERT((*this)["bias"].isShapeOf(F));

    const auto& inShape = (*this)["in0"].shape();
    uint32_t H = inShape[1], W = inShape[2];
    uint32_t H_ = (H + 2 * padding - K) / S + 1;
    uint32_t W_ = (W + 2 * padding - K) / S + 1;
    (*this)["im2colOut"] = Tensor(H_ * W_, C * K * K);
    (*this)["out0"] = Tensor(F, H_, W_);
}

void ConvolutionNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    uint32_t H = inShape[1], W = inShape[2];
    uint32_t H_ = (H + 2 * padding - K) / S + 1;
    uint32_t W_ = (W + 2 * padding - K) / S + 1;

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

    uint32_t im2colConstants[] = {C, H, W, K, S, padding};

    uint32_t M = H_ * W_;       // N
    uint32_t K_ = C * K * K;    // I
    uint32_t N = F;             // O
    uint32_t reluFlag = useRelu ? 1 : 0;
    uint32_t gemmConstants[] = {M, K_, N, reluFlag};

    cmdBuff
        .bindPipeline(im2col)
        .bindDescSets({im2colDescSet})
        .setPushConstants(0, sizeof(im2colConstants), im2colConstants)
        .bindPipeline(im2col)
        .bindDescSets({im2colDescSet})
        .setPushConstants(0, sizeof(im2colConstants), im2colConstants)
        .dispatch(H_ * W_, C * K * K)
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
// ReluNode
/////////////////////////////////////////////////////////////////////////////////////////
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


/////////////////////////////////////////////////////////////////////////////////////////
// LRNNode
/////////////////////////////////////////////////////////////////////////////////////////
LRNNode::LRNNode()
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);

    lrn = requestPipeline(src_lrn);
    lrnDescSet = lrn.descSetLayout(0).newDescSet(gDestSetPool);
}

void LRNNode::prepare()
{
    _ASSERT((*this)["in0"].validShape());
    (*this)["out0"] = Tensor((*this)["in0"].shape());
}

void LRNNode::run(CommandBuffer cmdBuff)
{
    // Bypass LRN for testing: just copy in0 to out0
    // If accuracy improves, it means the weights were trained without LRN (e.g. TorchVision)
    
    // We can use the copy pipeline or just dispatch a simple copy shader.
    // Or simpler: reuse the LRN shader but set alpha=0, beta=1, k=1?
    // norm = pow(k + alpha*sum, beta) = pow(1 + 0, 1) = 1.
    // out = in / 1 = in.
    
    const auto& inShape = (*this)["in0"].shape();
    uint32_t C = inShape[0], H = inShape[1], W = inShape[2];

    lrnDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
    });

    struct { int C, H, W, size; float alpha, beta, k; } push{};
    push.C = (int)C; push.H = (int)H; push.W = (int)W;
    push.size = 5;
    push.alpha = 0.0001f / 5.0f;
    push.beta = 0.75f;
    push.k = 2.0f;

    cmdBuff
        .bindPipeline(lrn)
        .bindDescSets({lrnDescSet})
        .setPushConstants(0, sizeof(push), &push)
        .dispatch(C * H * W)
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}


/////////////////////////////////////////////////////////////////////////////////////////
// MaxPoolingNode
/////////////////////////////////////////////////////////////////////////////////////////
MaxPoolingNode::MaxPoolingNode(uint32_t poolSize, uint32_t stride, uint32_t padding)
: P(poolSize)
, S(stride ? stride : poolSize)
, padding(padding)
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
    uint32_t C = inShape[0], H = inShape[1], W = inShape[2];

    uint32_t H_ = (H + 2 * padding - P + S - 1) / S + 1; // ceil_mode
    uint32_t W_ = (W + 2 * padding - P + S - 1) / S + 1;

    (*this)["out0"] = Tensor(C, H_, W_);
}

void MaxPoolingNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    uint32_t C = inShape[0], H = inShape[1], W = inShape[2];
    uint32_t H_ = (H + 2 * padding - P + S - 1) / S + 1; // ceil_mode
    uint32_t W_ = (W + 2 * padding - P + S - 1) / S + 1;

    maxpoolDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
    });

    uint32_t maxpoolConstants[] = {C, H, W, P, S, padding};

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
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
    addSlot("weight", NodeSlot::input);
    addSlot("bias", NodeSlot::input);

    // Use shared-memory GEMM to avoid atomic float extension requirements
    const char* gemmSrc = src_gemm_shared;
    gemm = requestPipeline(gemmSrc);
    gemmTileSize = gGemmTileSize.at(gemmSrc);
    gemmDescSet = gemm.descSetLayout(0).newDescSet(gDestSetPool);

    setZero = requestPipeline(src_setZero);
    setZeroDescSet = setZero.descSetLayout(0).newDescSet(gDestSetPool);
}

void FullyConnectedNode::prepare() 
{
    _ASSERT((*this)["in0"].isShapeOf(I));
    _ASSERT((*this)["weight"].isShapeOf(I, O));
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
        .dispatch0(CEIL_DIV(N, gemmTileSize), CEIL_DIV(M, gemmTileSize))
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

    avgpool = requestPipeline(src_avgpool);
    avgpoolDescSet = avgpool.descSetLayout(0).newDescSet(gDestSetPool);
}

void GlobalAvgPoolNode::prepare()
{
    const auto& inShape = (*this)["in0"].shape();
    _ASSERT(inShape.size() == 3);
    uint32_t C = inShape[0];
    (*this)["out0"] = Tensor(C, 1, 1);
}

void GlobalAvgPoolNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    uint32_t C = inShape[0], H = inShape[1], W = inShape[2];

    avgpoolDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
    });

    uint32_t pushConstants[] = {C, H, W};

    cmdBuff
        .bindPipeline(avgpool)
        .bindDescSets({avgpoolDescSet})
        .setPushConstants(0, sizeof(pushConstants), pushConstants)
        .dispatch(C)
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}

/////////////////////////////////////////////////////////////////////////////////////////
// IdentityNode
/////////////////////////////////////////////////////////////////////////////////////////
IdentityNode::IdentityNode()
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
}

void IdentityNode::prepare()
{
    _ASSERT((*this)["in0"].validShape());
    (*this)["out0"] = (*this)["in0"];
}

void IdentityNode::run(CommandBuffer cmdBuff)
{
}

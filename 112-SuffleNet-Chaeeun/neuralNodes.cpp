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
layout(set = 0, binding = 1) readonly buffer InBuffer { float in0[]; };
layout(push_constant) uniform PushConstants {
    int H;
    int W;
    int C;
    int K;
    int stride;
    int pad;
    int outH;
    int outW;
};

void main() 
{
    int i = int(gl_GlobalInvocationID.x); 
    int j = int(gl_GlobalInvocationID.y); 
    int KK = K * K;
    int CKK = C * KK;
    int HW_out = outH * outW;
    if (i >= HW_out || j >= CKK) 
        return;

    int h_ = i / outW;      // output row
    int w_ = i % outW;      // output col
    int c = j / KK;         // channel
    int k = j % KK;         // kernel index
    int kh = k / K;
    int kw = k % K;

    float value = 0.0;
    int h = h_ * stride + kh - pad;
    int w = w_ * stride + kw - pad;
    if (0 <= h && h < H && 0 <= w && w < W) 
        value = in0[(h * W + w) * C + c];

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

static const char* src_adaptiveavgpool = R"(
    #version 450
    layout(local_size_x = 64, local_size_y = 4, local_size_z = 4) in;

    layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
    layout(set = 0, binding = 1) buffer InBuffer  { float in0[]; };

    layout(push_constant) uniform PushConstants {
        int H;      // input height
        int W;      // input width
        int C;      // channels
        int outH;   // output height (adaptive target)
        int outW;   // output width  (adaptive target)
    };

    void main()
    {
        int h_ = int(gl_GlobalInvocationID.x);  // output row index (0 .. outH-1)
        int w_ = int(gl_GlobalInvocationID.y);  // output col index (0 .. outW-1)
        int c  = int(gl_GlobalInvocationID.z);  // channel index   (0 .. C-1)

        if (h_ >= outH || w_ >= outW || c >= C)
            return;

        // PyTorch adaptive_avg_pool2d 방식:
        // h_start = floor( h_      * H / outH )
        // h_end   = ceil( (h_ + 1) * H / outH )
        // w_start = floor( w_      * W / outW )
        // w_end   = ceil( (w_ + 1) * W / outW )

        int h_start = (h_ * H) / outH;
        int h_end   = ((h_ + 1) * H + outH - 1) / outH;  // ceil
        int w_start = (w_ * W) / outW;
        int w_end   = ((w_ + 1) * W + outW - 1) / outW;  // ceil

        // 안전하게 클램프
        if (h_start < 0) h_start = 0;
        if (h_end   > H) h_end   = H;
        if (w_start < 0) w_start = 0;
        if (w_end   > W) w_end   = W;

        float sum   = 0.0;
        int   count = 0;

        for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
                int idxIn = (h * W + w) * C + c;
                sum += in0[idxIn];
                count++;
            }
        }

        float avg = (count > 0) ? (sum / float(count)) : 0.0;

        int idxOut = (h_ * outW + w_) * C + c;
        out0[idxOut] = avg;
    }
)";

static const char* src_hs = R"(
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
    float clip = min(max(in0[o]+3, 0.0f), 6.0f) / 6.0f;
    out0[o] = in0[o] * clip;
})";

static const char* src_multiply = R"(
    #version 450
    layout(local_size_x = 64) in;
    
    layout(set = 0, binding = 0) writeonly buffer OutBuffer { float out0[]; };
    layout(set = 0, binding = 1) readonly buffer InBuffer { float in0[]; };
    layout(set = 0, binding = 2) readonly buffer AttenBuffer { float atten[]; };
    
    layout(push_constant) uniform PushConstants {
        int H;
        int W;
        int C;
    };
    
    void main()
    {
        int idx = int(gl_GlobalInvocationID.x);
        int total = H * W * C;
        if (idx >= total)
            return;
    
        int c = idx % C;
    
        float x = in0[idx];
        float s = atten[c];
        out0[idx] = x * s;
    }
)"; 

static const char* src_batchnorm2d = R"(
    #version 450
    layout(local_size_x = 64) in;
    
    layout(set = 0, binding = 0) writeonly buffer OutBuffer { float out0[]; };
    layout(set = 0, binding = 1) readonly  buffer InBuffer  { float in0[]; };
    layout(set = 0, binding = 2) readonly  buffer GammaBuffer  { float gamma[]; };
    layout(set = 0, binding = 3) readonly  buffer BetaBuffer   { float beta[]; };
    layout(set = 0, binding = 4) readonly  buffer MeanBuffer   { float running_mean[]; };
    layout(set = 0, binding = 5) readonly  buffer VarBuffer    { float running_var[]; };
    
    layout(push_constant) uniform PushConstants {
        int H;
        int W;
        int C;
        float eps;
    };
    
    void main()
    {
        int idx = int(gl_GlobalInvocationID.x);
        int total = H * W * C;
        if (idx >= total) return;
    
        int c = idx % C;
    
        float x = in0[idx];
        float mu = running_mean[c];
        float var = running_var[c];
    
        float x_hat = (x - mu) * inversesqrt(var + eps);
        out0[idx] = gamma[c] * x_hat + beta[c];
    }
)";

static const char* src_channel_shuffle = R"(
    #version 450
    layout(local_size_x = 64) in;

    // out_even: even-indexed channels  (0,2,4,...)
    // out_odd : odd-indexed channels   (1,3,5,...)
    layout(set = 0, binding = 0) writeonly buffer OutEven { float out_even[]; };
    layout(set = 0, binding = 1) writeonly buffer OutOdd  { float out_odd[];  };
    layout(set = 0, binding = 2) readonly  buffer InBuf   { float in0[];      };

    layout(push_constant) uniform PushConstants {
        int H;
        int W;
        int C;   // total channels, assumed even
    };

    void main()
    {
        int idx = int(gl_GlobalInvocationID.x);
        int C2  = C / 2;
        int total = H * W * C2;
        if (idx >= total) return;

        int hw = idx / C2;
        int k  = idx % C2;
        int h  = hw / W;
        int w  = hw % W;

        int base_in = (h * W + w) * C;
        int c_even  = 2 * k;
        int c_odd   = 2 * k + 1;

        out_even[idx] = in0[base_in + c_even];
        out_odd[idx]  = in0[base_in + c_odd];
    }
)";

static const char* src_concat = R"(
    #version 450
    layout(local_size_x = 64) in;

    // Concatenate along channel dimension:
    // in0: (H, W, C0)
    // in1: (H, W, C1)
    // out0: (H, W, C0 + C1)
    layout(set = 0, binding = 0) writeonly buffer OutBuffer { float out0[]; };
    layout(set = 0, binding = 1) readonly  buffer InBuffer0 { float in0[];  };
    layout(set = 0, binding = 2) readonly  buffer InBuffer1 { float in1[];  };

    layout(push_constant) uniform PushConstants {
        int H;
        int W;
        int C0;
        int C1;
    };

    void main()
    {
        int idx = int(gl_GlobalInvocationID.x);
        int C  = C0 + C1;
        int total = H * W * C;
        if (idx >= total) return;

        int hw = idx / C;
        int c  = idx % C;
        int h  = hw / W;
        int w  = hw % W;

        if (c < C0)
        {
            int base0 = (h * W + w) * C0;
            out0[idx] = in0[base0 + c];
        }
        else
        {
            int c1 = c - C0;
            int base1 = (h * W + w) * C1;
            out0[idx] = in1[base1 + c1];
        }
    }
)";

static const char* src_copy2 = R"(
    #version 450
    layout(local_size_x = 64) in;
    layout(set = 0, binding = 0) buffer Out0 { float out0[]; };
    layout(set = 0, binding = 1) buffer Out1 { float out1[]; };
    layout(set = 0, binding = 2) readonly buffer In0 { float in0[]; };
    layout(push_constant) uniform PushConstants {
        int N; // total elements
    };
    
    void main()
    {
        int i = int(gl_GlobalInvocationID.x);
        if (i >= N) return;
        float v = in0[i];
        out0[i] = v;
        out1[i] = v;
    }
)";

Device netGlobalDevice = VulkanApp::get().device();

static DescriptorPool gDestSetPool = netGlobalDevice.createDescriptorPool({
    .maxTypes = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER <= 200}, 
    .maxSets = 100
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
    requestPipeline(src_batchnorm2d);
    requestPipeline(src_channel_shuffle);
    requestPipeline(src_concat);

    // TODO: impelement shaders
    // requestPipeline(src_adaptiveavgpool);
    // requestPipeline(src_hs);
    // requestPipeline(src_batchnorm2d);
}

/////////////////////////////////////////////////////////////////////////////////////////
// ConvolutionNode
/////////////////////////////////////////////////////////////////////////////////////////
ConvolutionNode::ConvolutionNode(uint32_t inChannels, uint32_t outChannels, uint32_t kernelWidth, uint32_t stride_, int32_t padding_)
:  C(inChannels)
,  F(outChannels)
,  K(kernelWidth)
,  stride(stride_)
,  padding(padding_ >= 0 ? padding_ : 0)
{
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
    _ASSERT((*this)["weight"].isShapeOf(C*K*K, F));
    _ASSERT((*this)["bias"].isShapeOf(F));

    const auto& inShape = (*this)["in0"].shape();
    int32_t H = static_cast<int32_t>(inShape[0]);
    int32_t W = static_cast<int32_t>(inShape[1]);
    int32_t numeratorH = H + 2 * padding - static_cast<int32_t>(K);
    int32_t numeratorW = W + 2 * padding - static_cast<int32_t>(K);
    int32_t outH = numeratorH / static_cast<int32_t>(stride) + 1;
    int32_t outW = numeratorW / static_cast<int32_t>(stride) + 1;

    uint32_t outH_u = static_cast<uint32_t>(outH);
    uint32_t outW_u = static_cast<uint32_t>(outW);

    (*this)["im2colOut"] = Tensor(outH_u, outW_u, C*K*K);
    (*this)["out0"] = Tensor(outH_u, outW_u, F);
}

void ConvolutionNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    uint32_t H = inShape[0], W = inShape[1];
    int32_t numeratorH = static_cast<int32_t>(H) + 2 * padding - static_cast<int32_t>(K);
    int32_t numeratorW = static_cast<int32_t>(W) + 2 * padding - static_cast<int32_t>(K);
    int32_t outH = numeratorH / static_cast<int32_t>(stride) + 1;
    int32_t outW = numeratorW / static_cast<int32_t>(stride) + 1;
    uint32_t outH_u = static_cast<uint32_t>(outH);
    uint32_t outW_u = static_cast<uint32_t>(outW);
    uint32_t HW_out = outH_u * outW_u;

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

    int32_t im2colConstants[] = {
        static_cast<int32_t>(H), static_cast<int32_t>(W),
        static_cast<int32_t>(C), static_cast<int32_t>(K),
        static_cast<int32_t>(stride), padding,
        static_cast<int32_t>(outH_u), static_cast<int32_t>(outW_u)
    };

    uint32_t M = HW_out;        // N
    uint32_t K_ = C * K * K;    // I
    uint32_t N = F;             // O
    uint32_t gemmConstants[] = {M, K_, N};

    cmdBuff
        .bindPipeline(im2col)
        .bindDescSets({im2colDescSet})
        .setPushConstants(0, sizeof(im2colConstants), im2colConstants)
        .dispatch(CEIL_DIV(HW_out, 64), CEIL_DIV(K_, 16))
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
// MaxPoolingNode
/////////////////////////////////////////////////////////////////////////////////////////
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

    /*const char* gemmSrc = src_gemm_naive;
    gemmTileSize = gGemmTileSize.at(gemmSrc);*/

    const char* gemmSrc = src_gemm_kSplit;

    gemm = requestPipeline(gemmSrc);
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
        .dispatch0(CEIL_DIV(N, 32), M, CEIL_DIV(K, 16))
        //.dispatch0(CEIL_DIV(N, gemmTileSize), CEIL_DIV(M, gemmTileSize))
        .barrier( 
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );

}  

/////////////////////////////////////////////////////////////////////////////////////////
// AdaptiveAvgPoolingNode
/////////////////////////////////////////////////////////////////////////////////////////
AdaptiveAvgPoolingNode::AdaptiveAvgPoolingNode(uint32_t outputSize)
: outH(outputSize), outW(outputSize)
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);

    avgpool = requestPipeline(src_adaptiveavgpool);
    avgpoolDescSet = avgpool.descSetLayout(0).newDescSet(gDestSetPool);
}

AdaptiveAvgPoolingNode::AdaptiveAvgPoolingNode(uint32_t outH_, uint32_t outW_)
: outH(outH_), outW(outW_)
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);

    avgpool = requestPipeline(src_adaptiveavgpool);
    avgpoolDescSet = avgpool.descSetLayout(0).newDescSet(gDestSetPool);
}

void AdaptiveAvgPoolingNode::prepare()
{
    const auto& inShape = (*this)["in0"].shape();
    _ASSERT(inShape.size() == 3);
    uint32_t C = inShape[2];
    (*this)["out0"] = Tensor(outH, outW, C);
}

void AdaptiveAvgPoolingNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    uint32_t H = inShape[0], W = inShape[1], C = inShape[2];

    avgpoolDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
    });

    uint32_t avgpoolConstants[] = {H, W, C, outH, outW};

    cmdBuff
        .bindPipeline(avgpool)
        .bindDescSets({avgpoolDescSet})
        .setPushConstants(0, sizeof(avgpoolConstants), avgpoolConstants)
        .dispatch(outH, outW, C)
        .barrier( 
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}  

/////////////////////////////////////////////////////////////////////////////////////////
// HSNode
/////////////////////////////////////////////////////////////////////////////////////////
HSNode::HSNode()
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);

    hs = requestPipeline(src_hs);
    hsDescSet = hs.descSetLayout(0).newDescSet(gDestSetPool);
}

void HSNode::prepare()
{
    _ASSERT((*this)["in0"].validShape());
    (*this)["out0"] = Tensor((*this)["in0"].shape());
}

void HSNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    int I = 1;
    for (int dim : inShape) I *= dim;
    
    hsDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
    });

    int hsConstants[] = {I};

    cmdBuff
        .bindPipeline(hs)
        .setPushConstants(0, sizeof(hsConstants), hsConstants)
        .bindDescSets({hsDescSet})
        .dispatch(I)
        .barrier( 
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}  

/////////////////////////////////////////////////////////////////////////////////////////
// BatchNormNode
/////////////////////////////////////////////////////////////////////////////////////////

BatchNormNode::BatchNormNode(uint32_t channel)
: C(channel)
{
    addSlot("in0", NodeSlot::input);
    addSlot("out0", NodeSlot::output);
    addSlot("gamma", NodeSlot::input);
    addSlot("beta", NodeSlot::input);
    addSlot("running_mean", NodeSlot::input);
    addSlot("running_var", NodeSlot::input);

    bn = requestPipeline(src_batchnorm2d);
    bnDescSet = bn.descSetLayout(0).newDescSet(gDestSetPool);
}

void BatchNormNode::prepare()
{
    _ASSERT((*this)["in0"].isShapeOf(-1, -1, C));
    _ASSERT((*this)["gamma"].isShapeOf(C));
    _ASSERT((*this)["beta"].isShapeOf(C));
    _ASSERT((*this)["running_mean"].isShapeOf(C));
    _ASSERT((*this)["running_var"].isShapeOf(C));
    const auto& inShape = (*this)["in0"].shape();
    (*this)["out0"] = Tensor(inShape);
}

void BatchNormNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    uint32_t H = inShape[0], W = inShape[1];
    // C is a member; also verify runtime if needed
    uint32_t C_ = inShape[2];

    bnDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
        (*this)["gamma"].buffer(),
        (*this)["beta"].buffer(),
        (*this)["running_mean"].buffer(),
        (*this)["running_var"].buffer(),
    });

    struct { uint32_t H, W, C; float eps; } push { H, W, C_, 1e-5f };

    cmdBuff
        .bindPipeline(bn)
        .bindDescSets({bnDescSet})
        .setPushConstants(0, sizeof(push), &push)
        .dispatch(H * W * C_)
        .barrier( 
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}  

/////////////////////////////////////////////////////////////////////////////////////////
// MultiplyNode (channel-wise broadcast multiply)
/////////////////////////////////////////////////////////////////////////////////////////
MultiplyNode::MultiplyNode()
{
    addSlot("in0", NodeSlot::input);
    addSlot("atten", NodeSlot::input);
    addSlot("out0", NodeSlot::output);

    multiply = requestPipeline(src_multiply);
    multiplyDescSet = multiply.descSetLayout(0).newDescSet(gDestSetPool);
}

void MultiplyNode::prepare()
{
    const auto& inShape = (*this)["in0"].shape();
    _ASSERT(inShape.size() == 3);
    uint32_t H = inShape[0], W = inShape[1], C = inShape[2];

    const auto& aShape = (*this)["atten"].shape();
    _ASSERT( (aShape.size() == 1 && aShape[0] == C)
          || (aShape.size() == 3 && aShape[0] == 1 && aShape[1] == 1 && aShape[2] == C));

    (*this)["out0"] = Tensor(H, W, C);
}

void MultiplyNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    uint32_t H = inShape[0], W = inShape[1], C = inShape[2];

    multiplyDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
        (*this)["atten"].buffer(),
    });

    uint32_t constants[] = {H, W, C};

    cmdBuff
        .bindPipeline(multiply)
        .bindDescSets({multiplyDescSet})
        .setPushConstants(0, sizeof(constants), constants)
        .dispatch(H * W * C)
        .barrier( 
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}  

/////////////////////////////////////////////////////////////////////////////////////////
// ChannelShuffleNode
/////////////////////////////////////////////////////////////////////////////////////////
ChannelShuffleNode::ChannelShuffleNode(uint32_t channels)
: C(channels)
{
    addSlot("in0", NodeSlot::input);
    addSlot("out_even", NodeSlot::output);
    addSlot("out_odd",  NodeSlot::output);

    cs = requestPipeline(src_channel_shuffle);
    csDescSet = cs.descSetLayout(0).newDescSet(gDestSetPool);
}

void ChannelShuffleNode::prepare()
{
    _ASSERT((*this)["in0"].isShapeOf(-1, -1, C));
    _ASSERT((C % 2) == 0);

    const auto& inShape = (*this)["in0"].shape();
    uint32_t H = inShape[0];
    uint32_t W = inShape[1];
    uint32_t C2 = C / 2;

    (*this)["out_even"] = Tensor(H, W, C2);
    (*this)["out_odd"]  = Tensor(H, W, C2);
}

void ChannelShuffleNode::run(CommandBuffer cmdBuff)
{
    const auto& inShape = (*this)["in0"].shape();
    uint32_t H = inShape[0];
    uint32_t W = inShape[1];
    uint32_t C2 = C / 2;
    uint32_t total = H * W * C2;

    csDescSet.write({
        (*this)["out_even"].buffer(),
        (*this)["out_odd"].buffer(),
        (*this)["in0"].buffer(),
    });

    struct { int H, W, C; } push { int(H), int(W), int(C) };

    cmdBuff
        .bindPipeline(cs)
        .bindDescSets({csDescSet})
        .setPushConstants(0, sizeof(push), &push)
        .dispatch(CEIL_DIV(total, 64))
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out_even"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}

/////////////////////////////////////////////////////////////////////////////////////////
// ConcatNode (channel-wise concatenation)
/////////////////////////////////////////////////////////////////////////////////////////
ConcatNode::ConcatNode()
{
    addSlot("in0",  NodeSlot::input);
    addSlot("in1",  NodeSlot::input);
    addSlot("out0", NodeSlot::output);

    concat = requestPipeline(src_concat);
    concatDescSet = concat.descSetLayout(0).newDescSet(gDestSetPool);
}

void ConcatNode::prepare()
{
    const auto& s0 = (*this)["in0"].shape();
    const auto& s1 = (*this)["in1"].shape();
    _ASSERT(s0.size() == 3 && s1.size() == 3);
    _ASSERT(s0[0] == s1[0] && s0[1] == s1[1]);

    uint32_t H  = s0[0];
    uint32_t W  = s0[1];
    uint32_t C0 = s0[2];
    uint32_t C1 = s1[2];

    (*this)["out0"] = Tensor(H, W, C0 + C1);
}

void ConcatNode::run(CommandBuffer cmdBuff)
{
    const auto& s0 = (*this)["in0"].shape();
    const auto& s1 = (*this)["in1"].shape();
    uint32_t H  = s0[0];
    uint32_t W  = s0[1];
    uint32_t C0 = s0[2];
    uint32_t C1 = s1[2];
    uint32_t C  = C0 + C1;
    uint32_t total = H * W * C;

    concatDescSet.write({
        (*this)["out0"].buffer(),
        (*this)["in0"].buffer(),
        (*this)["in1"].buffer(),
    });

    struct { int H, W, C0, C1; } push { int(H), int(W), int(C0), int(C1) };

    cmdBuff
        .bindPipeline(concat)
        .bindDescSets({concatDescSet})
        .setPushConstants(0, sizeof(push), &push)
        .dispatch(CEIL_DIV(total, 64))
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}

/////////////////////////////////////////////////////////////////////////////////////////
// SplitNode (duplicate input to two outputs)
/////////////////////////////////////////////////////////////////////////////////////////
SplitNode::SplitNode()
{
    addSlot("in0",  NodeSlot::input);
    addSlot("out0", NodeSlot::output);
    addSlot("out1", NodeSlot::output);

    dup2 = requestPipeline(src_copy2);
    dup2DescSet = dup2.descSetLayout(0).newDescSet(gDestSetPool);
}

void SplitNode::prepare()
{
    _ASSERT((*this)["in0"].validShape());
    const auto& s = (*this)["in0"].shape();
    if (s.size() == 3)
        (*this)["out0"] = Tensor(s[0], s[1], s[2]);
    else
        (*this)["out0"] = Tensor(s);
    (*this)["out1"] = (*this)["out0"]; // same shape
}

void SplitNode::run(CommandBuffer cmdBuff)
{
    const auto& s = (*this)["in0"].shape();
    int N = 1; for (auto d : s) N *= int(d);

    dup2DescSet.write({
        (*this)["out0"].buffer(),
        (*this)["out1"].buffer(),
        (*this)["in0"].buffer(),
    });

    cmdBuff
        .bindPipeline(dup2)
        .bindDescSets({dup2DescSet})
        .setPushConstants(0, sizeof(N), &N)
        .dispatch(N)
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out0"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        )
        .barrier(
            (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_WRITE)
            / (*this)["out1"].buffer()
            / (PIPELINE_STAGE::COMPUTE_SHADER, ACCESS::SHADER_READ)
        );
}

// 1. Concat Shader - Concatenates tensors along specified dimension (CHW layout)
static const char* src_concat = R"(
#version 450
// 한 워크그룹당 256 스레드
layout(local_size_x = 256) in;

// 논리 레이아웃: [Outer, Axis, Inner]
// HWC 텐서에서 예:
//   - 채널(C) 축 기준 concat → Axis = C, Outer = H*W, Inner = 1
//   - 공간(H, W) 축 기준 concat 도 동일 원리

layout(set = 0, binding = 0) writeonly buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) readonly  buffer InBuffer0 { float in0[];  };
layout(set = 0, binding = 2) readonly  buffer InBuffer1 { float in1[];  };

layout(push_constant) uniform PushConstants {
    uint outer_size; // concat 축 이전 차원들의 곱 (없으면 1)
    uint inner_size; // concat 축 이후 차원들의 곱 (없으면 1)
    uint axis_dim0;  // 입력 0의 concat 축 크기
    uint axis_dim1;  // 입력 1의 concat 축 크기
};

void main()
{
    uint idx = gl_GlobalInvocationID.x;

    uint axis_total     = axis_dim0 + axis_dim1;
    uint total_elements = outer_size * axis_total * inner_size;

    if (idx >= total_elements) return;

    // idx = (outer_idx * axis_total * inner_size)
    //     + (axis_idx  * inner_size)
    //     + inner_idx
    uint inner_idx = idx % inner_size;
    uint tmp_idx   = idx / inner_size;
    uint axis_idx  = tmp_idx % axis_total;
    uint outer_idx = tmp_idx / axis_total;

    if (axis_idx < axis_dim0)
    {
        // Input 0: [Outer, Axis0, Inner]
        uint src_idx = (outer_idx * axis_dim0 * inner_size)
                     + (axis_idx  * inner_size)
                     + inner_idx;
        out0[idx] = in0[src_idx];
    }
    else
    {
        // Input 1: [Outer, Axis1, Inner]
        uint src_axis_idx = axis_idx - axis_dim0;
        uint src_idx = (outer_idx * axis_dim1 * inner_size)
                     + (src_axis_idx * inner_size)
                     + inner_idx;
        out0[idx] = in1[src_idx];
    }
}
)";


// 2. BatchNorm Shader - Inference mode with precomputed parameters (CHW layout)
static const char* src_batchnorm = R"(
#version 450
layout(local_size_x = 64) in;

layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer  { float in0[]; };
layout(set = 0, binding = 2) buffer Gamma     { float gamma[]; };    // scale
layout(set = 0, binding = 3) buffer Beta      { float beta[]; };     // shift
layout(set = 0, binding = 4) buffer Mean      { float mean[]; };     // running mean
layout(set = 0, binding = 5) buffer Var       { float variance[]; }; // running variance

layout(push_constant) uniform PushConstants {
    int C;      // channels
    int H;      // height
    int W;      // width
    float eps;  // epsilon for numerical stability
};

void main()
{
    int idx = int(gl_GlobalInvocationID.x);
    int total_size = C * H * W;

    if (idx >= total_size)
        return;

    // HWC layout: flat index = (h * W + w) * C + c
    // → channel index만 필요하면 c = idx % C 로 얻으면 됨.
    int c = idx % C;

    float x = in0[idx];

    float std_inv   = inversesqrt(variance[c] + eps);
    float normalized = (x - mean[c]) * std_inv;

    out0[idx] = gamma[c] * normalized + beta[c];
}
)";


// 3. ConvTranspose Shader - Transposed Convolution with stride and padding (CHW layout)
static const char* src_conv_transpose = R"(
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 4) in;

// HWC layout으로 가정
// out0: [H_out, W_out, C_out]
// in0 : [H_in,  W_in,  C_in ]
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer  { float in0[]; };

// weight: [C_in * K * K, C_out]
//   row = c_in * K * K + kh * K + kw
//   col = c_out
layout(set = 0, binding = 2) buffer Weight { float weight[]; };

// bias: [C_out]
layout(set = 0, binding = 3) buffer Bias { float bias[]; };

layout(push_constant) uniform PushConstants {
    int C_in;       // input channels
    int H_in;       // input height
    int W_in;       // input width
    int C_out;      // output channels
    int H_out;      // output height
    int W_out;      // output width
    int K;          // kernel size (K x K)
    int stride;     // stride
    int padding;    // padding
};

void main()
{
    int h_out = int(gl_GlobalInvocationID.x);
    int w_out = int(gl_GlobalInvocationID.y);
    int c_out = int(gl_GlobalInvocationID.z);

    if (h_out >= H_out || w_out >= W_out || c_out >= C_out)
        return;

    float sum = bias[c_out];

    // deconv(ConvTranspose2D) 역연산 관점
    for (int c_in = 0; c_in < C_in; ++c_in)
    {
        for (int kh = 0; kh < K; ++kh)
        {
            for (int kw = 0; kw < K; ++kw)
            {
                // forward conv:
                //   h_out = h_in * stride - padding + kh
                //   w_out = w_in * stride - padding + kw
                //
                // => h_in = (h_out + padding - kh) / stride (단, 나누어 떨어져야 함)
                //    w_in = (w_out + padding - kw) / stride
                int h_in_raw = h_out + padding - kh;
                int w_in_raw = w_out + padding - kw;

                if (h_in_raw % stride != 0 || w_in_raw % stride != 0)
                    continue;

                int h_in = h_in_raw / stride;
                int w_in = w_in_raw / stride;

                if (h_in < 0 || h_in >= H_in ||
                    w_in < 0 || w_in >= W_in)
                    continue;

                // HWC: index = (h * W + w) * C + c
                int in_idx = (h_in * W_in + w_in) * C_in + c_in;

                // weight: [C_in * K * K, C_out]
                int filterIndex = c_in * K * K + kh * K + kw; // row
                int weight_idx  = filterIndex * C_out + c_out; // col

                sum += in0[in_idx] * weight[weight_idx];
            }
        }
    }

    int out_idx = (h_out * W_out + w_out) * C_out + c_out;
    out0[out_idx] = sum;
}
)";

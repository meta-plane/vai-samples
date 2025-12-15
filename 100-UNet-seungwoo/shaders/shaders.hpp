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
    // �� channel index�� �ʿ��ϸ� c = idx % C �� ������ ��.
    int c = idx % C;

    float x = in0[idx];

    float std_inv   = inversesqrt(variance[c] + eps);
    float normalized = (x - mean[c]) * std_inv;

    out0[idx] = gamma[c] * normalized + beta[c];
}
)";

//static const char* src_batchnorm2d = R"(
//    #version 450
//    layout(local_size_x = 64) in;
//    
//    layout(set = 0, binding = 0) writeonly buffer OutBuffer { float out0[]; };
//    layout(set = 0, binding = 1) readonly  buffer InBuffer  { float in0[]; };
//    layout(set = 0, binding = 2) readonly  buffer GammaBuffer  { float gamma[]; };
//    layout(set = 0, binding = 3) readonly  buffer BetaBuffer   { float beta[]; };
//    layout(set = 0, binding = 4) readonly  buffer MeanBuffer   { float running_mean[]; };
//    layout(set = 0, binding = 5) readonly  buffer VarBuffer    { float running_var[]; };
//    
//    layout(push_constant) uniform PushConstants {
//        int H;
//        int W;
//        int C;
//        float eps;
//    };
//    
//    void main()
//    {
//        int idx = int(gl_GlobalInvocationID.x);
//        int total = H * W * C;
//        if (idx >= total) return;
//    
//        int c = idx % C;
//    
//        float x = in0[idx];
//        float mu = running_mean[c];
//        float var = running_var[c];
//    
//        float x_hat = (x - mu) * inversesqrt(var + eps);
//        out0[idx] = gamma[c] * x_hat + beta[c];
//    }
//)";

// 3. ConvTranspose Shader - Transposed Convolution with stride and padding (CHW layout)
static const char* src_conv_transpose = R"(
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 4) in;

// HWC layout���� ����
// out0: [H_out, W_out, C_out]
// in0 : [H_in,  W_in,  C_in ]
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer  { float in0[]; };
layout(set = 0, binding = 2) buffer Weight { float weight[]; };
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

    // deconv(ConvTranspose2D)
    for (int c_in = 0; c_in < C_in; ++c_in)
    {
        for (int kh = 0; kh < K; ++kh)
        {
            for (int kw = 0; kw < K; ++kw)
            {
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

static const char* src_sigmoid = R"(
#version 450

layout(local_size_x = 64) in;

layout(set = 0, binding = 0) writeonly buffer OutBuffer { float out0[];};

layout(set = 0, binding = 1) readonly buffer InBuffer { float in0[]; };

layout(push_constant) uniform PushConstants { int num_elements; };

float sigmoid(float x)
{
    if (x >= 0.0) {
        float e = exp(-x);
        return 1.0 / (1.0 + e);
    } else {
        float e = exp(x);
        return e / (1.0 + e);
    }
}

void main()
{
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= num_elements) return;

    out0[idx] = sigmoid(in0[idx]);
}
)";



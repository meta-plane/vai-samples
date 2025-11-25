// 1. Concat Shader - Concatenates tensors along specified dimension (CHW layout)
static const char* src_concat = R"(
    #version 450
    layout(local_size_x = 256) in;

    // Generic Concatenation Shader
    // Supports concatenation along any dimension by flattening the tensor conceptually.
    //
    // logical layout: [Outer, Axis, Inner]
    // in0: [Outer, Axis0, Inner]
    // in1: [Outer, Axis1, Inner]
    // out0: [Outer, Axis0 + Axis1, Inner]

    layout(set = 0, binding = 0) writeonly buffer OutBuffer { float out0[]; };
    layout(set = 0, binding = 1) readonly  buffer InBuffer0 { float in0[];  };
    layout(set = 0, binding = 2) readonly  buffer InBuffer1 { float in1[];  };

    layout(push_constant) uniform PushConstants {
        uint outer_size; // Concat 축 이전 차원들의 곱 (없으면 1)
        uint inner_size; // Concat 축 이후 차원들의 곱 (없으면 1)
        uint axis_dim0;  // 입력 0의 Concat 축 크기
        uint axis_dim1;  // 입력 1의 Concat 축 크기
    };

    void main()
    {
        uint idx = gl_GlobalInvocationID.x;
        
        // 출력 텐서의 Axis 크기 합
        uint axis_total = axis_dim0 + axis_dim1;
        
        // 전체 요소 개수 계산
        uint total_elements = outer_size * axis_total * inner_size;

        if (idx >= total_elements) return;

        // 1. 현재 스레드가 처리하는 논리적 좌표 계산
        // idx = (outer_idx * axis_total * inner_size) + (axis_idx * inner_size) + inner_idx
        
        uint inner_idx = idx % inner_size;
        uint tmp_idx   = idx / inner_size;
        uint axis_idx  = tmp_idx % axis_total;
        uint outer_idx = tmp_idx / axis_total;

        // 2. 좌표가 어느 입력 텐서에 속하는지 판별하고 데이터 복사
        if (axis_idx < axis_dim0)
        {
            // Input 0에서 읽기
            // Input 0의 메모리 레이아웃: [Outer, Axis0, Inner]
            uint src_idx = (outer_idx * axis_dim0 * inner_size) + (axis_idx * inner_size) + inner_idx;
            out0[idx] = in0[src_idx];
        }
        else
        {
            // Input 1에서 읽기
            // Input 1의 좌표로 변환 (axis_idx에서 앞부분 offset 제거)
            uint src_axis_idx = axis_idx - axis_dim0;
            
            // Input 1의 메모리 레이아웃: [Outer, Axis1, Inner]
            uint src_idx = (outer_idx * axis_dim1 * inner_size) + (src_axis_idx * inner_size) + inner_idx;
            out0[idx] = in1[src_idx];
        }
    }
)";

// 2. BatchNorm Shader - Inference mode with precomputed parameters (CHW layout)
static const char* src_batchnorm = R"(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in0[]; };
layout(set = 0, binding = 2) buffer Gamma { float gamma[]; };      // scale
layout(set = 0, binding = 3) buffer Beta { float beta[]; };        // shift
layout(set = 0, binding = 4) buffer Mean { float mean[]; };        // running mean
layout(set = 0, binding = 5) buffer Var { float variance[]; };     // running variance
layout(push_constant) uniform PushConstants {
    int C;          // channels
    int H;          // height
    int W;          // width
    float eps;      // epsilon for numerical stability
};

void main() 
{
    int idx = int(gl_GlobalInvocationID.x);
    int total_size = C * H * W;
    
    if (idx >= total_size) return;
    
    // Extract channel index from linear index (CHW layout)
    int c = idx / (H * W);
    
    // BatchNorm inference: y = gamma * (x - mean) / sqrt(var + eps) + beta
    float x = in0[idx];
    float std_inv = inversesqrt(variance[c] + eps);
    float normalized = (x - mean[c]) * std_inv;
    out0[idx] = gamma[c] * normalized + beta[c];
})";

// 3. ConvTranspose Shader - Transposed Convolution with stride and padding (CHW layout)
static const char* src_conv_transpose = R"(
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 4) in;
layout(set = 0, binding = 0) buffer OutBuffer { float out0[]; };
layout(set = 0, binding = 1) buffer InBuffer { float in0[]; };
layout(set = 0, binding = 2) buffer Weight { float weight[]; };
layout(set = 0, binding = 3) buffer Bias { float bias[]; };
layout(push_constant) uniform PushConstants {
    int C_in;       // input channels
    int H_in;       // input height
    int W_in;       // input width
    int C_out;      // output channels
    int H_out;      // output height
    int W_out;      // output width
    int K;          // kernel size
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
    
    // For each output position, accumulate contributions from all input positions
    // that could have contributed to it in the forward convolution
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                // Calculate which input position contributes to this output position
                // Reverse the convolution operation:
                // out[h_out, w_out] += in[h_in, w_in] * weight[kh, kw]
                // where h_out = h_in * stride - padding + kh
                //       w_out = w_in * stride - padding + kw
                
                int h_in_raw = h_out + padding - kh;
                int w_in_raw = w_out + padding - kw;
                
                // Check if this maps to a valid input position
                if (h_in_raw % stride == 0 && w_in_raw % stride == 0) {
                    int h_in = h_in_raw / stride;
                    int w_in = w_in_raw / stride;
                    
                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        // CHW layout: index = c * H * W + h * W + w
                        int in_idx = (c_in * H_in + h_in) * W_in + w_in;
                        // Weight layout: [C_out, C_in, K, K]
                        int weight_idx = ((c_out * C_in + c_in) * K + kh) * K + kw;
                        sum += in0[in_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    
    // CHW layout output
    int out_idx = (c_out * H_out + h_out) * W_out + w_out;
    out0[out_idx] = sum;
})";
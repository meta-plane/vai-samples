#include "neuralNet.h"
#include "neuralNodes.h"
#include "jsonParser.h"
#include "timeChecker.hpp"
#include <stb/stb_image.h>
#include <cstring>  // memcpy

#include "mobilenetv2_spec.h"


template<uint32_t Channels>
auto readImage(const char* filename)
{
    int w, h, c0, c = Channels;
    std::vector<uint8_t> srcImage;

    if (uint8_t* input = stbi_load(filename, &w, &h, &c0, c))
    {
        srcImage.assign(input, input + w * h * c);
        stbi_image_free(input);
    }
    else
    {
        printf(stbi_failure_reason());
        fflush(stdout);
        throw;
    }

    return std::make_tuple(srcImage, (uint32_t)w, (uint32_t)h);
}

struct ConvBnFused {
    Tensor weight; // fused W'
    Tensor bias;   // fused b'
};

// convW: [C_out, C_in, kH, kW]
// convB: [C_out] or empty (bias 없는 conv이면 nullptr)
ConvBnFused fuseConvBn(
    const Tensor& convW,
    const Tensor* convB,        // nullptr이면 bias 없음
    const Tensor& bnWeight,     // gamma
    const Tensor& bnBias,       // beta
    const Tensor& bnRunningMean,
    const Tensor& bnRunningVar,
    float eps)
{
    _ASSERT(convW.isShapeOf(-1, -1, -1, -1)); // 4D
    const auto& wShape = convW.shape();
    uint32_t C_out = wShape[0];
    uint32_t C_in  = wShape[1];
    uint32_t kH    = wShape[2];
    uint32_t kW    = wShape[3];

    // 1) W 복사 (깊은 복사)
    Tensor fusedW = convW.clone();
    _ASSERT(fusedW.hasHostData());
    float* wData = fusedW.hostData(); // [C_out * C_in * kH * kW]

    // 2) B 생성 및 초기화
    Tensor fusedB({C_out});
    {
        std::vector<float> buf(C_out, 0.0f);
        fusedB.set(std::move(buf));
    }
    float* bData = fusedB.hostData();

    const float* gamma = bnWeight.hostData();
    const float* beta  = bnBias.hostData();
    const float* mean  = bnRunningMean.hostData();
    const float* var   = bnRunningVar.hostData();

    const float* convBData = convB && convB->hasHostData() ? convB->hostData() : nullptr;

    size_t channelSize = size_t(C_in) * kH * kW; // 한 output channel당 element 개수

    for (uint32_t oc = 0; oc < C_out; ++oc)
    {
        float g      = gamma[oc];
        float bt     = beta[oc];
        float m      = mean[oc];
        float v      = var[oc];
        float invStd = 1.0f / std::sqrt(v + eps);
        float b      = convBData ? convBData[oc] : 0.0f;

        float scale = g * invStd;

        // W'[oc, :, :, :] = W[oc, :, :, :] * scale
        size_t offset = size_t(oc) * channelSize;
        for (size_t i = 0; i < channelSize; ++i)
            wData[offset + i] *= scale;

        // b'_oc = (b - mean) * scale + beta
        bData[oc] = (b - m) * scale + bt;
    }

    return { fusedW, fusedB };
}

void loadConvBnBlock(MobileNetV2& net, const JsonParser& json, const ConvBnSpec& spec)
{
    // 1) Load conv weight / bias
    Tensor convW = Tensor(json[spec.json_conv_w]);
    Tensor convB;
    Tensor* convB_ptr = nullptr;
    if (!spec.json_conv_b.empty()) { // conv bias는 보통 존재하지 않음, 있는 경우에만 load
        convB = Tensor(json[spec.json_conv_b]);
        convB_ptr = &convB;
    }

    // 2) Load BN params
    Tensor bnW  = Tensor(json[spec.json_bn_w]);
    Tensor bnB  = Tensor(json[spec.json_bn_b]);
    Tensor bnRm = Tensor(json[spec.json_bn_rm]); // running_mean
    Tensor bnRv = Tensor(json[spec.json_bn_rv]); // running_var
    float  eps  = json[spec.json_bn_eps].asFloat();

    // 3) Conv+BN fuse
    auto fused = fuseConvBn(convW, convB_ptr, bnW, bnB, bnRm, bnRv, eps);

    Tensor fusedW = fused.weight;
    Tensor fusedB = fused.bias;

    // 4) Format conversion: PyTorch [OC, IC, kH, kW] → VAI framework [IC*kH*kW, OC]
    if (spec.transpose_OC_ICk) {
        fusedW = fusedW.reshape(spec.OC, spec.IC * spec.kH * spec.kW).permute(1, 0);
    }

    // 5) Set to NeuralNet
    net[spec.net_conv_w] = fusedW;
    net[spec.net_conv_b] = fusedB;
}

void loadMobilenetV2Weights(MobileNetV2& net, const JsonParser& json)
{
    // Load convolution + batchnorm blocks
    for (const auto& spec : g_mobilenetv2_convs) 
    {
        loadConvBnBlock(net, json, spec);
    }

    // Load FC layer weights
    Tensor fcW = Tensor(json["classifier.1.weight"]);
    Tensor fcB = Tensor(json["classifier.1.bias"]);

    net["fc.weight"] = fcW;
    net["fc.bias"]   = fcB;
}

class ConvBlock : public NodeGroup
{
    uint32_t C, F, K;
    ConvolutionNode conv;
    ReluNode relu;
    MaxPoolingNode maxpool;

public:
    ConvBlock(uint32_t inChannels, uint32_t outChannels, uint32_t kernel)
    : C(inChannels), F(outChannels), K(kernel),
    conv(inChannels, outChannels, kernel), maxpool(2)
    {
        conv - relu - maxpool;
        defineSlot("in0", conv.slot("in0"));
        defineSlot("out0", maxpool.slot("out0"));
    }

    Tensor& operator[](const std::string& name)
    {
        return conv[name];
    }
};

template<uint32_t nBlocks>
class ConvSequence : public NodeGroup
{
    uint32_t K;
    uint32_t C[nBlocks + 1];
    std::unique_ptr<ConvBlock> blocks[nBlocks];
    
public:
    ConvSequence(const uint32_t(&channels)[nBlocks + 1], uint32_t kernel)
    : K(kernel)
    {
        for (uint32_t i = 0; i <= nBlocks; ++i)
            C[i] = channels[i];

        for (uint32_t i = 0; i < nBlocks; ++i)
            blocks[i] = std::make_unique<ConvBlock>(C[i], C[i + 1], K);

        for (uint32_t i = 0; i < nBlocks - 1; ++i)
            *blocks[i] - *blocks[i + 1];

        defineSlot("in0", blocks[0]->slot("in0"));
        defineSlot("out0", blocks[nBlocks - 1]->slot("out0"));
    }

    Tensor& operator[](const std::string& name)
    {
        for (uint32_t i = 0; i < nBlocks; ++i)
            if (name.starts_with("conv" + std::to_string(i) + "."))
                return (*blocks[i])[name.substr(6)];
        throw std::runtime_error("No such layer in ConvSequence: " + name);
    }
};
template<std::size_t N>
ConvSequence(const uint32_t (&)[N], uint32_t) -> ConvSequence<N - 1>;


class MnistNet : public NeuralNet
{
    ConvSequence<2> convX2;
    FlattenNode flatten;
    FullyConnectedNode fc;

public:
    MnistNet(Device& device)
    : NeuralNet(device, 1, 1)
    , convX2({1, 32, 64}, 3)
    , fc(7 * 7 * 64, 10)
    {
        input(0) - convX2 - flatten - fc - output(0);
    }

    Tensor& operator[](const std::string& name)
    {
        if (name.starts_with("conv"))
            return convX2[name];  // conv0, conv1 등을 convX2에 전달
        else if (name == "weight" || name == "bias")
            return fc[name];  // fc의 weight와 bias
        else if (name.starts_with("fc."))
            return fc[name.substr(3)];
        else
            throw std::runtime_error("No such layer in MnistNet: " + name);
    }
};

class InvertedBottleneck : public NodeGroup
{
    // expand: 1x1 conv
    std::unique_ptr<ConvolutionNode> expandConv;
    std::unique_ptr<Relu6Node>       expandAct;

    // depthwise: 3x3 depthwise conv
    std::unique_ptr<DepthwiseConvNode> depthwiseConv;
    std::unique_ptr<Relu6Node>         depthwiseAct;

    // project: 1x1 conv
    std::unique_ptr<ConvolutionNode> projectConv;

    // skip 연결이 필요하면 AddNode 등 추가
    bool useResidual;

public:
    expandConv    = std::make_unique<ConvolutionNode>(inC, expandC, 1);
    expandAct     = std::make_unique<Relu6Node>();
    depthwiseConv = std::make_unique<DepthwiseConvNode>(expandC, 3, stride);
    depthwiseAct  = std::make_unique<Relu6Node>();
    projectConv   = std::make_unique<ConvolutionNode>(expandC, outC, 1);
    // (skip용 AddNode가 있다면 여기서 생성)


    // 내부 그래프 연결
    *expandConv - *expandAct - *depthwiseConv - *depthwiseAct - *projectConv;
    // residual이 있다면 projectConv 출력과 입력을 더하는 구조로 NodeGroup 안에서 연결

    defineSlot("in0",  expandConv->slot("in0"));
    defineSlot("out0", projectConv->slot("out0"));
}

Tensor& operator[](const std::string& name) {
    // "expand.weight", "depthwise.weight", "project.bias" 등 이름으로 접근 가능하게
    if (name.starts_with("expand."))
        return (*expandConv)[name.substr(7)];
    if (name.starts_with("depthwise."))
        return (*depthwiseConv)[name.substr(10)];
    if (name.starts_with("project."))
        return (*projectConv)[name.substr(8)];
    throw std::runtime_error("No such tensor in InvertedBottleneck: " + name);
};

class MobileNetV2 : public NeuralNet
{
    ConvSequence<2> convX2;
    FlattenNode flatten;
    FullyConnectedNode fc;

public:
    MnistNet(Device& device)
    : NeuralNet(device, 1, 1)
    , convX2({1, 32, 64}, 3)
    , fc(7 * 7 * 64, 10)
    {
        input(0) - convX2 - flatten - fc - output(0);
    }

    Tensor& operator[](const std::string& name)
    {
        if (name.starts_with("conv"))
            return convX2[name];  // conv0, conv1 등을 convX2에 전달
        else if (name == "weight" || name == "bias")
            return fc[name];  // fc의 weight와 bias
        else if (name.starts_with("fc."))
            return fc[name.substr(3)];
        else
            throw std::runtime_error("No such layer in MnistNet: " + name);
    }
};

Tensor eval_ImageNet(const std::vector<float>& srcImage, const JsonParser& json, uint32_t iter) // srcImage layout: [H][W][C]
{
    MobileNetV2 mobilenetv2(netGlobalDevice);

    loadMobilenetV2Weights(mobilenetv2, json);

    Tensor result;
    Tensor inputTensor = Tensor(28, 28, 1).set(srcImage);

    for (uint32_t i = 0; i < iter; ++i)
        result = mobilenetv2(inputTensor)[0];

    return result;
}


Tensor eval_mnist(const std::vector<float>& srcImage, const JsonParser& json, uint32_t iter) // srcImage layout: [H][W][C]
{
    MnistNet mnistNet(netGlobalDevice);

    // Conv weight: PyTorch [OC, IC, kH, kW] → VAI framework [IC*kH*kW, OC]
    // FC weight: PyTorch [OC, IC*kH*kW] → VAI framework [IC*kH*kW, OC]
    mnistNet["conv0.weight"] = Tensor(json["layer1.0.weight"]).reshape(32, 1*3*3).permute(1, 0);
    mnistNet["conv0.bias"] = Tensor(json["layer1.0.bias"]);
    mnistNet["conv1.weight"] = Tensor(json["layer2.0.weight"]).reshape(64, 32*3*3).permute(1, 0);
    mnistNet["conv1.bias"] = Tensor(json["layer2.0.bias"]);
    mnistNet["weight"] = Tensor(json["fc.weight"]).reshape(10, 64, 7*7).permute(2, 1, 0).reshape(7*7*64, 10);
    mnistNet["bias"] = Tensor(json["fc.bias"]);
    
    Tensor result;
    Tensor inputTensor = Tensor(28, 28, 1).set(srcImage);

    for (uint32_t i = 0; i < iter; ++i)
        result = mnistNet(inputTensor)[0];

    return result;
}

void test()
{
    void loadShaders();
    loadShaders();

    const uint32_t channels = 1;
    auto [srcImage, width, height] = readImage<channels>(PROJECT_ROOT_DIR"/103-MobileNetV2-SangIn/utils/0.png");
    _ASSERT(width == 28 && height == 28);
    _ASSERT(width * height * channels == srcImage.size());

    std::vector<float> inputData(width * height * channels);
    for (size_t i = 0; i < srcImage.size(); ++i)
        inputData[i] = srcImage[i] / 255.0f;

    JsonParser json = JsonParser(PROJECT_CURRENT_DIR"/utils/weights.json");

    uint32_t iter = 1;
    Tensor eval;

    {
        TimeChecker timer("(VAI) MNIST evaluation: {} iterations", iter);
        eval = eval_ImageNet(inputData, json, iter);
    }

    vk::Buffer outBuffer = netGlobalDevice.createBuffer({
        10 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });

    vk::Buffer evalBuffer = eval.buffer();
    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuffer, evalBuffer)
        .end()
        .submit()
        .wait();

    float data[10];
    memcpy(data, outBuffer.map(), 10 * sizeof(float));

    for(int i=0; i<10; ++i)
        printf("data[%d] = %f\n", i, data[i]);
}

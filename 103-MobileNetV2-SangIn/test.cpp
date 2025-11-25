#include "neuralNet.h"
#include "neuralNodes.h"
#include "jsonParser.h"
#include "timeChecker.hpp"
#include <stb/stb_image.h>
#include <cstring>  // memcpy
#include <iostream>
#include <stdexcept>



template<uint32_t Channels> // Template 함수, 호출되는 곳에서 Channels 값을 지정
auto readImage(const char* filename) 
// auto: 반환 타입을 자동으로 추론
// 여기서는 std::tuple<std::vector<uint8_t>, uint32_t, uint32_t>
{
    int w, h, c0, c = Channels;
    std::vector<uint8_t> srcImage;

    if (uint8_t* input = stbi_load(filename, &w, &h, &c0, c))
    {
        srcImage.assign(input, input + w * h * c);
        stbi_image_free(input); // stbi_load로부터 받은 메모리 해제
    }
    else
    {
        const char* reason = stbi_failure_reason();
        std::string errorMsg = "Failed to load image from " + std::string(filename);
        if (reason)
            errorMsg += ": " + std::string(reason);
        throw std::runtime_error(errorMsg);
    }

    return std::make_tuple(srcImage, (uint32_t)w, (uint32_t)h);
}


class ConvBlock : public NodeGroup // NodeGroup을 public 상속 (부모의 public 멤버 -> 자식에서도 public ...)
{
// public: 외부에서 접근 가능
// protected: 자식 클래스에서만 접근 가능, 외부에서는 불가
// private: 오직 자기 클래스 내부에서만 접근 가능

// public: 등을 사용하기 전 부분은 기본이 private
    uint32_t C, F, K;
    ConvolutionNode conv;
    ReluNode relu;
    MaxPoolingNode maxpool;

public:
    ConvBlock(uint32_t inChannels, uint32_t outChannels, uint32_t kernel) // generator
    : C(inChannels), F(outChannels), K(kernel),       // 생성자의 초기화 리스트, 멤버 변수들을 생성과 동시에 초기화하는 구문
    conv(inChannels, outChannels, kernel, 1), maxpool(2) // C = inChannels, conv = ConvolutionNode(inChannels, outChannels, kernel, stride=1) ...
    {
        conv - relu - maxpool;                        // - 연산자를 오버로딩해서 노드들을 그래프처럼 연결하는 문법 설계
        defineSlot("in0", conv.slot("in0"));          // 부모 클래스의 함수, ConvBlock 전체의 "in0" 슬롯이 내부 Conv 노드의 "in0" 슬롯과 연결됨을 의미
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
    ConvSequence(const uint32_t(&channels)[nBlocks + 1], uint32_t kernel) // generator
    : K(kernel) // 멤버 변수 초기화
    {
        for (uint32_t i = 0; i <= nBlocks; ++i)
            C[i] = channels[i];

        for (uint32_t i = 0; i < nBlocks; ++i)
            blocks[i] = std::make_unique<ConvBlock>(C[i], C[i + 1], K); // in_channel, out_channel, kernel_size

        for (uint32_t i = 0; i < nBlocks - 1; ++i)
            *blocks[i] - *blocks[i + 1]; // conv block들을 연결

        defineSlot("in0", blocks[0]->slot("in0"));
        defineSlot("out0", blocks[nBlocks - 1]->slot("out0")); // 밖에서는 ConvSequence를 하나의 노드처럼 사용 가능
    }

    Tensor& operator[](const std::string& name)
    {
        for (uint32_t i = 0; i < nBlocks; ++i)
            if (name.starts_with("conv" + std::to_string(i) + ".")) // conv0.weight와 같은 이름을 가정, 해당 tensor에 접근 가능하도록
                return (*blocks[i])[name.substr(6)];

        throw std::runtime_error("No such layer in ConvSequence: " + name);
    }
};
template<std::size_t N>
ConvSequence(const uint32_t (&)[N], uint32_t) -> ConvSequence<N - 1>; // template deduction guide, uint32_t 타입의 배열을 받으면, ConvSequence<N - 1> 타입으로 추론


// Inverted Residual Block for MobileNetV2
// Structure: expand (1x1 conv) -> depthwise (3x3) -> project (1x1 conv)
// With residual connection if inChannels == outChannels and stride == 1
class InvertedResidualBlock : public NodeGroup
{
    uint32_t inChannels, outChannels, expansion, stride;
    ConvolutionNode expandConv;      // 1x1 expansion
    ReluNode relu1;
    DepthwiseConvolutionNode depthwiseConv;  // 3x3 depthwise
    ReluNode relu2;
    std::unique_ptr<MaxPoolingNode> stridePool;  // MaxPool for stride > 1
    ConvolutionNode projectConv;     // 1x1 projection
    bool useResidual;

public:
    InvertedResidualBlock(uint32_t inCh, uint32_t outCh, uint32_t exp, uint32_t str)
    : inChannels(inCh), outChannels(outCh), expansion(exp), stride(str),
      expandConv(inCh, inCh * exp, 1, 1),  // 1x1 conv, stride 1
      depthwiseConv(inCh * exp, 3),        // 3x3 depthwise (stride handled separately)
      projectConv(inCh * exp, outCh, 1, 1), // 1x1 conv, stride 1
      useResidual(inCh == outCh && str == 1)
    {
        if (stride > 1)
        {
            stridePool = std::make_unique<MaxPoolingNode>(stride);
            expandConv - relu1 - depthwiseConv - relu2 - *stridePool - projectConv;
        }
        else
        {
            expandConv - relu1 - depthwiseConv - relu2 - projectConv;
        }
        defineSlot("in0", expandConv.slot("in0"));
        defineSlot("out0", projectConv.slot("out0"));
    }

    Tensor& operator[](const std::string& name)
    {
        if (name.starts_with("expand."))
            return expandConv[name.substr(7)];
        else if (name.starts_with("depthwise."))
            return depthwiseConv[name.substr(10)];
        else if (name.starts_with("project."))
            return projectConv[name.substr(8)];
        else
            throw std::runtime_error("No such layer in InvertedResidualBlock: " + name);
    }
};


class MobileNetV2 : public NeuralNet
{
    // Initial conv layer
    ConvolutionNode initialConv;
    ReluNode initialRelu;
    
    // Inverted residual blocks
    // MobileNetV2 architecture: 
    // t=expansion, c=channels, n=repeats, s=stride
    // t=1, c=16, n=1, s=1
    // t=6, c=24, n=2, s=2 (first), s=1 (second)
    // t=6, c=32, n=3, s=2 (first), s=1 (others)
    // t=6, c=64, n=4, s=2 (first), s=1 (others)
    // t=6, c=96, n=3, s=1
    // t=6, c=160, n=3, s=2 (first), s=1 (others)
    // t=6, c=320, n=1, s=1
    
    std::vector<std::unique_ptr<InvertedResidualBlock>> blocks;
    
    // Final layers
    ConvolutionNode finalConv;
    ReluNode finalRelu;
    GlobalAveragePoolingNode globalAvgPool;
    FullyConnectedNode classifier;

public:
    MobileNetV2(Device& device)
    : NeuralNet(device, 1, 1)
    , initialConv(3, 32, 3, 2)  // 3 input channels (RGB), 32 output, 3x3 kernel, stride 2
    , finalConv(320, 1280, 1, 1)  // 1x1 conv to 1280 channels, stride 1
    , classifier(1280, 1000)  // 1000 classes for ImageNet
    {
        // Build inverted residual blocks
        // Block 0: t=1, c=16, n=1, s=1
        blocks.push_back(std::make_unique<InvertedResidualBlock>(32, 16, 1, 1));
        
        // Block 1-2: t=6, c=24, n=2, s=2 (first), s=1 (second)
        blocks.push_back(std::make_unique<InvertedResidualBlock>(16, 24, 6, 2));
        blocks.push_back(std::make_unique<InvertedResidualBlock>(24, 24, 6, 1));
        
        // Block 3-5: t=6, c=32, n=3, s=2 (first), s=1 (others)
        blocks.push_back(std::make_unique<InvertedResidualBlock>(24, 32, 6, 2));
        blocks.push_back(std::make_unique<InvertedResidualBlock>(32, 32, 6, 1));
        blocks.push_back(std::make_unique<InvertedResidualBlock>(32, 32, 6, 1));
        
        // Block 6-9: t=6, c=64, n=4, s=2 (first), s=1 (others)
        blocks.push_back(std::make_unique<InvertedResidualBlock>(32, 64, 6, 2));
        blocks.push_back(std::make_unique<InvertedResidualBlock>(64, 64, 6, 1));
        blocks.push_back(std::make_unique<InvertedResidualBlock>(64, 64, 6, 1));
        blocks.push_back(std::make_unique<InvertedResidualBlock>(64, 64, 6, 1));
        
        // Block 10-12: t=6, c=96, n=3, s=1
        blocks.push_back(std::make_unique<InvertedResidualBlock>(64, 96, 6, 1));
        blocks.push_back(std::make_unique<InvertedResidualBlock>(96, 96, 6, 1));
        blocks.push_back(std::make_unique<InvertedResidualBlock>(96, 96, 6, 1));
        
        // Block 13-15: t=6, c=160, n=3, s=2 (first), s=1 (others)
        blocks.push_back(std::make_unique<InvertedResidualBlock>(96, 160, 6, 2));
        blocks.push_back(std::make_unique<InvertedResidualBlock>(160, 160, 6, 1));
        blocks.push_back(std::make_unique<InvertedResidualBlock>(160, 160, 6, 1));
        
        // Block 16: t=6, c=320, n=1, s=1
        blocks.push_back(std::make_unique<InvertedResidualBlock>(160, 320, 6, 1));
        
        // Connect all layers
        input(0) - initialConv - initialRelu;
        
        NodeFlow flow = initialRelu;
        for (auto& block : blocks)
            flow = flow - *block;
        
        flow - finalConv - finalRelu - globalAvgPool - classifier - output(0);
    }

    Tensor& operator[](const std::string& name)
    {
        if (name == "initial.weight" || name == "initial.bias")
            return initialConv[name.substr(8)];
        else if (name == "final.weight" || name == "final.bias")
            return finalConv[name.substr(6)];
        else if (name == "classifier.weight" || name == "classifier.bias")
            return classifier[name.substr(11)];
        else if (name.starts_with("blocks."))
        {
            // Parse "blocks.N.layer.weight" format
            size_t dot1 = name.find('.', 7);  // After "blocks."
            if (dot1 != std::string::npos)
            {
                uint32_t blockIdx = std::stoi(name.substr(7, dot1 - 7));
                if (blockIdx < blocks.size())
                    return (*blocks[blockIdx])[name.substr(dot1 + 1)];
            }
        }
        throw std::runtime_error("No such layer in MobileNetV2: " + name);
    }
};


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

    Tensor& operator[](const std::string& name) // 
    {
        if (name.starts_with("conv"))                       // "conv0.weight" → convX2["conv0.weight"] → ConvBlock 0의 weight 텐서
            return convX2[name];
        else if (name == "weight" || name == "bias")        // fc의 weight와 bias
            return fc[name];
        else if (name.starts_with("fc."))                   // fc.weight와 같은 이름도 지원
            return fc[name.substr(3)];
        else
            throw std::runtime_error("No such layer in MnistNet: " + name);
    }
};


Tensor eval_mnist(const std::vector<float>& srcImage, const JsonParser& json, uint32_t iter) // srcImage layout: [H][W][C]
{
    try
    {
        MnistNet mnistNet(netGlobalDevice);

        mnistNet["conv0.weight"] = Tensor(json["layer1.0.weight"]).reshape(32, 1*3*3).permute(1, 0); // [9, 32], ConvBlock kernel이 사용하는 행렬 레이아웃으로 변경
        mnistNet["conv0.bias"] = Tensor(json["layer1.0.bias"]);
        mnistNet["conv1.weight"] = Tensor(json["layer2.0.weight"]).reshape(64, 32*3*3).permute(1, 0);
        mnistNet["conv1.bias"] = Tensor(json["layer2.0.bias"]);
        mnistNet["weight"] = Tensor(json["fc.weight"]).reshape(10, 64, 7*7).permute(2, 1, 0).reshape(7*7*64, 10);
        mnistNet["bias"] = Tensor(json["fc.bias"]);
        
        Tensor result;
        Tensor inputTensor = Tensor(28, 28, 1).set(srcImage); // CPU -> GPU 메모리 복사

        for (uint32_t i = 0; i < iter; ++i)
            result = mnistNet(inputTensor)[0];

        return result;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error in eval_mnist(): " << e.what() << std::endl;
        throw;
    }
    catch (...)
    {
        std::cerr << "Unknown error in eval_mnist()" << std::endl;
        throw;
    }
}

void test()
{
    try
    {
        // Load Vulkan compute shader pipelines
        void loadShaders();
        loadShaders();

        // Load image and normalize
        const uint32_t channels = 1;
        auto [srcImage, width, height] = readImage<channels>(PROJECT_ROOT_DIR"/utils/0.png");
        
        _ASSERT(width == 28 && height == 28);
        _ASSERT(width * height * channels == srcImage.size());

        std::vector<float> inputData(width * height * channels);
        for (size_t i = 0; i < srcImage.size(); ++i)
            inputData[i] = srcImage[i] / 255.0f;

        // Load model weights from json
		JsonParser json = JsonParser(PROJECT_CURRENT_DIR"/utils/mnist_weights.json");

        uint32_t iter = 1;
        Tensor eval;

        TimeChecker timer("(VAI) MNIST evaluation: {} iterations", iter);
        eval = eval_mnist(inputData, json, iter);

        // GPU 결과(device tensor) → host-visible 버퍼 → CPU 배열로 복사하는 전체 플로우
        vk::Buffer outBuffer = netGlobalDevice.createBuffer({
            10 * sizeof(float),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, // CPU에서 읽을 수 있도록 설정
        });

        vk::Buffer evalBuffer = eval.buffer();
        netGlobalDevice.newCommandBuffer(queue_compute)
            .begin()
            .copyBuffer(outBuffer, evalBuffer)
            .end()
            .submit()
            .wait(); // GPU 메모리에 있는 eval 결과를 CPU에서 읽을 수 있는 outBuffer로 복사하고, 끝날 때까지 기다림 

        float data[1000];
        memcpy(data, outBuffer.map(), 1000 * sizeof(float)); // outBuffer의 메모리를 CPU 주소 공간에 매핑하고 복사

        for (int i=0; i<1000; ++i)
            printf("data[%d] = %f\n", i, data[i]);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error in test(): " << e.what() << std::endl;
        throw;  // Re-throw to be caught by main
    }
    catch (...)
    {
        std::cerr << "Unknown error in test()" << std::endl;
        throw;  // Re-throw to be caught by main
    }
}

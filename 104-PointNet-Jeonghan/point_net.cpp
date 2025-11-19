#include "neuralNet.h"
#include "neuralNodes.h"
#include "jsonParser.h"
#include "timeChecker.hpp"
#include <stb/stb_image.h>
#include <cstring>  // memcpy



template<uint32_t nBlocks>
class FCSequence : public NodeGroup
{
    uint32_t dims[nBlocks + 1];
    std::unique_ptr<FullyConnectedNode> blocks[nBlocks];

public:
    FCSequence(const uint32_t(&channels)[nBlocks + 1])
    {
        for (uint32_t i = 0; i <= nBlocks; ++i)
            dims[i] = channels[i];

        for (uint32_t i = 0; i < nBlocks; ++i)
            blocks[i] = std::make_unique<FullyConnectedNode>(dims[i], dims[i + 1]);

        for (uint32_t i = 0; i < nBlocks - 1; ++i)
            *blocks[i] - *blocks[i + 1];

        defineSlot("in0", blocks[0]->slot("in0"));
        defineSlot("out0", blocks[nBlocks - 1]->slot("out0"));
    }

    Tensor& operator[](const std::string& name)
    {
        for (uint32_t i = 0; i < nBlocks; ++i)
        {
            const std::string prefix = "fc" + std::to_string(i) + ".";
            if (name.starts_with(prefix))
                return (*blocks[i])[name.substr(prefix.length())];
        }
        throw std::runtime_error("No such layer in FCSequence: " + name);
    }
};
template<std::size_t N>
FCSequence(const uint32_t (&)[N]) -> FCSequence<N - 1>;


template<uint32_t nBlocks>
class MLPSequence : public NodeGroup
{
    uint32_t dims[nBlocks + 1];
    std::unique_ptr<PointWiseMLPNode> blocks[nBlocks];

public:
    MLPSequence(const uint32_t(&channels)[nBlocks + 1])
    {
        for (uint32_t i = 0; i <= nBlocks; ++i)
            dims[i] = channels[i];

        for (uint32_t i = 0; i < nBlocks; ++i)
            blocks[i] = std::make_unique<PointWiseMLPNode>(dims[i], dims[i + 1]);

        for (uint32_t i = 0; i < nBlocks - 1; ++i)
            *blocks[i] - *blocks[i + 1];

        defineSlot("in0", blocks[0]->slot("in0"));
        defineSlot("out0", blocks[nBlocks - 1]->slot("out0"));
    }

    Tensor& operator[](const std::string& name)
    {
        for (uint32_t i = 0; i < nBlocks; ++i)
        {
            const std::string prefix = "mlp" + std::to_string(i) + ".";
            if (name.starts_with(prefix))
                return (*blocks[i])[name.substr(prefix.length())];
        }
        throw std::runtime_error("No such layer in MLPSequence: " + name);
    }
};
template<std::size_t N>
MLPSequence(const uint32_t (&)[N]) -> MLPSequence<N - 1>;

class TNetBlock : public NodeGroup
{
    uint32_t K;

    MLPSequence<3> mlp;     // K -> 64 -> 128 -> 1024
    MaxPooling1DNode maxpool;
    FCSequence<4> fc;       // 1024 -> 512 -> 256 -> K*K

public:
    TNetBlock(uint32_t inputDim)
    : K(inputDim)
    , mlp({K, 64, 128, 1024})
    , maxpool()
    , fc({1024, 512, 256, K*K})
    {
        mlp - maxpool - fc;
        defineSlot("in0", mlp.slot("in0"));
        defineSlot("out0", fc.slot("out0"));
    }

    Tensor& operator[](const std::string& name)
    {
        if (name.starts_with("mlp."))
            return mlp[name]; // mlp0.weight, mlp0.bias, mlp1.weight, mlp1.bias, mlp2.weight, mlp2.bias
        if (name.starts_with("fc."))
            return fc[name]; // fc0.weight, fc0.bias, fc1.weight, fc1.bias, fc2.weight, fc2.bias

        throw std::runtime_error("No such layer in TNetBlock: " + name);
    }
};


class PointNet : public NodeGroup
{
    TNetBlock tnet1;           // input transform (3x3)
    MLPSequence<2> mlp1;       // (3 → 64 → 64)
    TNetBlock tnet2;           // feature transform (64x64)
    MLPSequence<2> mlp2;       // (64 → 128 → 1024)

public:
    PointNet(uint32_t numClasses)
    : tnet1(3)
    , mlp1({3, 64, 64})
    , tnet2(64)
    , mlp2({64, 128, 1024})

    {
        tnet1 - mlp1 - tnet2 - mlp2;
        defineSlot("in0", tnet1.slot("in0"));
        defineSlot("out0", mlp2.slot("out0"));
    }

    Tensor& operator[](const std::string& name)
    {
        if (name.starts_with("tnet1.")) return tnet1[name];
        if (name.starts_with("mlp1."))  return mlp1[name];
        if (name.starts_with("tnet2.")) return tnet2[name];
        if (name.starts_with("mlp2."))  return mlp2[name];

        throw std::runtime_error("Unknown parameter: " + name);
    }
};


class PointNetNet : public NeuralNet
{
    PointNet pointNet;
    MaxPooling1DNode maxpool;  // (N → 1)
    FCSequence<3> fc;          // 1024 → 512 → 256 → numClasses

    uint32_t numClasses;
public:
    PointNetNet(Device& device, uint32_t numClasses)
    : NeuralNet(device, 1, 1)
    , pointNet(numClasses)
    , maxpool()  
    , fc({1024, 512, 256, numClasses})
    {
        input(0) - pointNet - maxpool - fc - output(0);
    }

    Tensor& operator[](const std::string& name)
    {
        if (name.starts_with("pointNet.")) return pointNet[name];
        if (name.starts_with("maxpool.")) return maxpool[name];
        if (name.starts_with("fc.")) return fc[name];
        throw std::runtime_error("Unknown parameter: " + name);
    }
};

Tensor eval_mnist(const std::vector<float>& srcImage, const JsonParser& json, uint32_t iter) // srcImage layout: [H][W][C]
{
    PointNetNet pointNetNet(netGlobalDevice, 10);

    pointNetNet["tnet1.weight"] = Tensor(json["layer1.0.weight"]).reshape(32, 1*3*3).permute(1, 0);
    pointNetNet["tnet1.bias"] = Tensor(json["layer1.0.bias"]);
    pointNetNet["mlp1.weight"] = Tensor(json["layer2.0.weight"]).reshape(64, 32*3*3).permute(1, 0);
    pointNetNet["mlp1.bias"] = Tensor(json["layer2.0.bias"]);
    pointNetNet["tnet2.weight"] = Tensor(json["layer3.0.weight"]).reshape(64, 64*3*3).permute(1, 0);
    pointNetNet["tnet2.bias"] = Tensor(json["layer3.0.bias"]);
    pointNetNet["mlp2.weight"] = Tensor(json["layer4.0.weight"]).reshape(128, 64*3*3).permute(1, 0);
    pointNetNet["mlp2.bias"] = Tensor(json["layer4.0.bias"]);
    pointNetNet["fc.weight"] = Tensor(json["fc.weight"]).reshape(10, 128, 7*7).permute(2, 1, 0).reshape(7*7*128, 10);
    pointNetNet["fc.bias"] = Tensor(json["fc.bias"]);
    
    Tensor result;
    Tensor inputTensor = Tensor(28, 28, 1).set(srcImage);

    for (uint32_t i = 0; i < iter; ++i)
        result = pointNetNet(inputTensor)[0];

    return result;
}

void test()
{
    void loadShaders();
    loadShaders();

    const uint32_t channels = 1;
    std::vector<float> inputData(28*28*channels);
    for (size_t i = 0; i < inputData.size(); ++i)
        inputData[i] = 0.0f;

    JsonParser json = JsonParser(PROJECT_CURRENT_DIR"/weights.json");

    uint32_t iter = 1;  
    Tensor eval;

    {
        TimeChecker timer("(VAI) MNIST evaluation: {} iterations", iter);
        eval = eval_mnist(inputData, json, iter);
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

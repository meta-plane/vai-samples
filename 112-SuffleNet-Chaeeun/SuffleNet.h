// SuffleNet public interface and building blocks
#pragma once

#include "neuralNet.h"
#include "neuralNodes.h"
#include <memory>

// ConvBlock: Conv -> ReLU -> MaxPool
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

// ConvSequence: sequence of ConvBlocks
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

// Squeeze-and-Excitation layer (skeleton)
class SELayer : public NodeGroup
{
    uint32_t C, F, K;
    AdaptiveAvgPoolingNode aap;
    ConvolutionNode conv_1;
    BatchNormNode bn;
    ReluNode relu;
    ConvolutionNode conv_2;
    HSNode hs;
    MultiplyNode mul;

public:
    SELayer(uint32_t inChannels)
    : C(inChannels),
      aap(1), conv_1(inChannels, inChannels/4, 1),
      bn(inChannels/4), conv_2(inChannels/4, inChannels, 1)
    {
        aap - conv_1 - bn - relu - conv_2 - hs - "atten" / mul;
        defineSlot("in0", aap.slot("in0"));
        defineSlot("in1", mul.slot("in0"));
        defineSlot("out0", mul.slot("out0"));
    }

    Tensor& operator[](const std::string& name)
    {
        return conv_1[name]; // weight & bias
    }
};

// SuffleNet model
class SuffleNet : public NeuralNet
{
    ConvSequence<2> convX2;
    FlattenNode flatten;
    FullyConnectedNode fc;

public:
    SuffleNet(Device& device)
    : NeuralNet(device, 1, 1)
    , convX2({1, 32, 64}, 3)
    , fc(7 * 7 * 64, 10)
    {
        input(0) - convX2 - flatten - fc - output(0);
    }

    Tensor& operator[](const std::string& name)
    {
        if (name.starts_with("conv"))
            return convX2[name];
        else if (name == "weight" || name == "bias")
            return fc[name];
        else if (name.starts_with("fc."))
            return fc[name.substr(3)];
        else
            throw std::runtime_error("No such layer in MnistNet: " + name);
    }
};


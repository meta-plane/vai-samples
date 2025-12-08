#include "MobileNetV2.h"


static std::vector<IRBConfig> getIRBConfigs() // Inverted Residual Block configurations
{
    return { // outChannels, expansionFactor, stride
        {16, 1, 1},
        {24, 6, 2}, {24, 6, 1},
        {32, 6, 2}, {32, 6, 1}, {32, 6, 1},
        {64, 6, 2}, {64, 6, 1}, {64, 6, 1}, {64, 6, 1},
        {96, 6, 1}, {96, 6, 1}, {96, 6, 1},
        {160, 6, 2}, {160, 6, 1}, {160, 6, 1},
        {320, 6, 1}
    };
}

MobileNetV2::MobileNetV2(Device& device, uint32_t numClasses)
: NeuralNet(device, 1, 1),
  stem(3, 32, 3, 2, 1), // inChannels=3, outChannels=32, kernel=3, stride=2, padding=1
  finalConv(320, 1280),
  classifier(1280, numClasses)
{
    // Create Inverted Residual Blocks based on the configuration
    auto irbConfigs = getIRBConfigs();

    uint32_t inChannels = 32;
    for (const auto& config : irbConfigs)
    {
        invertedResidualBlocks.push_back(
            std::make_unique<InvertedResidualBlock>(
                inChannels,
                config.outChannels,
                config.expansionFactor,
                config.stride
            )
        );
        inChannels = config.outChannels;
    }

    /////////////////////////////////////////////////
    ///////// Build the computational graph ///////// 
    /////////////////////////////////////////////////

    // Connect input node to stem
    input(0).slot("out0") - stem.slot("in0");
    NodeSlot* lastOutput = &stem.slot("out0");
    
    // Connect Inverted Residual Blocks sequentially
    for (const auto& irb : invertedResidualBlocks)
    {
        *lastOutput - irb->slot("in0");
        lastOutput = &irb->slot("out0");
    }

    // Connect Head
    *lastOutput - finalConv.slot("in0");
    lastOutput = &finalConv.slot("out0");

    *lastOutput - globalAvgPool.slot("in0");
    lastOutput = &globalAvgPool.slot("out0");

    *lastOutput - flatten.slot("in0");
    lastOutput = &flatten.slot("out0");

    *lastOutput - classifier.slot("in0");

    // Connect output node
    classifier.slot("out0") - output(0).slot("in0");
}

Tensor& MobileNetV2::operator[](const std::string& name)
{
    if (name.rfind("stem.", 0) == 0)
        return stem[name.substr(5)];  // "conv.weight", "bn.mean" ...

    if (name.rfind("features.", 0) == 0) {
        size_t idxStart = 9;
        size_t dotPos = name.find('.', idxStart);
        int blockIdx = std::stoi(name.substr(idxStart, dotPos - idxStart));
        auto& block = *invertedResidualBlocks[blockIdx];
        return block[name.substr(dotPos + 1)];  // e.g. "pwConvBNReLU6.pointwiseConv.weight"
    }

    if (name.rfind("finalConv.", 0) == 0)
        return finalConv[name.substr(10)];

    if (name.rfind("classifier.", 0) == 0)
        return classifier[name.substr(11)];

    throw std::runtime_error("MobileNetV2: Tensor not found: " + name);
}
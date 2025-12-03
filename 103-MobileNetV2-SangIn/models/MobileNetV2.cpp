#include "MobileNetV2.h"

// MobileNetV2::MobileNetV2(Device& device, uint32_t numClasses)
// : NeuralNet(device)
// {
//     // 1) Initial Conv-BN layer
//     initialConv = std::make_unique<ConvBN>(3, 32, 3, 2, 1); // inChannels=3, outChannels=32, kernel=3, stride=2, padding=1
//     *input(0) - *initialConv;

//     // 2) Inverted Residual Blocks configuration
//     struct BlockConfig {
//         uint32_t inChannels;
//         uint32_t outChannels;
//         uint32_t expansionFactor;
//         uint32_t stride;
//         uint32_t numBlocks;
//     };

//     std::vector<BlockConfig> blockConfigs = {
//         {32, 16, 1, 1, 1},
//         {16, 24, 6, 2, 2},
//         {24, 32, 6, 2, 3},
//         {32, 64, 6, 2, 4},
//         {64, 96, 6, 1, 3},
//         {96, 160, 6, 2, 3},
//         {160, 320, 6, 1, 1},
//     };

//     // 3) Create Inverted Residual Blocks
//     for (const auto& config : blockConfigs)
//     {
//         for (uint32_t i = 0; i < config.numBlocks; ++i)
//         {
//             uint32_t stride = (i == 0) ? config.stride : 1;
//             auto block = std::make_unique<InvertedResidualBlock>(
//                 config.inChannels,
//                 config.outChannels,
//                 config.expansionFactor,
//                 stride
//             );
//             *initialConv->slot("out0") - *block;
//             initialConv = std::move(block);
//         }
//     }

//     // 4) Global Average Pooling
//     globalAvgPool = std::make_unique<GlobalAvgPoolNode>();
//     *initialConv->slot("out0") - *globalAvgPool;

//     // 5) Fully Connected layer for classification
//     fc = std::make_unique<FullyConnectedNode>(320, numClasses);
//     *globalAvgPool->slot("out0") - *fc - *output(0);
// }
#ifndef MOBILENET_MOBILENET_V2_H
#define MOBILENET_MOBILENET_V2_H

#include "core/neuralNet.h"

// TODO: Phase 7 - MobileNetV2 implementation
// MobileNetV2 should inherit from NeuralNet and implement the architecture
class MobileNetV2 : public NeuralNet {
public:
    // TODO: Phase 7 - Implement constructor with Device parameter
    // MobileNetV2(vk::Device& device) : NeuralNet(device) {}
    
    void initialize();
    void loadWeights(const std::string& weightsPath);
};

#endif // MOBILENET_MOBILENET_V2_H


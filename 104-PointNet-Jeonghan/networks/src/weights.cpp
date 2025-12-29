#include "weights.h"
#include "weightLoader.hpp"
#include <iostream>

namespace networks {

void loadPointNetWeights(PointNetSegment& model, const std::string& weights_file) {
    // Only SafeTensors format supported
    if (weights_file.size() < 12 ||
        weights_file.substr(weights_file.size() - 12) != ".safetensors") {
        throw std::runtime_error("Only SafeTensors format supported (.safetensors)");
    }

    std::cout << "Loading weights from SafeTensors: " << weights_file << std::endl;

    SafeTensorsParser weights(weights_file.c_str());
    WeightLoader loader(weights);
    loader.loadSegment(model);

    std::cout << "✓ All weights loaded from SafeTensors" << std::endl;
}

} // namespace networks

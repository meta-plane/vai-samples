#ifndef POINTNET_WEIGHTS_H
#define POINTNET_WEIGHTS_H

#include "pointnet.hpp"
#include "safeTensorsParser.h"
#include <string>

/**
 * PointNet Weight Loading
 *
 * Loads pretrained weights from SafeTensors files using WeightLoader.
 * Uses PyTorch state_dict keys directly (no key transformation needed).
 */

namespace networks {

/**
 * Load PointNet weights from SafeTensors file
 *
 * @param model PointNetSegment network
 * @param weights_file Path to SafeTensors file (.safetensors)
 */
void loadPointNetWeights(PointNetSegment& model, const std::string& weights_file);

} // namespace networks

#endif // POINTNET_WEIGHTS_H

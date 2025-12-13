#ifndef POINTNET_WEIGHTS_H
#define POINTNET_WEIGHTS_H

#include "pointnet.hpp"
#include "jsonParser.h"
#include "safeTensorsParser.h"
#include <string>

/**
 * PointNet Weight Loading
 *
 * Handles loading pretrained weights from JSON or SafeTensors files.
 * Separated from inference for cleaner organization.
 */

namespace networks {

/**
 * Load PointNet weights from JSON or SafeTensors file
 * Automatically detects format based on file extension.
 *
 * @param model PointNetSegment network
 * @param weights_file Path to weights file (.json or .safetensors)
 */
void loadPointNetWeights(PointNetSegment& model, const std::string& weights_file);

/**
 * Load PointNet weights from SafeTensors file
 *
 * @param model PointNetSegment network
 * @param weights_file Path to SafeTensors file
 */
void loadPointNetWeightsFromSafeTensors(PointNetSegment& model, const std::string& weights_file);

/**
 * Load PointNet weights from JSON file (legacy)
 *
 * @param model PointNetSegment network
 * @param weights_file Path to JSON file
 */
void loadPointNetWeightsFromJSON(PointNetSegment& model, const std::string& weights_file);

} // namespace networks

#endif // POINTNET_WEIGHTS_H


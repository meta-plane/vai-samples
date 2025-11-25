#ifndef POINTNET_WEIGHTS_H
#define POINTNET_WEIGHTS_H

#include "pointnet.hpp"
#include "jsonParser.h"
#include <string>

/**
 * PointNet Weight Loading
 *
 * Handles loading pretrained weights from JSON files.
 * Separated from inference for cleaner organization.
 */

namespace networks {

/**
 * Load PointNet weights from JSON file
 *
 * @param model PointNetSegment network
 * @param weights_file Path to JSON weights file
 */
void loadPointNetWeights(PointNetSegment& model, const std::string& weights_file);

} // namespace networks

#endif // POINTNET_WEIGHTS_H


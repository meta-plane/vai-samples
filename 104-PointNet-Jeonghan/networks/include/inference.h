#ifndef POINTNET_INFERENCE_H
#define POINTNET_INFERENCE_H

#include "pointnet.hpp"
#include <string>
#include <vector>

/**
 * PointNet Inference Module
 *
 * Handles loading pretrained weights and running point cloud segmentation.
 * Clean API inspired by GPT-2 inference module.
 */

namespace networks {

/**
 * Inference Configuration
 */
struct InferenceConfig {
    // Paths (relative to 104-PointNet-Jeonghan directory)
    std::string weights_file = "assets/weights/pointnet_weights.json";
    
    // Model parameters
    uint32_t num_classes = 10;
    uint32_t channel = 9;  // Input channels (3 for xyz, 9 for xyz+rgb+normalized)
    
    // Inference settings
    uint32_t num_iterations = 1;  // For benchmarking
    bool verbose = true;
};

/**
 * Segmentation Result
 */
struct SegmentationResult {
    std::vector<std::vector<float>> predictions;  // [N, numClasses]
    std::vector<uint32_t> predicted_labels;       // [N] - argmax of predictions
    
    uint32_t num_points;
    uint32_t num_classes;
    
    double inference_time_sec;
    double points_per_sec;
    
    bool success = false;
    std::string error_message;
};

/**
 * Load pretrained PointNet model
 *
 * @param config Inference configuration
 * @return Initialized PointNetSegment with loaded weights, or nullptr on failure
 */
PointNetSegment* loadPretrainedModel(const InferenceConfig& config = InferenceConfig());

/**
 * Segment point cloud
 *
 * @param model PointNetSegment with loaded weights
 * @param point_cloud Input point cloud [N, 3]
 * @param config Inference configuration
 * @return Segmentation result with predictions, labels, and timing
 */
SegmentationResult segment(
    PointNetSegment& model,
    const std::vector<float>& point_cloud,
    const InferenceConfig& config = InferenceConfig()
);

/**
 * Segment point cloud from file
 *
 * @param model PointNetSegment with loaded weights
 * @param filename Path to point cloud file (.txt, .ply, etc.)
 * @param config Inference configuration
 * @return Segmentation result
 */
SegmentationResult segmentFromFile(
    PointNetSegment& model,
    const std::string& filename,
    const InferenceConfig& config = InferenceConfig()
);

/**
 * Load point cloud from text file
 * Format: Each line contains x, y, z coordinates
 *
 * @param filename Path to file
 * @return Point cloud data [N*3] (flattened)
 */
std::vector<float> loadPointCloudFromFile(const std::string& filename);

} // namespace networks

#endif // POINTNET_INFERENCE_H


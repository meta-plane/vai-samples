#include <iostream>
#include <iomanip>
#include "inference.h"

using namespace networks;

/**
 * PointNet Segmentation - Main Entry Point
 *
 * Clean API inspired by GPT-2 inference module.
 */

void printUsage() {
    std::cout << "\nPointNet Segmentation\n";
    std::cout << "=====================\n\n";
    std::cout << "Usage:\n";
    std::cout << "  1. Place weights at: assets/weights/pointnet_weights.json\n";
    std::cout << "  2. Place point cloud at: assets/data/sample.txt\n";
    std::cout << "  3. Run this executable\n\n";
    std::cout << "Point cloud format:\n";
    std::cout << "  Each line: x y z (space-separated)\n";
    std::cout << "  Example:\n";
    std::cout << "    0.5 0.3 0.1\n";
    std::cout << "    0.6 0.4 0.2\n";
    std::cout << "    ...\n\n";
}

int main()
{
    std::cout << "╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║      PointNet Segmentation - Vulkan Inference         ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n\n";

    // Configuration
    InferenceConfig config;
    config.weights_file = "assets/weights/pointnet_weights.json";
    config.num_classes = 10;
    config.verbose = true;

    // Load pretrained model
    std::cout << "Step 1: Loading pretrained model...\n";
    PointNetSegment* model = loadPretrainedModel(config);
    
    if (!model) {
        printUsage();
        return 1;
    }

    // Example 1: Segment from generated data
    std::cout << "\n" << std::string(56, '-') << "\n";
    std::cout << "Example: Segmenting random point cloud\n";
    std::cout << std::string(56, '-') << "\n\n";
    
    const uint32_t num_points = 1024;
    std::vector<float> point_cloud(num_points * 3);
    
    // Generate random points
    for (size_t i = 0; i < point_cloud.size(); ++i) {
        point_cloud[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }
    
    SegmentationResult result = segment(*model, point_cloud, config);
    
    if (result.success) {
        std::cout << "\nSample predictions (first 5 points):\n";
        for (uint32_t i = 0; i < std::min(5u, result.num_points); ++i) {
            std::cout << "Point " << i << " → Class " << result.predicted_labels[i];
            std::cout << " (scores: ";
            for (uint32_t c = 0; c < std::min(3u, result.num_classes); ++c) {
                std::cout << std::fixed << std::setprecision(3) 
                          << result.predictions[i][c];
                if (c < std::min(3u, result.num_classes) - 1) std::cout << ", ";
            }
            std::cout << "...)\n";
        }
    }

    // Example 2: Segment from file (if exists)
    std::cout << "\n" << std::string(56, '-') << "\n";
    std::cout << "Attempting to load from file...\n";
    std::cout << std::string(56, '-') << "\n\n";
    
    SegmentationResult file_result = segmentFromFile(
        *model, 
        "assets/data/sample.txt", 
        config
    );
    
    if (file_result.success) {
        std::cout << "\nFile segmentation successful!\n";
        std::cout << "Performance: " << std::fixed << std::setprecision(1)
                  << file_result.points_per_sec << " points/sec\n";
    }

    // Cleanup
    delete model;

    std::cout << "\n" << std::string(56, '=') << "\n";
    std::cout << "PointNet segmentation demo complete!\n";
    std::cout << std::string(56, '=') << "\n\n";

    return 0;
}


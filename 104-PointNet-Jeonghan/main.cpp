#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include "inference.h"
#include "pointCloudLoader.h"

using namespace networks;
using namespace pointcloud;

/**
 * PointNet Segmentation - Main Entry Point
 *
 * Clean API inspired by GPT-2 inference module.
 */

void printUsage() {
    std::cout << "\nPointNet Segmentation (yanx27)\n";
    std::cout << "==============================\\n";
    std::cout << "Model: S3DIS Semantic Segmentation (13 classes)\\n";
    std::cout << "Input: 3D xyz coordinates\\n\\n";
    std::cout << "Usage:\\n";
    std::cout << "  1. Place weights at: assets/weights/pointnet_yanx27.json\\n";
    std::cout << "  2. Place point cloud at: assets/data/sample.txt\\n";
    std::cout << "  3. Run this executable\\n\\n";
    std::cout << "Point cloud format:\\n";
    std::cout << "  Each line: x y z (space-separated)\\n";
    std::cout << "  Example:\\n";
    std::cout << "    0.5 0.3 0.1\\n";
    std::cout << "    0.6 0.4 0.2\\n";
    std::cout << "    ...\\n\\n";
}

int main()
{
    std::cout << "╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║      PointNet Segmentation - Vulkan Inference         ║\n";
    std::cout << "║           with ModelNet40 Dataset Support             ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n\n";

    // Initialize random seed
    srand(static_cast<unsigned int>(time(nullptr)));

    // Configuration (yanx27: 3-dim xyz input, 13 semantic classes)
    // Note: yanx27 weights are for S3DIS semantic segmentation (3D xyz input)
    InferenceConfig config;
    config.weights_file = "assets/weights/pointnet_yanx27.json";  // yanx27 pretrained weights
    config.num_classes = 13;  // S3DIS semantic segmentation
    config.channel = 3;  // xyz coordinates only (yanx27 format)
    config.verbose = true;

    // Load and segment from ModelNet40
    const std::string modelnet_path = "assets/datasets/ModelNet40";
    
    if (std::filesystem::exists(modelnet_path)) {
        std::cout << "\nLoading ModelNet40 sample...\n";
        
        auto [class_name, class_idx, point_cloud] = 
            modelnet40::loadSample(modelnet_path, 1024, "test");
        
        if (!point_cloud.empty()) {
            // Load model
            std::cout << "\nLoading pretrained model...\n";
            PointNetSegment* model = loadPretrainedModel(config);
            
            if (!model) {
                printUsage();
                return 1;
            }
            
            // Run segmentation (note: yanx27 is semantic seg, not classification)
            std::cout << "\nRunning inference...\n";
            SegmentationResult result = segment(*model, point_cloud, config);
            
            if (result.success) {
                std::cout << "\n" << std::string(56, '=') << "\n";
                std::cout << "Segmentation Result\n";
                std::cout << std::string(56, '=') << "\n";
                std::cout << "Object: " << class_name << " (ModelNet40 class " << class_idx << ")\n";
                std::cout << "Points: " << result.num_points << "\n";
                
                // Count predicted semantic classes
                std::vector<int> class_counts(config.num_classes, 0);
                for (uint32_t j = 0; j < result.num_points; ++j) {
                    class_counts[result.predicted_labels[j]]++;
                }
                
                // Show top 3 predicted semantic classes
                std::cout << "\nTop semantic classes detected:\n";
                std::vector<std::pair<int, int>> counts;
                for (int c = 0; c < config.num_classes; ++c) {
                    counts.push_back({class_counts[c], c});
                }
                std::sort(counts.rbegin(), counts.rend());
                
                for (int k = 0; k < 3 && k < config.num_classes; ++k) {
                    float percentage = 100.0f * counts[k].first / result.num_points;
                    std::cout << "  Class " << counts[k].second << ": " 
                              << std::fixed << std::setprecision(1) << percentage 
                              << "% (" << counts[k].first << " points)\n";
                }
                
                std::cout << "\nPerformance: " << std::fixed << std::setprecision(0)
                          << result.points_per_sec << " points/sec\n";
                std::cout << std::string(56, '=') << "\n";
            }
            
            // Cleanup
            delete model;
        } else {
            std::cout << "Failed to load ModelNet40 sample\n";
        }
    } else {
        std::cout << "ModelNet40 not found at: " << modelnet_path << "\n";
        std::cout << "Please download ModelNet40 dataset.\n";
    }

    std::cout << "\nPointNet segmentation complete!\n";

    return 0;
}


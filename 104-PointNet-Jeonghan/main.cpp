#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <algorithm>
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
    std::cout << "\nPointNet Segmentation\n";
    std::cout << "==============================\\n";
    std::cout << "Model: S3DIS Semantic Segmentation (13 classes)\\n";
    std::cout << "Input: 3D xyz coordinates\\n\\n";
    std::cout << "Usage:\\n";
    std::cout << "  1. Place weights at: assets/weights/pointnet_sem_seg.safetensors\\n";
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

    InferenceConfig config;
    config.weights_file = "assets/weights/pointnet_sem_seg.safetensors";
    config.num_classes = 13;  // S3DIS semantic segmentation
    config.channel = 9;       // yanx27 S3DIS model uses 9 channels (xyz + rgb + normalized_xyz)
    config.verbose = true;

    // Load and segment from ModelNet40
    const std::string modelnet_path = "assets/datasets/ModelNet40";
    
    if (std::filesystem::exists(modelnet_path)) {

        std::cout << "\nLoading pretrained model...\n";
        PointNetSegment* model = loadPretrainedModel(config);
        
        if (!model) {
            printUsage();
            return 1;
        }
        
        // Run segmentation benchmark (10 iterations)
        std::cout << "\nRunning inference benchmark (10 iterations)...\n";
        
        const int num_iterations = 10;
        std::vector<double> iteration_times;
        SegmentationResult result;
        
        auto [class_name, class_idx, point_cloud] = 
            modelnet40::loadSample(modelnet_path, 1024, "test");
        for (int iter = 0; iter < num_iterations; ++iter) {

            if (point_cloud.empty()) {
                std::cout << "Failed to load ModelNet40 sample\n";
                break;
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            result = segment(*model, point_cloud, config);
            auto end = std::chrono::high_resolution_clock::now();
            
            double elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();
            iteration_times.push_back(elapsed_us);
            
            std::cout << "  Iteration " << (iter + 1) << ": " 
                        << "(" << std::fixed << std::setprecision(3) << elapsed_us << " μs)\n";
        }
        
        // Calculate statistics
        double total_time = 0.0;
        for (double t : iteration_times) total_time += t;
        double avg_time = total_time / num_iterations;
        
        double min_time = *std::min_element(iteration_times.begin(), iteration_times.end());
        double max_time = *std::max_element(iteration_times.begin(), iteration_times.end());
        
        if (result.success) {
            std::cout << "\n" << std::string(56, '=') << "\n";
            std::cout << "Benchmark Results\n";
            std::cout << std::string(56, '=') << "\n";
            std::cout << "Average time: " << std::fixed << std::setprecision(2) << avg_time/1000.0 << " ms\n";
            std::cout << "Min time:     " << std::fixed << std::setprecision(2) << min_time/1000.0 << " ms\n";
            std::cout << "Max time:     " << std::fixed << std::setprecision(2) << max_time/1000.0 << " ms\n";
            std::cout << "Throughput:   " << std::fixed << std::setprecision(0) 
                        << (result.num_points / (avg_time / 1000000.0)) << " points/sec\n";
            
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
            std::cout << std::string(56, '=') << "\n";
        }
        else {
            std::cout << "Segmentation failed: " << result.error_message << "\n";
        }
        
        // Cleanup
        delete model;
    } else {
        std::cout << "ModelNet40 not found at: " << modelnet_path << "\n";
        std::cout << "Please download ModelNet40 dataset.\n";
    }

    std::cout << "\nPointNet segmentation complete!\n";

    return 0;
}


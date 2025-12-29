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

int main()
{
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║     PointNet Segmentation - Performance Benchmark     ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n\n";

    srand(static_cast<unsigned int>(time(nullptr)));

    InferenceConfig config;
    config.weights_file = "assets/weights/pointnet_sem_seg.safetensors";
    config.num_classes = 13;
    config.channel = 9;
    config.verbose = false;

    const std::string modelnet_path = "assets/datasets/ModelNet40";

    if (!std::filesystem::exists(modelnet_path)) {
        std::cout << "ModelNet40 not found at: " << modelnet_path << "\n";
        return 1;
    }

    // Load model
    std::cout << "Loading model...\n";
    PointNetSegment* model = loadPretrainedModel(config);
    if (!model) return 1;
    std::cout << "  Weights: " << config.weights_file << "\n";
    std::cout << "  Classes: " << config.num_classes << "\n\n";

    // Load sample
    auto [class_name, class_idx, point_cloud] =
        modelnet40::loadSample(modelnet_path, 1024, "test");

    if (point_cloud.empty()) {
        std::cout << "Failed to load sample\n";
        return 1;
    }

    uint32_t num_points = point_cloud.size() / 3;
    std::cout << "Input:\n";
    std::cout << "  Sample:  " << class_name << " (" << num_points << " points)\n";
    std::cout << "  Shape:   [" << config.channel << ", " << num_points << "]\n\n";

    // Warmup (3 iterations) - exclude initialization costs
    std::cout << "Warmup (3 iterations)...\n";
    for (int i = 0; i < 3; ++i) {
        segment(*model, point_cloud, config);
    }
    std::cout << "  Done\n\n";

    // Benchmark
    const int num_iterations = 10;
    std::vector<double> times;
    SegmentationResult result;

    std::cout << "Running " << num_iterations << " iterations...\n";

    for (int i = 0; i < num_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        result = segment(*model, point_cloud, config);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(elapsed_ms);

        std::cout << "  [" << std::setw(2) << (i + 1) << "] "
                  << std::fixed << std::setprecision(2) << std::setw(8) << elapsed_ms << " ms\n";
    }

    if (!result.success) {
        std::cout << "Error: " << result.error_message << "\n";
        return 1;
    }

    // Statistics
    double total = 0.0;
    for (double t : times) total += t;
    double avg = total / num_iterations;
    double min_t = *std::min_element(times.begin(), times.end());
    double max_t = *std::max_element(times.begin(), times.end());

    // Class distribution
    std::vector<int> class_counts(config.num_classes, 0);
    for (uint32_t j = 0; j < result.num_points; ++j) {
        class_counts[result.predicted_labels[j]]++;
    }

    std::vector<std::pair<int, int>> sorted_counts;
    for (int c = 0; c < config.num_classes; ++c) {
        sorted_counts.push_back({class_counts[c], c});
    }
    std::sort(sorted_counts.rbegin(), sorted_counts.rend());

    // Results
    double throughput = num_points / (avg / 1000.0);
    std::cout << "\nResults:\n";
    std::cout << "  ┌──────────────────┬──────────────────────┐\n";
    std::cout << "  │ Metric           │ Value                │\n";
    std::cout << "  ├──────────────────┼──────────────────────┤\n";
    std::cout << "  │ Average time     │ " << std::setw(16) << std::fixed << std::setprecision(2) << avg << " ms │\n";
    std::cout << "  │ Min time         │ " << std::setw(16) << min_t << " ms │\n";
    std::cout << "  │ Max time         │ " << std::setw(16) << max_t << " ms │\n";
    std::cout << "  │ Throughput       │ " << std::setw(13) << std::setprecision(0) << throughput << " pts/s │\n";
    std::cout << "  └──────────────────┴──────────────────────┘\n";

    std::cout << "\nPrediction Distribution (top 3):\n";
    std::cout << "  ┌─────────┬──────────┬────────────┐\n";
    std::cout << "  │ Class   │ Points   │ Percentage │\n";
    std::cout << "  ├─────────┼──────────┼────────────┤\n";

    for (int k = 0; k < 3 && k < config.num_classes; ++k) {
        float pct = 100.0f * sorted_counts[k].first / result.num_points;
        std::cout << "  │ " << std::setw(7) << sorted_counts[k].second << " │ "
                  << std::setw(8) << sorted_counts[k].first << " │ "
                  << std::setw(9) << std::fixed << std::setprecision(1) << pct << "% │\n";
    }
    std::cout << "  └─────────┴──────────┴────────────┘\n";

    delete model;
    std::cout << "\n";

    return 0;
}


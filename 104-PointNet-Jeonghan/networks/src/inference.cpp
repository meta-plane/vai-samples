#include "inference.h"
#include "weights.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace vk;

namespace networks {

PointNetSegment* loadPretrainedModel(const InferenceConfig& config) {
    try {
        // Check if weights exist
        std::ifstream test_file(config.weights_file);
        if (!test_file) {
            std::cout << "⚠ Pretrained weights not found at: " << config.weights_file << std::endl;
            std::cout << "  Please provide PointNet weights in JSON format" << std::endl;
            return nullptr;
        }
        test_file.close();

        if (config.verbose) {
            std::cout << "Loading PointNet Segmentation model..." << std::endl;
            std::cout << "  Weights: " << config.weights_file << std::endl;
            std::cout << "  Num classes: " << config.num_classes << std::endl;
        }

        // Create network
        PointNetSegment* model = new PointNetSegment(netGlobalDevice, config.num_classes);
        
        if (config.verbose) {
            std::cout << "✓ Network created" << std::endl;
        }

        // Load weights
        if (config.verbose) {
            std::cout << "Loading pretrained weights..." << std::endl;
        }
        
        loadPointNetWeights(*model, config.weights_file);
        
        if (config.verbose) {
            std::cout << "✓ Weights loaded\n" << std::endl;
        }

        return model;

    } catch (const std::exception& e) {
        std::cout << "✗ Error loading pretrained model: " << e.what() << std::endl;
        return nullptr;
    }
}

SegmentationResult segment(
    PointNetSegment& model,
    const std::vector<float>& point_cloud,
    const InferenceConfig& config
) {
    SegmentationResult result;
    
    try {
        // Validate input
        if (point_cloud.empty() || point_cloud.size() % 3 != 0) {
            result.error_message = "Invalid point cloud: size must be multiple of 3";
            return result;
        }
        
        uint32_t num_points = point_cloud.size() / 3;
        result.num_points = num_points;
        result.num_classes = config.num_classes;
        
        // Create input tensor [N, 3]
        Tensor input_tensor = Tensor(num_points, 3).set(point_cloud);
        
        // Run inference with timing
        auto start = std::chrono::high_resolution_clock::now();
        
        std::cout << "DEBUG: About to run model inference..." << std::endl;
        std::cout << "DEBUG: Input tensor shape = [" << input_tensor.shape()[0] << ", " << input_tensor.shape()[1] << "]" << std::endl;
        
        Tensor output;
        for (uint32_t i = 0; i < config.num_iterations; ++i) {
            std::cout << "DEBUG: Calling model (iteration " << i << ")..." << std::endl;
            auto outputs = model(input_tensor);
            std::cout << "DEBUG: Model call complete, outputs size = " << outputs.size() << std::endl;
            output = outputs[0];
            std::cout << "DEBUG: Output tensor shape = [" << output.shape()[0] << ", " << output.shape()[1] << "]" << std::endl;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        
        result.inference_time_sec = elapsed.count() / config.num_iterations;
        result.points_per_sec = num_points / result.inference_time_sec;
        
        // Copy results back to CPU
        vk::Buffer cpu_buffer = netGlobalDevice.createBuffer({
            num_points * config.num_classes * sizeof(float),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        });
        
        netGlobalDevice.newCommandBuffer(queue_compute)
            .begin()
            .copyBuffer(cpu_buffer, output.buffer())
            .end()
            .submit()
            .wait();
        
        float* data = (float*)cpu_buffer.map();
        
        // Parse predictions
        result.predictions.resize(num_points);
        result.predicted_labels.resize(num_points);
        
        for (uint32_t i = 0; i < num_points; ++i) {
            result.predictions[i].resize(config.num_classes);
            
            // Copy scores
            float max_score = -std::numeric_limits<float>::infinity();
            uint32_t max_idx = 0;
            
            for (uint32_t c = 0; c < config.num_classes; ++c) {
                float score = data[i * config.num_classes + c];
                result.predictions[i][c] = score;
                
                if (score > max_score) {
                    max_score = score;
                    max_idx = c;
                }
            }
            
            result.predicted_labels[i] = max_idx;
        }
        
        result.success = true;
        
        if (config.verbose) {
            std::cout << "Segmentation complete:" << std::endl;
            std::cout << "  Points: " << num_points << std::endl;
            std::cout << "  Time: " << std::fixed << std::setprecision(3) 
                      << result.inference_time_sec * 1000.0 << " ms" << std::endl;
            std::cout << "  Throughput: " << std::fixed << std::setprecision(1)
                      << result.points_per_sec << " points/sec" << std::endl;
        }
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
        std::cout << "✗ Segmentation error: " << e.what() << std::endl;
    }
    
    return result;
}

std::vector<float> loadPointCloudFromFile(const std::string& filename) {
    std::vector<float> point_cloud;
    std::ifstream file(filename);
    
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        float x, y, z;
        
        if (iss >> x >> y >> z) {
            point_cloud.push_back(x);
            point_cloud.push_back(y);
            point_cloud.push_back(z);
        }
    }
    
    if (point_cloud.empty()) {
        throw std::runtime_error("No points loaded from file");
    }
    
    return point_cloud;
}

SegmentationResult segmentFromFile(
    PointNetSegment& model,
    const std::string& filename,
    const InferenceConfig& config
) {
    SegmentationResult result;
    
    try {
        if (config.verbose) {
            std::cout << "Loading point cloud from: " << filename << std::endl;
        }
        
        std::vector<float> point_cloud = loadPointCloudFromFile(filename);
        
        if (config.verbose) {
            std::cout << "✓ Loaded " << (point_cloud.size() / 3) << " points\n" << std::endl;
        }
        
        return segment(model, point_cloud, config);
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
        std::cout << "✗ Error: " << e.what() << std::endl;
    }
    
    return result;
}

} // namespace networks


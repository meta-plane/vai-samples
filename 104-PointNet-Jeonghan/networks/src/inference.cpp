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
            std::cout << "  Input channels: " << config.channel << std::endl;
        }

        // Create network with specified channel dimension
        PointNetSegment* model = new PointNetSegment(netGlobalDevice, config.num_classes, config.channel);
        
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
        // Detect actual data format (3 for xyz, 9 for xyz+rgb+normalized)
        uint32_t data_channels = (point_cloud.size() % 9 == 0) ? 9 : 3;
        uint32_t num_points = point_cloud.size() / data_channels;
        result.num_points = num_points;
        result.num_classes = config.num_classes;
        
        // Expand 3-channel data to 9-channel if model expects 9
        std::vector<float> model_input;
        if (data_channels == 3 && config.channel == 9) {
            // Expand xyz to [xyz, rgb=1.0, normalized_xyz]
            model_input.reserve(num_points * 9);
            for (uint32_t i = 0; i < num_points; ++i) {
                float x = point_cloud[i * 3 + 0];
                float y = point_cloud[i * 3 + 1];
                float z = point_cloud[i * 3 + 2];
                
                // xyz
                model_input.push_back(x);
                model_input.push_back(y);
                model_input.push_back(z);
                
                // rgb (default to white for geometric-only data)
                model_input.push_back(1.0f);
                model_input.push_back(1.0f);
                model_input.push_back(1.0f);
                
                // normalized xyz (same as xyz since already normalized)
                model_input.push_back(x);
                model_input.push_back(y);
                model_input.push_back(z);
            }
            
            if (config.verbose) {
                std::cout << "Input: [" << num_points << ", 3] expanded to [" 
                          << num_points << ", 9] (xyz+rgb+normalized)" << std::endl;
            }
        } else {
            model_input = point_cloud;
            if (config.verbose) {
                std::cout << "Input: [" << num_points << ", " << data_channels << "] "
                          << (data_channels == 9 ? "(xyz+rgb+normalized)" : "(xyz only)") << std::endl;
            }
        }
        
        // Create input tensor
        Tensor input_tensor = Tensor(num_points, config.channel).set(model_input);
        
        // Run inference with timing
        auto start = std::chrono::high_resolution_clock::now();
        
        Tensor output;
        for (uint32_t i = 0; i < config.num_iterations; ++i) {
            auto outputs = model(input_tensor);
            output = outputs[0];
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
        std::vector<float> values;
        float val;
        
        // Read all values from line
        while (iss >> val) {
            values.push_back(val);
        }
        
        // Support both 3-dim (xyz) and 9-dim (xyz+rgb+normalized)
        if (values.size() == 3 || values.size() == 9) {
            for (float v : values) {
                point_cloud.push_back(v);
            }
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
            uint32_t channels = (point_cloud.size() % 9 == 0) ? 9 : 3;
            std::cout << "✓ Loaded " << (point_cloud.size() / channels) << " points ";
            std::cout << "(" << channels << "-dimensional)\n" << std::endl;
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


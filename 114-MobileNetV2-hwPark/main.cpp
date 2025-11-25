#include <iostream>
#include "core/error.h"
#include "core/vulkanApp.h"
#include "core/tensor.h"
#include "core/neuralNet.h"
#include "nodes/base_node.h"
#include "model/mobilenet_v2.h"
#include "dataloader/image_loader.h"
#include "utils/jsonParser.h"

int main()
{
    std::cout << "Hello, MobileNetV2-hwPark" << std::endl;
    
    // TODO: Phase 7-9 implementation needed
    // VulkanApp is a singleton, use get() instead of constructor
    auto& app = vk::VulkanApp::get();
    
    Tensor tensor(1, 3, 224, 224);
    
    // device() returns Device by value, need to store it first for reference
    auto device = app.device();
    NeuralNet net(device, 1, 1);  // numInputs=1, numOutputs=1
    
    // BaseNode is not implemented - nodes inherit directly from Node
    // BaseNode node("test");
    
    // MobileNetV2 requires Device parameter
    // MobileNetV2 model(device);
    
    // ImageLoader is in mobilenet namespace
    mobilenet::ImageLoader loader;
    
    // JsonParser requires file path
    // JsonParser parser("weights.json");
    
    std::cout << "All modules loaded successfully!" << std::endl;
    return 0;
}
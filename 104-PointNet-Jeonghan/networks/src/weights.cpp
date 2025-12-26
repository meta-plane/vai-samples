#include "weights.h"
#include <iostream>
#include <cctype>

namespace networks {

void loadPointNetWeights(PointNetSegment& model, const std::string& weights_file) {
    // Auto-detect format based on file extension
    size_t dot_pos = weights_file.rfind('.');
    if (dot_pos == std::string::npos) {
        throw std::runtime_error("Cannot determine file format (no extension)");
    }
    
    std::string ext = weights_file.substr(dot_pos + 1);
    // Convert to lowercase
    for (char& c : ext) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    
    if (ext == "safetensors") {
        std::cout << "Loading weights from SafeTensors format..." << std::endl;
        loadPointNetWeightsFromSafeTensors(model, weights_file);
    } else if (ext == "json") {
        std::cout << "Loading weights from JSON format (legacy)..." << std::endl;
        loadPointNetWeightsFromJSON(model, weights_file);
    } else {
        throw std::runtime_error("Unknown weights format: " + ext + " (expected .safetensors or .json)");
    }
}

void loadPointNetWeightsFromSafeTensors(PointNetSegment& model, const std::string& weights_file) {
    // Load SafeTensors
    SafeTensorsParser st(weights_file.c_str());
    
    // TNet1 (input transformation network) MLP layers
    // SafeTensors: tnet1.* -> Model: feat.stn.*
    model["feat.stn.mlp.mlp0.weight"] = Tensor(st["tnet1.mlp.0.weight"]);
    model["feat.stn.mlp.mlp0.bias"] = Tensor(st["tnet1.mlp.0.bias"]);
    model["feat.stn.mlp.mlp0.bn_mean"] = Tensor(st["tnet1.mlp.0.bn_mean"]);
    model["feat.stn.mlp.mlp0.bn_var"] = Tensor(st["tnet1.mlp.0.bn_var"]);
    model["feat.stn.mlp.mlp0.bn_gamma"] = Tensor(st["tnet1.mlp.0.bn_gamma"]);
    model["feat.stn.mlp.mlp0.bn_beta"] = Tensor(st["tnet1.mlp.0.bn_beta"]);
    
    model["feat.stn.mlp.mlp1.weight"] = Tensor(st["tnet1.mlp.1.weight"]);
    model["feat.stn.mlp.mlp1.bias"] = Tensor(st["tnet1.mlp.1.bias"]);
    model["feat.stn.mlp.mlp1.bn_mean"] = Tensor(st["tnet1.mlp.1.bn_mean"]);
    model["feat.stn.mlp.mlp1.bn_var"] = Tensor(st["tnet1.mlp.1.bn_var"]);
    model["feat.stn.mlp.mlp1.bn_gamma"] = Tensor(st["tnet1.mlp.1.bn_gamma"]);
    model["feat.stn.mlp.mlp1.bn_beta"] = Tensor(st["tnet1.mlp.1.bn_beta"]);
    
    model["feat.stn.mlp.mlp2.weight"] = Tensor(st["tnet1.mlp.2.weight"]);
    model["feat.stn.mlp.mlp2.bias"] = Tensor(st["tnet1.mlp.2.bias"]);
    model["feat.stn.mlp.mlp2.bn_mean"] = Tensor(st["tnet1.mlp.2.bn_mean"]);
    model["feat.stn.mlp.mlp2.bn_var"] = Tensor(st["tnet1.mlp.2.bn_var"]);
    model["feat.stn.mlp.mlp2.bn_gamma"] = Tensor(st["tnet1.mlp.2.bn_gamma"]);
    model["feat.stn.mlp.mlp2.bn_beta"] = Tensor(st["tnet1.mlp.2.bn_beta"]);
    
    // TNet1 FC layers with BatchNorm
    model["feat.stn.fc.block0.weight"] = Tensor(st["tnet1.fc.0.weight"]);
    model["feat.stn.fc.block0.bias"] = Tensor(st["tnet1.fc.0.bias"]);
    model["feat.stn.fc.block0.mean"] = Tensor(st["tnet1.fc.0.mean"]);
    model["feat.stn.fc.block0.var"] = Tensor(st["tnet1.fc.0.var"]);
    model["feat.stn.fc.block0.gamma"] = Tensor(st["tnet1.fc.0.gamma"]);
    model["feat.stn.fc.block0.beta"] = Tensor(st["tnet1.fc.0.beta"]);
    
    model["feat.stn.fc.block1.weight"] = Tensor(st["tnet1.fc.1.weight"]);
    model["feat.stn.fc.block1.bias"] = Tensor(st["tnet1.fc.1.bias"]);
    model["feat.stn.fc.block1.mean"] = Tensor(st["tnet1.fc.1.mean"]);
    model["feat.stn.fc.block1.var"] = Tensor(st["tnet1.fc.1.var"]);
    model["feat.stn.fc.block1.gamma"] = Tensor(st["tnet1.fc.1.gamma"]);
    model["feat.stn.fc.block1.beta"] = Tensor(st["tnet1.fc.1.beta"]);
    
    model["feat.stn.fc.lastBlock.weight"] = Tensor(st["tnet1.fc.2.weight"]);
    model["feat.stn.fc.lastBlock.bias"] = Tensor(st["tnet1.fc.2.bias"]);
    
    // MLP1 (first feature extraction)
    // SafeTensors: mlp1.0.* -> Model: feat.conv.mlp0.*
    model["feat.conv.mlp0.weight"] = Tensor(st["mlp1.0.weight"]);
    model["feat.conv.mlp0.bias"] = Tensor(st["mlp1.0.bias"]);
    model["feat.conv.mlp0.bn_mean"] = Tensor(st["mlp1.0.bn_mean"]);
    model["feat.conv.mlp0.bn_var"] = Tensor(st["mlp1.0.bn_var"]);
    model["feat.conv.mlp0.bn_gamma"] = Tensor(st["mlp1.0.bn_gamma"]);
    model["feat.conv.mlp0.bn_beta"] = Tensor(st["mlp1.0.bn_beta"]);
    
    // TNet2 (feature transformation network - 64x64) MLP layers
    // SafeTensors: tnet2.* -> Model: feat.fstn.*
    model["feat.fstn.mlp.mlp0.weight"] = Tensor(st["tnet2.mlp.0.weight"]);
    model["feat.fstn.mlp.mlp0.bias"] = Tensor(st["tnet2.mlp.0.bias"]);
    model["feat.fstn.mlp.mlp0.bn_mean"] = Tensor(st["tnet2.mlp.0.bn_mean"]);
    model["feat.fstn.mlp.mlp0.bn_var"] = Tensor(st["tnet2.mlp.0.bn_var"]);
    model["feat.fstn.mlp.mlp0.bn_gamma"] = Tensor(st["tnet2.mlp.0.bn_gamma"]);
    model["feat.fstn.mlp.mlp0.bn_beta"] = Tensor(st["tnet2.mlp.0.bn_beta"]);
    
    model["feat.fstn.mlp.mlp1.weight"] = Tensor(st["tnet2.mlp.1.weight"]);
    model["feat.fstn.mlp.mlp1.bias"] = Tensor(st["tnet2.mlp.1.bias"]);
    model["feat.fstn.mlp.mlp1.bn_mean"] = Tensor(st["tnet2.mlp.1.bn_mean"]);
    model["feat.fstn.mlp.mlp1.bn_var"] = Tensor(st["tnet2.mlp.1.bn_var"]);
    model["feat.fstn.mlp.mlp1.bn_gamma"] = Tensor(st["tnet2.mlp.1.bn_gamma"]);
    model["feat.fstn.mlp.mlp1.bn_beta"] = Tensor(st["tnet2.mlp.1.bn_beta"]);
    
    model["feat.fstn.mlp.mlp2.weight"] = Tensor(st["tnet2.mlp.2.weight"]);
    model["feat.fstn.mlp.mlp2.bias"] = Tensor(st["tnet2.mlp.2.bias"]);
    model["feat.fstn.mlp.mlp2.bn_mean"] = Tensor(st["tnet2.mlp.2.bn_mean"]);
    model["feat.fstn.mlp.mlp2.bn_var"] = Tensor(st["tnet2.mlp.2.bn_var"]);
    model["feat.fstn.mlp.mlp2.bn_gamma"] = Tensor(st["tnet2.mlp.2.bn_gamma"]);
    model["feat.fstn.mlp.mlp2.bn_beta"] = Tensor(st["tnet2.mlp.2.bn_beta"]);
    
    // TNet2 FC layers with BatchNorm
    model["feat.fstn.fc.block0.weight"] = Tensor(st["tnet2.fc.0.weight"]);
    model["feat.fstn.fc.block0.bias"] = Tensor(st["tnet2.fc.0.bias"]);
    model["feat.fstn.fc.block0.mean"] = Tensor(st["tnet2.fc.0.mean"]);
    model["feat.fstn.fc.block0.var"] = Tensor(st["tnet2.fc.0.var"]);
    model["feat.fstn.fc.block0.gamma"] = Tensor(st["tnet2.fc.0.gamma"]);
    model["feat.fstn.fc.block0.beta"] = Tensor(st["tnet2.fc.0.beta"]);
    
    model["feat.fstn.fc.block1.weight"] = Tensor(st["tnet2.fc.1.weight"]);
    model["feat.fstn.fc.block1.bias"] = Tensor(st["tnet2.fc.1.bias"]);
    model["feat.fstn.fc.block1.mean"] = Tensor(st["tnet2.fc.1.mean"]);
    model["feat.fstn.fc.block1.var"] = Tensor(st["tnet2.fc.1.var"]);
    model["feat.fstn.fc.block1.gamma"] = Tensor(st["tnet2.fc.1.gamma"]);
    model["feat.fstn.fc.block1.beta"] = Tensor(st["tnet2.fc.1.beta"]);
    
    model["feat.fstn.fc.lastBlock.weight"] = Tensor(st["tnet2.fc.2.weight"]);
    model["feat.fstn.fc.lastBlock.bias"] = Tensor(st["tnet2.fc.2.bias"]);
    
    // MLP2 (second feature extraction)
    // SafeTensors: mlp2.0.* -> Model: feat.conv.mlp1.*
    model["feat.conv.mlp1.weight"] = Tensor(st["mlp2.0.weight"]);
    model["feat.conv.mlp1.bias"] = Tensor(st["mlp2.0.bias"]);
    model["feat.conv.mlp1.bn_mean"] = Tensor(st["mlp2.0.bn_mean"]);
    model["feat.conv.mlp1.bn_var"] = Tensor(st["mlp2.0.bn_var"]);
    model["feat.conv.mlp1.bn_gamma"] = Tensor(st["mlp2.0.bn_gamma"]);
    model["feat.conv.mlp1.bn_beta"] = Tensor(st["mlp2.0.bn_beta"]);
    
    // SafeTensors: mlp2.1.* -> Model: feat.conv.mlp2.* (PointWiseConvNode - no ReLU)
    model["feat.conv.mlp2.weight"] = Tensor(st["mlp2.1.weight"]);
    model["feat.conv.mlp2.bias"] = Tensor(st["mlp2.1.bias"]);
    model["feat.conv.mlp2.bn_mean"] = Tensor(st["mlp2.1.bn_mean"]);
    model["feat.conv.mlp2.bn_var"] = Tensor(st["mlp2.1.bn_var"]);
    model["feat.conv.mlp2.bn_gamma"] = Tensor(st["mlp2.1.bn_gamma"]);
    model["feat.conv.mlp2.bn_beta"] = Tensor(st["mlp2.1.bn_beta"]);
    
    // Segmentation head (MLPSequence with BatchNorm + ReLU on all but last layer)
    // SafeTensors: segHead.0-3.* -> Model: conv1-4.*
    model["conv1.weight"] = Tensor(st["segHead.0.weight"]);
    model["conv1.bias"] = Tensor(st["segHead.0.bias"]);
    model["conv1.bn_mean"] = Tensor(st["segHead.0.bn_mean"]);
    model["conv1.bn_var"] = Tensor(st["segHead.0.bn_var"]);
    model["conv1.bn_gamma"] = Tensor(st["segHead.0.bn_gamma"]);
    model["conv1.bn_beta"] = Tensor(st["segHead.0.bn_beta"]);
    
    model["conv2.weight"] = Tensor(st["segHead.1.weight"]);
    model["conv2.bias"] = Tensor(st["segHead.1.bias"]);
    model["conv2.bn_mean"] = Tensor(st["segHead.1.bn_mean"]);
    model["conv2.bn_var"] = Tensor(st["segHead.1.bn_var"]);
    model["conv2.bn_gamma"] = Tensor(st["segHead.1.bn_gamma"]);
    model["conv2.bn_beta"] = Tensor(st["segHead.1.bn_beta"]);
    
    model["conv3.weight"] = Tensor(st["segHead.2.weight"]);
    model["conv3.bias"] = Tensor(st["segHead.2.bias"]);
    model["conv3.bn_mean"] = Tensor(st["segHead.2.bn_mean"]);
    model["conv3.bn_var"] = Tensor(st["segHead.2.bn_var"]);
    model["conv3.bn_gamma"] = Tensor(st["segHead.2.bn_gamma"]);
    model["conv3.bn_beta"] = Tensor(st["segHead.2.bn_beta"]);
    
    model["conv4.weight"] = Tensor(st["segHead.3.weight"]);
    model["conv4.bias"] = Tensor(st["segHead.3.bias"]);
    
    std::cout << "✓ All weights loaded from SafeTensors" << std::endl;
}

void loadPointNetWeightsFromJSON(PointNetSegment& model, const std::string& weights_file) {
    // Load JSON (yanx27 format)
    JsonParser json(weights_file.c_str());
    
    // Helper lambda to load tensor - NO TRANSPOSE needed with PyTorch convention!
    // PyTorch format is now used directly: [output_dim, input_dim]
    auto loadTensor = [&json](const std::string& key) -> Tensor {
        try {
            // Load tensor as-is from JSON (PyTorch native format)
            return Tensor(json[key]);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to load tensor '" + key + "': " + e.what());
        }
    };
    
    // STN (input transformation network - 3x3)
    std::cout << "  Loading STN weights..." << std::endl;
    model["feat.stn.mlp.mlp0.weight"] = loadTensor("feat.stn.mlp.mlp0.weight");
    model["feat.stn.mlp.mlp0.bias"] = loadTensor("feat.stn.mlp.mlp0.bias");
    model["feat.stn.mlp.mlp0.bn_mean"] = loadTensor("feat.stn.mlp.mlp0.mean");
    model["feat.stn.mlp.mlp0.bn_var"] = loadTensor("feat.stn.mlp.mlp0.var");
    model["feat.stn.mlp.mlp0.bn_gamma"] = loadTensor("feat.stn.mlp.mlp0.gamma");
    model["feat.stn.mlp.mlp0.bn_beta"] = loadTensor("feat.stn.mlp.mlp0.beta");
    
    model["feat.stn.mlp.mlp1.weight"] = loadTensor("feat.stn.mlp.mlp1.weight");
    model["feat.stn.mlp.mlp1.bias"] = loadTensor("feat.stn.mlp.mlp1.bias");
    model["feat.stn.mlp.mlp1.bn_mean"] = loadTensor("feat.stn.mlp.mlp1.mean");
    model["feat.stn.mlp.mlp1.bn_var"] = loadTensor("feat.stn.mlp.mlp1.var");
    model["feat.stn.mlp.mlp1.bn_gamma"] = loadTensor("feat.stn.mlp.mlp1.gamma");
    model["feat.stn.mlp.mlp1.bn_beta"] = loadTensor("feat.stn.mlp.mlp1.beta");
    
    model["feat.stn.mlp.mlp2.weight"] = loadTensor("feat.stn.mlp.mlp2.weight");
    model["feat.stn.mlp.mlp2.bias"] = loadTensor("feat.stn.mlp.mlp2.bias");
    model["feat.stn.mlp.mlp2.bn_mean"] = loadTensor("feat.stn.mlp.mlp2.mean");
    model["feat.stn.mlp.mlp2.bn_var"] = loadTensor("feat.stn.mlp.mlp2.var");
    model["feat.stn.mlp.mlp2.bn_gamma"] = loadTensor("feat.stn.mlp.mlp2.gamma");
    model["feat.stn.mlp.mlp2.bn_beta"] = loadTensor("feat.stn.mlp.mlp2.beta");
    
    // STN FC layers
    model["feat.stn.fc.block0.weight"] = loadTensor("feat.stn.fc.block0.weight");
    model["feat.stn.fc.block0.bias"] = loadTensor("feat.stn.fc.block0.bias");
    model["feat.stn.fc.block0.mean"] = loadTensor("feat.stn.fc.block0.mean");
    model["feat.stn.fc.block0.var"] = loadTensor("feat.stn.fc.block0.var");
    model["feat.stn.fc.block0.gamma"] = loadTensor("feat.stn.fc.block0.gamma");
    model["feat.stn.fc.block0.beta"] = loadTensor("feat.stn.fc.block0.beta");
    
    model["feat.stn.fc.block1.weight"] = loadTensor("feat.stn.fc.block1.weight");
    model["feat.stn.fc.block1.bias"] = loadTensor("feat.stn.fc.block1.bias");
    model["feat.stn.fc.block1.mean"] = loadTensor("feat.stn.fc.block1.mean");
    model["feat.stn.fc.block1.var"] = loadTensor("feat.stn.fc.block1.var");
    model["feat.stn.fc.block1.gamma"] = loadTensor("feat.stn.fc.block1.gamma");
    model["feat.stn.fc.block1.beta"] = loadTensor("feat.stn.fc.block1.beta");
    
    model["feat.stn.fc.lastBlock.weight"] = loadTensor("feat.stn.fc.lastBlock.weight");
    model["feat.stn.fc.lastBlock.bias"] = loadTensor("feat.stn.fc.lastBlock.bias");

    // Conv layers (feature extraction)
    model["feat.conv.mlp0.weight"] = loadTensor("feat.conv.mlp0.weight");
    model["feat.conv.mlp0.bias"] = loadTensor("feat.conv.mlp0.bias");
    model["feat.conv.mlp0.bn_mean"] = loadTensor("feat.conv.mlp0.mean");
    model["feat.conv.mlp0.bn_var"] = loadTensor("feat.conv.mlp0.var");
    model["feat.conv.mlp0.bn_gamma"] = loadTensor("feat.conv.mlp0.gamma");
    model["feat.conv.mlp0.bn_beta"] = loadTensor("feat.conv.mlp0.beta");
    
    model["feat.conv.mlp1.weight"] = loadTensor("feat.conv.mlp1.weight");
    model["feat.conv.mlp1.bias"] = loadTensor("feat.conv.mlp1.bias");
    model["feat.conv.mlp1.bn_mean"] = loadTensor("feat.conv.mlp1.mean");
    model["feat.conv.mlp1.bn_var"] = loadTensor("feat.conv.mlp1.var");
    model["feat.conv.mlp1.bn_gamma"] = loadTensor("feat.conv.mlp1.gamma");
    model["feat.conv.mlp1.bn_beta"] = loadTensor("feat.conv.mlp1.beta");
    
    model["feat.conv3.weight"] = loadTensor("feat.conv.mlp2.weight");
    model["feat.conv3.bias"] = loadTensor("feat.conv.mlp2.bias");
    model["feat.conv3.bn_mean"] = loadTensor("feat.conv.mlp2.mean");
    model["feat.conv3.bn_var"] = loadTensor("feat.conv.mlp2.var");
    model["feat.conv3.bn_gamma"] = loadTensor("feat.conv.mlp2.gamma");
    model["feat.conv3.bn_beta"] = loadTensor("feat.conv.mlp2.beta");

    // FSTN (feature transformation network - 64x64)
    model["feat.fstn.mlp.mlp0.weight"] = loadTensor("feat.fstn.mlp.mlp0.weight");
    model["feat.fstn.mlp.mlp0.bias"] = loadTensor("feat.fstn.mlp.mlp0.bias");
    model["feat.fstn.mlp.mlp0.bn_mean"] = loadTensor("feat.fstn.mlp.mlp0.mean");
    model["feat.fstn.mlp.mlp0.bn_var"] = loadTensor("feat.fstn.mlp.mlp0.var");
    model["feat.fstn.mlp.mlp0.bn_gamma"] = loadTensor("feat.fstn.mlp.mlp0.gamma");
    model["feat.fstn.mlp.mlp0.bn_beta"] = loadTensor("feat.fstn.mlp.mlp0.beta");
    
    model["feat.fstn.mlp.mlp1.weight"] = loadTensor("feat.fstn.mlp.mlp1.weight");
    model["feat.fstn.mlp.mlp1.bias"] = loadTensor("feat.fstn.mlp.mlp1.bias");
    model["feat.fstn.mlp.mlp1.bn_mean"] = loadTensor("feat.fstn.mlp.mlp1.mean");
    model["feat.fstn.mlp.mlp1.bn_var"] = loadTensor("feat.fstn.mlp.mlp1.var");
    model["feat.fstn.mlp.mlp1.bn_gamma"] = loadTensor("feat.fstn.mlp.mlp1.gamma");
    model["feat.fstn.mlp.mlp1.bn_beta"] = loadTensor("feat.fstn.mlp.mlp1.beta");
    
    model["feat.fstn.mlp.mlp2.weight"] = loadTensor("feat.fstn.mlp.mlp2.weight");
    model["feat.fstn.mlp.mlp2.bias"] = loadTensor("feat.fstn.mlp.mlp2.bias");
    model["feat.fstn.mlp.mlp2.bn_mean"] = loadTensor("feat.fstn.mlp.mlp2.mean");
    model["feat.fstn.mlp.mlp2.bn_var"] = loadTensor("feat.fstn.mlp.mlp2.var");
    model["feat.fstn.mlp.mlp2.bn_gamma"] = loadTensor("feat.fstn.mlp.mlp2.gamma");
    model["feat.fstn.mlp.mlp2.bn_beta"] = loadTensor("feat.fstn.mlp.mlp2.beta");
    
    // FSTN FC layers
    model["feat.fstn.fc.block0.weight"] = loadTensor("feat.fstn.fc.block0.weight");
    model["feat.fstn.fc.block0.bias"] = loadTensor("feat.fstn.fc.block0.bias");
    model["feat.fstn.fc.block0.mean"] = loadTensor("feat.fstn.fc.block0.mean");
    model["feat.fstn.fc.block0.var"] = loadTensor("feat.fstn.fc.block0.var");
    model["feat.fstn.fc.block0.gamma"] = loadTensor("feat.fstn.fc.block0.gamma");
    model["feat.fstn.fc.block0.beta"] = loadTensor("feat.fstn.fc.block0.beta");
    
    model["feat.fstn.fc.block1.weight"] = loadTensor("feat.fstn.fc.block1.weight");
    model["feat.fstn.fc.block1.bias"] = loadTensor("feat.fstn.fc.block1.bias");
    model["feat.fstn.fc.block1.mean"] = loadTensor("feat.fstn.fc.block1.mean");
    model["feat.fstn.fc.block1.var"] = loadTensor("feat.fstn.fc.block1.var");
    model["feat.fstn.fc.block1.gamma"] = loadTensor("feat.fstn.fc.block1.gamma");
    model["feat.fstn.fc.block1.beta"] = loadTensor("feat.fstn.fc.block1.beta");
    
    model["feat.fstn.fc.lastBlock.weight"] = loadTensor("feat.fstn.fc.lastBlock.weight");
    model["feat.fstn.fc.lastBlock.bias"] = loadTensor("feat.fstn.fc.lastBlock.bias");

    // Segmentation head (conv1-4 with BatchNorm)
    model["conv1.weight"] = loadTensor("conv1.weight");
    model["conv1.bias"] = loadTensor("conv1.bias");
    model["conv1.bn_mean"] = loadTensor("bn1.mean");
    model["conv1.bn_var"] = loadTensor("bn1.var");
    model["conv1.bn_gamma"] = loadTensor("bn1.weight");
    model["conv1.bn_beta"] = loadTensor("bn1.bias");
    
    model["conv2.weight"] = loadTensor("conv2.weight");
    model["conv2.bias"] = loadTensor("conv2.bias");
    model["conv2.bn_mean"] = loadTensor("bn2.mean");
    model["conv2.bn_var"] = loadTensor("bn2.var");
    model["conv2.bn_gamma"] = loadTensor("bn2.weight");
    model["conv2.bn_beta"] = loadTensor("bn2.bias");
    
    model["conv3.weight"] = loadTensor("conv3.weight");
    model["conv3.bias"] = loadTensor("conv3.bias");
    model["conv3.bn_mean"] = loadTensor("bn3.mean");
    model["conv3.bn_var"] = loadTensor("bn3.var");
    model["conv3.bn_gamma"] = loadTensor("bn3.weight");
    model["conv3.bn_beta"] = loadTensor("bn3.bias");
    
    model["conv4.weight"] = loadTensor("conv4.weight");
    model["conv4.bias"] = loadTensor("conv4.bias");
}

} // namespace networks

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
    
    // TNet1 (input transformation network - 3x3)
    model["encoder.tnet1.mlp.mlp0.weight"] = Tensor(st["tnet1.mlp.0.weight"]);
    model["encoder.tnet1.mlp.mlp0.bias"] = Tensor(st["tnet1.mlp.0.bias"]);
    model["encoder.tnet1.mlp.mlp1.weight"] = Tensor(st["tnet1.mlp.1.weight"]);
    model["encoder.tnet1.mlp.mlp1.bias"] = Tensor(st["tnet1.mlp.1.bias"]);
    model["encoder.tnet1.mlp.mlp2.weight"] = Tensor(st["tnet1.mlp.2.weight"]);
    model["encoder.tnet1.mlp.mlp2.bias"] = Tensor(st["tnet1.mlp.2.bias"]);
    
    // FCBNSequence uses block0, block1, lastBlock
    model["encoder.tnet1.fc.block0.weight"] = Tensor(st["tnet1.fc.0.weight"]);
    model["encoder.tnet1.fc.block0.bias"] = Tensor(st["tnet1.fc.0.bias"]);
    uint32_t st_fc0_out = model["encoder.tnet1.fc.block0.weight"].shape()[1];
    model["encoder.tnet1.fc.block0.bn_mean"] = Tensor(st_fc0_out).set(std::vector<float>(st_fc0_out, 0.0f));
    model["encoder.tnet1.fc.block0.bn_var"] = Tensor(st_fc0_out).set(std::vector<float>(st_fc0_out, 1.0f));
    model["encoder.tnet1.fc.block0.bn_gamma"] = Tensor(st_fc0_out).set(std::vector<float>(st_fc0_out, 1.0f));
    model["encoder.tnet1.fc.block0.bn_beta"] = Tensor(st_fc0_out).set(std::vector<float>(st_fc0_out, 0.0f));
    
    model["encoder.tnet1.fc.block1.weight"] = Tensor(st["tnet1.fc.1.weight"]);
    model["encoder.tnet1.fc.block1.bias"] = Tensor(st["tnet1.fc.1.bias"]);
    uint32_t st_fc1_out = model["encoder.tnet1.fc.block1.weight"].shape()[1];
    model["encoder.tnet1.fc.block1.bn_mean"] = Tensor(st_fc1_out).set(std::vector<float>(st_fc1_out, 0.0f));
    model["encoder.tnet1.fc.block1.bn_var"] = Tensor(st_fc1_out).set(std::vector<float>(st_fc1_out, 1.0f));
    model["encoder.tnet1.fc.block1.bn_gamma"] = Tensor(st_fc1_out).set(std::vector<float>(st_fc1_out, 1.0f));
    model["encoder.tnet1.fc.block1.bn_beta"] = Tensor(st_fc1_out).set(std::vector<float>(st_fc1_out, 0.0f));
    
    model["encoder.tnet1.fc.lastBlock.weight"] = Tensor(st["tnet1.fc.2.weight"]);
    model["encoder.tnet1.fc.lastBlock.bias"] = Tensor(st["tnet1.fc.2.bias"]);
    
    // MLP1 (first feature extraction)
    model["encoder.mlp1.mlp0.weight"] = Tensor(st["mlp1.0.weight"]);
    model["encoder.mlp1.mlp0.bias"] = Tensor(st["mlp1.0.bias"]);
    model["encoder.mlp1.mlp1.weight"] = Tensor(st["mlp1.1.weight"]);
    model["encoder.mlp1.mlp1.bias"] = Tensor(st["mlp1.1.bias"]);
    
    // TNet2 (feature transformation network - 64x64)
    model["encoder.tnet2.mlp.mlp0.weight"] = Tensor(st["tnet2.mlp.0.weight"]);
    model["encoder.tnet2.mlp.mlp0.bias"] = Tensor(st["tnet2.mlp.0.bias"]);
    model["encoder.tnet2.mlp.mlp1.weight"] = Tensor(st["tnet2.mlp.1.weight"]);
    model["encoder.tnet2.mlp.mlp1.bias"] = Tensor(st["tnet2.mlp.1.bias"]);
    model["encoder.tnet2.mlp.mlp2.weight"] = Tensor(st["tnet2.mlp.2.weight"]);
    model["encoder.tnet2.mlp.mlp2.bias"] = Tensor(st["tnet2.mlp.2.bias"]);
    
    // FCBNSequence uses block0, block1, lastBlock
    model["encoder.tnet2.fc.block0.weight"] = Tensor(st["tnet2.fc.0.weight"]);
    model["encoder.tnet2.fc.block0.bias"] = Tensor(st["tnet2.fc.0.bias"]);
    uint32_t st_tnet2_fc0_out = model["encoder.tnet2.fc.block0.weight"].shape()[1];
    model["encoder.tnet2.fc.block0.bn_mean"] = Tensor(st_tnet2_fc0_out).set(std::vector<float>(st_tnet2_fc0_out, 0.0f));
    model["encoder.tnet2.fc.block0.bn_var"] = Tensor(st_tnet2_fc0_out).set(std::vector<float>(st_tnet2_fc0_out, 1.0f));
    model["encoder.tnet2.fc.block0.bn_gamma"] = Tensor(st_tnet2_fc0_out).set(std::vector<float>(st_tnet2_fc0_out, 1.0f));
    model["encoder.tnet2.fc.block0.bn_beta"] = Tensor(st_tnet2_fc0_out).set(std::vector<float>(st_tnet2_fc0_out, 0.0f));
    
    model["encoder.tnet2.fc.block1.weight"] = Tensor(st["tnet2.fc.1.weight"]);
    model["encoder.tnet2.fc.block1.bias"] = Tensor(st["tnet2.fc.1.bias"]);
    uint32_t st_tnet2_fc1_out = model["encoder.tnet2.fc.block1.weight"].shape()[1];
    model["encoder.tnet2.fc.block1.bn_mean"] = Tensor(st_tnet2_fc1_out).set(std::vector<float>(st_tnet2_fc1_out, 0.0f));
    model["encoder.tnet2.fc.block1.bn_var"] = Tensor(st_tnet2_fc1_out).set(std::vector<float>(st_tnet2_fc1_out, 1.0f));
    model["encoder.tnet2.fc.block1.bn_gamma"] = Tensor(st_tnet2_fc1_out).set(std::vector<float>(st_tnet2_fc1_out, 1.0f));
    model["encoder.tnet2.fc.block1.bn_beta"] = Tensor(st_tnet2_fc1_out).set(std::vector<float>(st_tnet2_fc1_out, 0.0f));
    
    model["encoder.tnet2.fc.lastBlock.weight"] = Tensor(st["tnet2.fc.2.weight"]);
    model["encoder.tnet2.fc.lastBlock.bias"] = Tensor(st["tnet2.fc.2.bias"]);
    
    // MLP2 (second feature extraction)
    model["encoder.mlp2.mlp0.weight"] = Tensor(st["mlp2.0.weight"]);
    model["encoder.mlp2.mlp0.bias"] = Tensor(st["mlp2.0.bias"]);
    model["encoder.mlp2.mlp1.weight"] = Tensor(st["mlp2.1.weight"]);
    model["encoder.mlp2.mlp1.bias"] = Tensor(st["mlp2.1.bias"]);
    
    // Segmentation head
    model["segHead.mlp0.weight"] = Tensor(st["segHead.0.weight"]);
    model["segHead.mlp0.bias"] = Tensor(st["segHead.0.bias"]);
    model["segHead.mlp1.weight"] = Tensor(st["segHead.1.weight"]);
    model["segHead.mlp1.bias"] = Tensor(st["segHead.1.bias"]);
    model["segHead.mlp2.weight"] = Tensor(st["segHead.2.weight"]);
    model["segHead.mlp2.bias"] = Tensor(st["segHead.2.bias"]);
    
    std::cout << "✓ All weights loaded from SafeTensors" << std::endl;
}

void loadPointNetWeightsFromJSON(PointNetSegment& model, const std::string& weights_file) {
    // Load JSON (yanx27 format)
    JsonParser json(weights_file.c_str());
    
    // Helper lambda to load tensor with automatic transpose for weights
    // PyTorch format: [output_dim, input_dim, ...] → Vulkan format: [input_dim, output_dim]
    auto loadTensor = [&json](const std::string& key) -> Tensor {
        try {
            std::vector<uint32_t> shape;
            std::vector<float> data = json[key].parseNDArray(shape);
            
            // Transpose weights (but not biases or BatchNorm params)
            if (key.find(".weight") != std::string::npos) {
                if (shape.size() == 2) {
                    // 2D weight: [M, N] → [N, M]
                    uint32_t M = shape[0], N = shape[1];
                    std::vector<float> transposed(M * N);
                    for (uint32_t i = 0; i < M; ++i) {
                        for (uint32_t j = 0; j < N; ++j) {
                            transposed[j * M + i] = data[i * N + j];
                        }
                    }
                    return Tensor(N, M).set(transposed);
                } else if (shape.size() == 3 && shape[2] == 1) {
                    // 3D weight with last dim = 1: [M, N, 1] → [N, M]
                    // Squeeze last dimension and transpose
                    uint32_t M = shape[0], N = shape[1];
                    std::vector<float> transposed(M * N);
                    for (uint32_t i = 0; i < M; ++i) {
                        for (uint32_t j = 0; j < N; ++j) {
                            transposed[j * M + i] = data[i * N + j];  // Skip last dim (always 0)
                        }
                    }
                    return Tensor(N, M).set(transposed);
                }
            }
            // No transpose needed (bias, mean, var, gamma, beta, or unsupported shape)
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

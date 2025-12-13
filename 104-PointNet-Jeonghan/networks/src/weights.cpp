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
    model["encoder.tnet1.fc.block0.mean"] = Tensor(st_fc0_out).set(std::vector<float>(st_fc0_out, 0.0f));
    model["encoder.tnet1.fc.block0.var"] = Tensor(st_fc0_out).set(std::vector<float>(st_fc0_out, 1.0f));
    model["encoder.tnet1.fc.block0.gamma"] = Tensor(st_fc0_out).set(std::vector<float>(st_fc0_out, 1.0f));
    model["encoder.tnet1.fc.block0.beta"] = Tensor(st_fc0_out).set(std::vector<float>(st_fc0_out, 0.0f));
    
    model["encoder.tnet1.fc.block1.weight"] = Tensor(st["tnet1.fc.1.weight"]);
    model["encoder.tnet1.fc.block1.bias"] = Tensor(st["tnet1.fc.1.bias"]);
    uint32_t st_fc1_out = model["encoder.tnet1.fc.block1.weight"].shape()[1];
    model["encoder.tnet1.fc.block1.mean"] = Tensor(st_fc1_out).set(std::vector<float>(st_fc1_out, 0.0f));
    model["encoder.tnet1.fc.block1.var"] = Tensor(st_fc1_out).set(std::vector<float>(st_fc1_out, 1.0f));
    model["encoder.tnet1.fc.block1.gamma"] = Tensor(st_fc1_out).set(std::vector<float>(st_fc1_out, 1.0f));
    model["encoder.tnet1.fc.block1.beta"] = Tensor(st_fc1_out).set(std::vector<float>(st_fc1_out, 0.0f));
    
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
    model["encoder.tnet2.fc.block0.mean"] = Tensor(st_tnet2_fc0_out).set(std::vector<float>(st_tnet2_fc0_out, 0.0f));
    model["encoder.tnet2.fc.block0.var"] = Tensor(st_tnet2_fc0_out).set(std::vector<float>(st_tnet2_fc0_out, 1.0f));
    model["encoder.tnet2.fc.block0.gamma"] = Tensor(st_tnet2_fc0_out).set(std::vector<float>(st_tnet2_fc0_out, 1.0f));
    model["encoder.tnet2.fc.block0.beta"] = Tensor(st_tnet2_fc0_out).set(std::vector<float>(st_tnet2_fc0_out, 0.0f));
    
    model["encoder.tnet2.fc.block1.weight"] = Tensor(st["tnet2.fc.1.weight"]);
    model["encoder.tnet2.fc.block1.bias"] = Tensor(st["tnet2.fc.1.bias"]);
    uint32_t st_tnet2_fc1_out = model["encoder.tnet2.fc.block1.weight"].shape()[1];
    model["encoder.tnet2.fc.block1.mean"] = Tensor(st_tnet2_fc1_out).set(std::vector<float>(st_tnet2_fc1_out, 0.0f));
    model["encoder.tnet2.fc.block1.var"] = Tensor(st_tnet2_fc1_out).set(std::vector<float>(st_tnet2_fc1_out, 1.0f));
    model["encoder.tnet2.fc.block1.gamma"] = Tensor(st_tnet2_fc1_out).set(std::vector<float>(st_tnet2_fc1_out, 1.0f));
    model["encoder.tnet2.fc.block1.beta"] = Tensor(st_tnet2_fc1_out).set(std::vector<float>(st_tnet2_fc1_out, 0.0f));
    
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
    // Load JSON
    JsonParser json = JsonParser(weights_file.c_str());
    
    // TNet1 (input transformation network)
    model["encoder.tnet1.mlp.mlp0.weight"] = Tensor(json["tnet1.mlp.0.weight"]);
    model["encoder.tnet1.mlp.mlp0.bias"] = Tensor(json["tnet1.mlp.0.bias"]);
    model["encoder.tnet1.mlp.mlp1.weight"] = Tensor(json["tnet1.mlp.1.weight"]);
    model["encoder.tnet1.mlp.mlp1.bias"] = Tensor(json["tnet1.mlp.1.bias"]);
    model["encoder.tnet1.mlp.mlp2.weight"] = Tensor(json["tnet1.mlp.2.weight"]);
    model["encoder.tnet1.mlp.mlp2.bias"] = Tensor(json["tnet1.mlp.2.bias"]);
    
    // FCBNSequence uses block0, block1, lastBlock
    // FC parameters
    model["encoder.tnet1.fc.block0.weight"] = Tensor(json["tnet1.fc.0.weight"]);
    model["encoder.tnet1.fc.block0.bias"] = Tensor(json["tnet1.fc.0.bias"]);
    // BatchNorm parameters (identity transform: mean=0, var=1, gamma=1, beta=0)
    uint32_t fc0_out = model["encoder.tnet1.fc.block0.weight"].shape()[1];
    model["encoder.tnet1.fc.block0.mean"] = Tensor(fc0_out).set(std::vector<float>(fc0_out, 0.0f));
    model["encoder.tnet1.fc.block0.var"] = Tensor(fc0_out).set(std::vector<float>(fc0_out, 1.0f));
    model["encoder.tnet1.fc.block0.gamma"] = Tensor(fc0_out).set(std::vector<float>(fc0_out, 1.0f));
    model["encoder.tnet1.fc.block0.beta"] = Tensor(fc0_out).set(std::vector<float>(fc0_out, 0.0f));
    
    model["encoder.tnet1.fc.block1.weight"] = Tensor(json["tnet1.fc.1.weight"]);
    model["encoder.tnet1.fc.block1.bias"] = Tensor(json["tnet1.fc.1.bias"]);
    uint32_t fc1_out = model["encoder.tnet1.fc.block1.weight"].shape()[1];
    model["encoder.tnet1.fc.block1.mean"] = Tensor(fc1_out).set(std::vector<float>(fc1_out, 0.0f));
    model["encoder.tnet1.fc.block1.var"] = Tensor(fc1_out).set(std::vector<float>(fc1_out, 1.0f));
    model["encoder.tnet1.fc.block1.gamma"] = Tensor(fc1_out).set(std::vector<float>(fc1_out, 1.0f));
    model["encoder.tnet1.fc.block1.beta"] = Tensor(fc1_out).set(std::vector<float>(fc1_out, 0.0f));
    
    model["encoder.tnet1.fc.lastBlock.weight"] = Tensor(json["tnet1.fc.2.weight"]);
    model["encoder.tnet1.fc.lastBlock.bias"] = Tensor(json["tnet1.fc.2.bias"]);

    // MLP1 (first feature extraction)
    model["encoder.mlp1.mlp0.weight"] = Tensor(json["mlp1.0.weight"]);
    model["encoder.mlp1.mlp0.bias"] = Tensor(json["mlp1.0.bias"]);
    model["encoder.mlp1.mlp1.weight"] = Tensor(json["mlp1.1.weight"]);
    model["encoder.mlp1.mlp1.bias"] = Tensor(json["mlp1.1.bias"]);

    // TNet2 (feature transformation network)
    model["encoder.tnet2.mlp.mlp0.weight"] = Tensor(json["tnet2.mlp.0.weight"]);
    model["encoder.tnet2.mlp.mlp0.bias"] = Tensor(json["tnet2.mlp.0.bias"]);
    model["encoder.tnet2.mlp.mlp1.weight"] = Tensor(json["tnet2.mlp.1.weight"]);
    model["encoder.tnet2.mlp.mlp1.bias"] = Tensor(json["tnet2.mlp.1.bias"]);
    model["encoder.tnet2.mlp.mlp2.weight"] = Tensor(json["tnet2.mlp.2.weight"]);
    model["encoder.tnet2.mlp.mlp2.bias"] = Tensor(json["tnet2.mlp.2.bias"]);
    
    // FCBNSequence uses block0, block1, lastBlock
    model["encoder.tnet2.fc.block0.weight"] = Tensor(json["tnet2.fc.0.weight"]);
    model["encoder.tnet2.fc.block0.bias"] = Tensor(json["tnet2.fc.0.bias"]);
    uint32_t tnet2_fc0_out = model["encoder.tnet2.fc.block0.weight"].shape()[1];
    model["encoder.tnet2.fc.block0.mean"] = Tensor(tnet2_fc0_out).set(std::vector<float>(tnet2_fc0_out, 0.0f));
    model["encoder.tnet2.fc.block0.var"] = Tensor(tnet2_fc0_out).set(std::vector<float>(tnet2_fc0_out, 1.0f));
    model["encoder.tnet2.fc.block0.gamma"] = Tensor(tnet2_fc0_out).set(std::vector<float>(tnet2_fc0_out, 1.0f));
    model["encoder.tnet2.fc.block0.beta"] = Tensor(tnet2_fc0_out).set(std::vector<float>(tnet2_fc0_out, 0.0f));
    
    model["encoder.tnet2.fc.block1.weight"] = Tensor(json["tnet2.fc.1.weight"]);
    model["encoder.tnet2.fc.block1.bias"] = Tensor(json["tnet2.fc.1.bias"]);
    uint32_t tnet2_fc1_out = model["encoder.tnet2.fc.block1.weight"].shape()[1];
    model["encoder.tnet2.fc.block1.mean"] = Tensor(tnet2_fc1_out).set(std::vector<float>(tnet2_fc1_out, 0.0f));
    model["encoder.tnet2.fc.block1.var"] = Tensor(tnet2_fc1_out).set(std::vector<float>(tnet2_fc1_out, 1.0f));
    model["encoder.tnet2.fc.block1.gamma"] = Tensor(tnet2_fc1_out).set(std::vector<float>(tnet2_fc1_out, 1.0f));
    model["encoder.tnet2.fc.block1.beta"] = Tensor(tnet2_fc1_out).set(std::vector<float>(tnet2_fc1_out, 0.0f));
    
    model["encoder.tnet2.fc.lastBlock.weight"] = Tensor(json["tnet2.fc.2.weight"]);
    model["encoder.tnet2.fc.lastBlock.bias"] = Tensor(json["tnet2.fc.2.bias"]);

    // MLP2 (second feature extraction)
    model["encoder.mlp2.mlp0.weight"] = Tensor(json["mlp2.0.weight"]);
    model["encoder.mlp2.mlp0.bias"] = Tensor(json["mlp2.0.bias"]);
    model["encoder.mlp2.mlp1.weight"] = Tensor(json["mlp2.1.weight"]);
    model["encoder.mlp2.mlp1.bias"] = Tensor(json["mlp2.1.bias"]);

    // Segmentation head
    model["segHead.mlp0.weight"] = Tensor(json["segHead.0.weight"]);
    model["segHead.mlp0.bias"] = Tensor(json["segHead.0.bias"]);
    model["segHead.mlp1.weight"] = Tensor(json["segHead.1.weight"]);
    model["segHead.mlp1.bias"] = Tensor(json["segHead.1.bias"]);
    model["segHead.mlp2.weight"] = Tensor(json["segHead.2.weight"]);
    model["segHead.mlp2.bias"] = Tensor(json["segHead.2.bias"]);
}

} // namespace networks

#include "weights.h"
#include <iostream>

namespace networks {

void loadPointNetWeights(PointNetSegment& model, const std::string& weights_file) {
    // Load JSON
    JsonParser json = JsonParser(weights_file.c_str());
    
    // TNet1 (input transformation network)
    model["encoder.tnet1.mlp.mlp0.weight"] = Tensor(json["tnet1.mlp.0.weight"]);
    model["encoder.tnet1.mlp.mlp0.bias"] = Tensor(json["tnet1.mlp.0.bias"]);
    model["encoder.tnet1.mlp.mlp1.weight"] = Tensor(json["tnet1.mlp.1.weight"]);
    model["encoder.tnet1.mlp.mlp1.bias"] = Tensor(json["tnet1.mlp.1.bias"]);
    model["encoder.tnet1.mlp.mlp2.weight"] = Tensor(json["tnet1.mlp.2.weight"]);
    model["encoder.tnet1.mlp.mlp2.bias"] = Tensor(json["tnet1.mlp.2.bias"]);
    
    model["encoder.tnet1.fc.fc0.weight"] = Tensor(json["tnet1.fc.0.weight"]);
    model["encoder.tnet1.fc.fc0.bias"] = Tensor(json["tnet1.fc.0.bias"]);
    model["encoder.tnet1.fc.fc1.weight"] = Tensor(json["tnet1.fc.1.weight"]);
    model["encoder.tnet1.fc.fc1.bias"] = Tensor(json["tnet1.fc.1.bias"]);
    model["encoder.tnet1.fc.fc2.weight"] = Tensor(json["tnet1.fc.2.weight"]);
    model["encoder.tnet1.fc.fc2.bias"] = Tensor(json["tnet1.fc.2.bias"]);
    model["encoder.tnet1.fc.fc3.weight"] = Tensor(json["tnet1.fc.3.weight"]);
    model["encoder.tnet1.fc.fc3.bias"] = Tensor(json["tnet1.fc.3.bias"]);

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
    
    model["encoder.tnet2.fc.fc0.weight"] = Tensor(json["tnet2.fc.0.weight"]);
    model["encoder.tnet2.fc.fc0.bias"] = Tensor(json["tnet2.fc.0.bias"]);
    model["encoder.tnet2.fc.fc1.weight"] = Tensor(json["tnet2.fc.1.weight"]);
    model["encoder.tnet2.fc.fc1.bias"] = Tensor(json["tnet2.fc.1.bias"]);
    model["encoder.tnet2.fc.fc2.weight"] = Tensor(json["tnet2.fc.2.weight"]);
    model["encoder.tnet2.fc.fc2.bias"] = Tensor(json["tnet2.fc.2.bias"]);
    model["encoder.tnet2.fc.fc3.weight"] = Tensor(json["tnet2.fc.3.weight"]);
    model["encoder.tnet2.fc.fc3.bias"] = Tensor(json["tnet2.fc.3.bias"]);

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

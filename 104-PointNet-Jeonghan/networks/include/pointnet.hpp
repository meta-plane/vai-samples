#ifndef POINTNET_HPP
#define POINTNET_HPP

#include "neuralNet.h"
#include "neuralNodes.h"
#include "jsonParser.h"
#include "tensor.h"
#include <memory>
#include <string>
#include <stdexcept>
#include <map>

namespace networks
{

// FCSequence template class
template<uint32_t nBlocks>
class FCSequence : public NodeGroup
{
    uint32_t dims[nBlocks + 1];
    std::unique_ptr<FullyConnectedNode> blocks[nBlocks];

public:
    FCSequence(const uint32_t(&channels)[nBlocks + 1])
    {
        for (uint32_t i = 0; i <= nBlocks; ++i)
            dims[i] = channels[i];

        for (uint32_t i = 0; i < nBlocks; ++i)
            blocks[i] = std::make_unique<FullyConnectedNode>(dims[i], dims[i + 1]);

        for (uint32_t i = 0; i < nBlocks - 1; ++i)
            *blocks[i] - *blocks[i + 1];

        defineSlot("in0", blocks[0]->slot("in0"));
        defineSlot("out0", blocks[nBlocks - 1]->slot("out0"));
    }

    Tensor& operator[](const std::string& name)
    {
        for (uint32_t i = 0; i < nBlocks; ++i)
        {
            const std::string prefix = "fc" + std::to_string(i) + ".";
            if (name.compare(0, prefix.length(), prefix) == 0)
                return (*blocks[i])[name.substr(prefix.length())];
        }
        throw std::runtime_error("No such layer in FCSequence: " + name);
    }
};
template<std::size_t N>
FCSequence(const uint32_t (&)[N]) -> FCSequence<N - 1>;


// FCBNNode - FullyConnected + BatchNorm + ReLU
// Single block matching PointNet paper FC layer structure
class FCBNNode : public NodeGroup
{
    FullyConnectedNode fc;
    ReShapeNode reshape_in;   // [C] -> [1, C] for BatchNorm
    BatchNorm1DNode bn;
    ReShapeNode reshape_out;  // [1, C] -> [C]
    ReluNode relu;

public:
    FCBNNode(uint32_t inDim, uint32_t outDim)
    : fc(inDim, outDim)
    , reshape_in({1, outDim})   // Reshape FC output to [1, outDim]
    , bn(outDim)                // BatchNorm needs [N, C] format
    , reshape_out({outDim})     // Reshape back to [outDim]
    , relu()
    {
        // Connect: FC → Reshape → BN → Reshape → ReLU
        fc - reshape_in - bn - reshape_out - relu;
        
        // Define input/output slots only
        defineSlot("in0", fc.slot("in0"));
        defineSlot("out0", relu.slot("out0"));
    }

    Tensor& operator[](const std::string& name)
    {
        // Route to appropriate sub-node based on parameter name
        // FC parameters: weight, bias
        // BN parameters: mean, var, gamma, beta
        if (name == "weight" || name == "bias")
            return fc[name];
        if (name == "mean" || name == "var" || name == "gamma" || name == "beta")
            return bn[name];
        
        throw std::runtime_error("Unknown parameter in FCBNNode: " + name);
    }
};


// FCBNSequence - Sequence of FC+BN+ReLU blocks (last block has no BN+ReLU)
template<uint32_t nBlocks>
class FCBNSequence : public NodeGroup
{
    uint32_t dims[nBlocks + 1];
    std::unique_ptr<FCBNNode> blocks[nBlocks - 1];  // All but last
    std::unique_ptr<FullyConnectedNode> lastBlock;  // Last block (no BN+ReLU)

public:
    FCBNSequence(const uint32_t(&channels)[nBlocks + 1])
    {
        for (uint32_t i = 0; i <= nBlocks; ++i)
            dims[i] = channels[i];

        // Create FC+BN+ReLU blocks (all except last)
        for (uint32_t i = 0; i < nBlocks - 1; ++i)
            blocks[i] = std::make_unique<FCBNNode>(dims[i], dims[i + 1]);
        
        // Last block: FC only (no BN, no ReLU)
        lastBlock = std::make_unique<FullyConnectedNode>(dims[nBlocks - 1], dims[nBlocks]);

        // Connect blocks
        for (uint32_t i = 0; i < nBlocks - 2; ++i)
            *blocks[i] - *blocks[i + 1];
        
        if (nBlocks > 1)
            *blocks[nBlocks - 2] - *lastBlock;

        defineSlot("in0", blocks[0]->slot("in0"));
        defineSlot("out0", lastBlock->slot("out0"));
    }

    Tensor& operator[](const std::string& name)
    {
        // Check FC+BN blocks (blocks 0 to nBlocks-2)
        for (uint32_t i = 0; i < nBlocks - 1; ++i)
        {
            const std::string prefix = "block" + std::to_string(i) + ".";
            if (name.compare(0, prefix.length(), prefix) == 0)
                return (*blocks[i])[name.substr(prefix.length())];
        }
        
        // Check last FC block (FullyConnectedNode - only has "weight" and "bias" slots)
        if (name.compare(0, 10, "lastBlock.") == 0) {
            return static_cast<Node&>(*lastBlock)[name.substr(10)];  // "lastBlock." length = 10
        }
        
        throw std::runtime_error("No such layer in FCBNSequence: " + name);
    }
};
template<std::size_t N>
FCBNSequence(const uint32_t (&)[N]) -> FCBNSequence<N - 1>;


// MLPSequence template class
template<uint32_t nBlocks>
class MLPSequence : public NodeGroup
{
    uint32_t dims[nBlocks + 1];
    std::unique_ptr<PointWiseMLPNode> blocks[nBlocks];

public:
    MLPSequence(const uint32_t(&channels)[nBlocks + 1])
    {
        for (uint32_t i = 0; i <= nBlocks; ++i)
            dims[i] = channels[i];

        for (uint32_t i = 0; i < nBlocks; ++i)
            blocks[i] = std::make_unique<PointWiseMLPNode>(dims[i], dims[i + 1]);

        for (uint32_t i = 0; i < nBlocks - 1; ++i)
            *blocks[i] - *blocks[i + 1];

        defineSlot("in0", blocks[0]->slot("in0"));
        defineSlot("out0", blocks[nBlocks - 1]->slot("out0"));
    }

    Tensor& operator[](const std::string& name)
    {
        for (uint32_t i = 0; i < nBlocks; ++i)
        {
            const std::string prefix = "mlp" + std::to_string(i) + ".";
            if (name.compare(0, prefix.length(), prefix) == 0)
                return (*blocks[i])[name.substr(prefix.length())];
        }
        throw std::runtime_error("No such layer in MLPSequence: " + name);
    }
};
template<std::size_t N>
MLPSequence(const uint32_t (&)[N]) -> MLPSequence<N - 1>;


// TNetBlock class - Transformation Matrix Generator
// Simplified: Only generates transformation matrix (MatMul separated)
// Input: [N, K] point cloud
// Output: [K, K] transformation matrix
class TNetBlock : public NodeGroup
{
    uint32_t K;
    
    // Transformation matrix generation pipeline
    MLPSequence<3> mlp;              // [N, K] → [N, 64] → [N, 128] → [N, 1024]
    MaxPooling1DNode maxpool;        // [N, 1024] → [1024]
    FCBNSequence<3> fc;              // [1024] → [512] → [256] → [K²]
    ReShapeNode reshape_matrix;      // [K²] → [K, K]
    AddIdentityNode add_identity;    // [K, K] + I → [K, K]

public:
    TNetBlock(uint32_t inputDim)
    : K(inputDim)
    , mlp({inputDim, 64, 128, 1024})
    , maxpool()
    , fc({1024, 512, 256, K * K})
    , reshape_matrix({K, K})
    , add_identity()
    {
        // Simple linear pipeline: Generate transformation matrix only
        // Input [N, K] → MLP → MaxPool → FC → Reshape → AddIdentity → Output [K, K]
        
        
        // Connect pipeline
        mlp - maxpool - fc - reshape_matrix - add_identity;
        
        defineSlot("in0", mlp.slot("in0"));  // Input: point cloud [N, K]
        defineSlot("out0", add_identity.slot("out0"));  // Output: transform matrix [K, K]
    }

    Tensor& operator[](const std::string& name)
    {
        // Route to appropriate sub-network
        if (name.compare(0, 4, "mlp.") == 0)
            return mlp[name.substr(4)];   // "mlp.mlp0.weight" → mlp["mlp0.weight"]
        if (name.compare(0, 3, "fc.") == 0)
            return fc[name.substr(3)];    // "fc.block0.weight" → fc["block0.weight"]
        
        throw std::runtime_error("Unknown parameter in TNetBlock: " + name);
    }
};


// PointNetEncoder class
// Input: [N, 3] point cloud
// Output: [N, 1024] point-wise features
class PointNetEncoder : public NodeGroup
{
    IdentityNode split1;       // Split input for TNet1 dual-path
    IdentityNode split2;       // Split MLP1 output for TNet2 dual-path
    
public:  // Make public for direct access
    TNetBlock tnet1;           // input transform: generates 3x3 matrix
    MatMulNode matmul1;        // apply transform: [N,3] @ [3,3] → [N,3]
    MLPSequence<2> mlp1;       // (3 → 64 → 64)
    TNetBlock tnet2;           // feature transform: generates 64x64 matrix
    MatMulNode matmul2;        // apply transform: [N,64] @ [64,64] → [N,64]
    MLPSequence<2> mlp2;       // (64 → 128 → 1024)

public:
    PointNetEncoder()
    : NodeGroup()
    , split1()
    , split2()
    , tnet1(3)
    , matmul1()
    , mlp1({3, 64, 64})
    , tnet2(64)
    , matmul2()
    , mlp2({64, 128, 1024})
    {
        // Full PointNet Encoder Architecture with IdentityNode for signal splitting:
        //
        // Input [N, 3]
        //   ↓
        // Split1 (IdentityNode)
        //   ├→ out0 → TNet1 → [3,3] matrix → MatMul1.in1
        //   └→ out1 ─────────────────────→ MatMul1.in0
        //                                     ↓
        //   MatMul1 output [N, 3] → MLP1 → [N, 64]
        //                            ↓
        //                         Split2 (IdentityNode)
        //   ├→ out0 → TNet2 → [64,64] matrix → MatMul2.in1
        //   └→ out1 ──────────────────────→ MatMul2.in0
        //                                     ↓
        //   MatMul2 output [N, 64] → MLP2 → [N, 1024]
        
        // Single external input - much cleaner!
        defineSlot("in0", split1.slot("in0"));
        
        // TNet1 path: Split input to TNet and MatMul
        // MatMul expects: in0=[N,K] (points), in1=[K,K] (transform matrix)
        split1 / "out0" - "in0" / matmul1;  // Input → MatMul1.in0 (points [N,3])
        split1 / "out1" - tnet1 - "in1" / matmul1;  // Input → TNet1 → MatMul1.in1 (matrix [3,3])
        matmul1 - mlp1;
        // MLP1: Point-wise features
        
        mlp1 - split2;
        // TNet2 path: Same pattern
        split2 / "out0" - "in0" / matmul2;  // MLP1 output → MatMul2.in0 (features [N,64])
        split2 / "out1" - tnet2 - "in1" / matmul2;  // MLP1 output → TNet2 → MatMul2.in1 (matrix [64,64])
        // MLP2: Final features
        matmul2 - mlp2;
        
        defineSlot("out0", mlp2.slot("out0"));  // Output
    }

    Tensor& operator[](const std::string& name)
    {
        if (name.compare(0, 6, "tnet1.") == 0) return tnet1[name.substr(6)]; // Remove "tnet1." prefix
        if (name.compare(0, 5, "mlp1.") == 0)  return mlp1[name.substr(5)];  // Remove "mlp1." prefix
        if (name.compare(0, 6, "tnet2.") == 0) return tnet2[name.substr(6)]; // Remove "tnet2." prefix
        if (name.compare(0, 5, "mlp2.") == 0)  return mlp2[name.substr(5)];  // Remove "mlp2." prefix

        throw std::runtime_error("Unknown parameter: " + name);
    }
};

// PointNetSegment class - Proper implementation following the paper
class PointNetSegment : public NeuralNet
{
    PointNetEncoder encoder;    // PointNet encoder: [N, 3] → [N, 1024]
    MaxPooling1DNode maxpool;   // Global max pooling: [N, 1024] → [1024]
    ReShapeNode reshape_global; // Reshape: [1024] → [1, 1024]
    BroadcastNode broadcast;    // Broadcast global feature: [1, 1024] → [N, 1024]
    ConcatNode concat;          // Concatenate: [N, 1024] + [N, 1024] → [N, 2048]
    MLPSequence<3> segHead;     // Segmentation head: [N, 2048] → [N, 512] → [N, 256] → [N, numClasses]

    uint32_t numClasses;

public:
    using u_ptr = std::unique_ptr<PointNetSegment>;

    PointNetSegment(Device& device, uint32_t numClasses)
    : NeuralNet(device, 1, 1)
    , numClasses(numClasses)
    , encoder()
    , maxpool()
    , reshape_global({1, 1024})  // Reshape [1024] → [1, 1024]
    , broadcast()
    , concat()
    , segHead({2048, 512, 256, numClasses})  // 1024 (point feature) + 1024 (global feature) = 2048
    {
        // PointNet Segmentation Architecture (following the paper):
        //
        // Input [N, 3]
        //   ↓
        // encoder → [N, 1024] (point-wise features)
        //   ├───→ maxpool → [1, 1024] (global feature)
        //   │         ↓
        //   │    broadcast → [N, 1024] (replicated global)
        //   │         ↓
        //   └───→ concat → [N, 2048] (point + global)
        //             ↓
        //         segHead → [N, numClasses]
        //             ↓
        //         output
        
        // 1. Input → Encoder
        input(0) - encoder;
        
        // 2. Split encoder output for two paths:
        //    Path A: encoder → concat.in0 (point features [N, 1024])
        //    Path B: encoder → maxpool → reshape → broadcast → concat.in1 (global feature [N, 1024])
        encoder - concat;                                  // Point features to concat.in0
        encoder - maxpool - reshape_global - broadcast;    // Global [1,C] to broadcast.in0
        encoder - "in1" / broadcast;                       // Point features [N,C] to broadcast.in1 (shape reference)
        broadcast - "in1" / concat;                        // Broadcasted global to concat.in1
        
        // 3. Segmentation head
        concat - segHead - output(0);
    }

    Tensor& operator[](const std::string& name)
    {
        if (name.compare(0, 8, "encoder.") == 0)
            return encoder[name.substr(8)];  // "encoder." 제거
        if (name.compare(0, 8, "segHead.") == 0)
            return segHead[name.substr(8)];  // "segHead." 제거
        throw std::runtime_error("Unknown parameter: " + name);
    }
    
    // Public accessors for testing
    PointNetEncoder& getEncoder() { return encoder; }
    MLPSequence<3>& getSegHead() { return segHead; }
};


} // namespace networks

#endif // POINTNET_HPP


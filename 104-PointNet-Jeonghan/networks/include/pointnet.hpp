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
    : NodeGroup()
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
    : NodeGroup()
    , fc(inDim, outDim)
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
    : NodeGroup()
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
    : NodeGroup()
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
    : NodeGroup()
    , K(inputDim)
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


// PointNetEncoder class (yanx27 structure)
// Input: [N, channel] point cloud (channel = 3 or 6 for normal_channel)
// Output: [N, 1024] point-wise features
class PointNetEncoder : public NodeGroup
{
    uint32_t channel;          // Input channel dimension (3 or 6)
    IdentityNode split1;       // Split input for STN dual-path
    IdentityNode split2;       // Split conv1 output for FSTN dual-path
    
public:  // Make public for direct access
    TNetBlock stn;             // input transform: generates channel x channel matrix (STN3d)
    MatMulNode matmul1;        // apply transform: [N, channel] @ [channel, channel] → [N, channel]
    MLPSequence<1> conv1;      // (channel → 64) - first convolution with ReLU
    TNetBlock fstn;            // feature transform: generates 64x64 matrix (STNkd)
    MatMulNode matmul2;        // apply transform: [N,64] @ [64,64] → [N,64]
    MLPSequence<1> conv2;      // (64 → 128) - second convolution with ReLU
    PointWiseConvNode conv3;   // (128 → 1024) - third convolution (Conv+BN, NO ReLU!)

public:
    PointNetEncoder(uint32_t channel = 3)
    : NodeGroup()
    , channel(channel)
    , split1()
    , split2()
    , stn(channel)             // STN3d with channel-dim input
    , matmul1()
    , conv1({channel, 64})     // First conv layer with ReLU
    , fstn(64)                 // STNkd with 64-dim feature
    , matmul2()
    , conv2({64, 128})         // Second conv layer with ReLU
    , conv3(128, 1024)         // Third conv layer (Conv+BN, NO ReLU)
    {
        // yanx27 PointNet Encoder Architecture:
        //
        // Input [N, channel] (channel = 3: x,y,z  OR  6: x,y,z + normals)
        //   ↓
        // Split1 (IdentityNode)
        //   ├→ out0 → STN → [channel, channel] matrix → MatMul1.in1
        //   └→ out1 ───────────────────────────────→ MatMul1.in0
        //                                              ↓
        //   MatMul1 output [N, channel] → Conv1 → [N, 64] (pointfeat)
        //                                  ↓
        //                              Split2 (IdentityNode)
        //   ├→ out0 → FSTN → [64,64] matrix → MatMul2.in1
        //   └→ out1 ───────────────────────→ MatMul2.in0
        //                                     ↓
        //   MatMul2 output [N, 64] → Conv2-3 → [N, 1024]
        //   ↓
        // Output: [N, 1024] features
        
        // Single external input
        defineSlot("in0", split1.slot("in0"));
        
        // Path 1: STN (channel x channel transformation)
        split1 / "out0" - "in0" / matmul1;        // Input → MatMul1.in0 [N, channel]
        split1 / "out1" - stn - "in1" / matmul1;  // Input → STN → MatMul1.in1 [channel, channel]
        
        // Apply first convolution
        matmul1 - conv1;  // [N, channel] → [N, 64]
        
        // Path 2: FSTN (64x64 transformation) on conv1 output
        conv1 - split2;
        split2 / "out0" - "in0" / matmul2;         // Conv1 output → MatMul2.in0 [N,64]
        split2 / "out1" - fstn - "in1" / matmul2;  // Conv1 output → FSTN → MatMul2.in1 [64,64]
        
        // Apply remaining convolutions
        matmul2 - conv2 - conv3;  // [N,64] → [N,128] → [N,1024]
        
        // Dual outputs:
        // out0: pointfeat [N, 64] from conv1 (for segmentation)
        // out1: full features [N, 1024] from conv3
        defineSlot("out0", conv1.slot("out0"));   // pointfeat [N,64]
        defineSlot("out1", conv3.slot("out0"));   // full features [N,1024]
    }

    Tensor& operator[](const std::string& name)
    {
        // yanx27 key mapping:
        // stn.* → stn.*
        // conv.mlp0.* → conv1["mlp0.*"]  (yanx27 format)
        // conv.mlp1.* → conv2["mlp0.*"]  (yanx27 format)
        // conv.mlp2.* → conv3.*  (yanx27 format)
        // conv1.mlp0.* → conv1["mlp0.*"]  (legacy format)
        // conv2.mlp0.* → conv2["mlp0.*"]  (legacy format)
        // conv3.* → conv3.*  (legacy format)
        // fstn.* → fstn.*
        if (name.compare(0, 4, "stn.") == 0)   return stn[name.substr(4)];
        
        // yanx27 format: conv.mlp0-2.*
        if (name.compare(0, 10, "conv.mlp0.") == 0) {
            // conv.mlp0.* → conv1["mlp0.*"]
            return conv1[name.substr(5)];  // Remove "conv." prefix, keep "mlp0.*"
        }
        if (name.compare(0, 10, "conv.mlp1.") == 0) {
            // conv.mlp1.* → conv2["mlp0.*"]
            return conv2["mlp0" + name.substr(9)];  // Remove "conv.mlp1", replace with "mlp0"
        }
        if (name.compare(0, 10, "conv.mlp2.") == 0) {
            // conv.mlp2.* → conv3.*
            return conv3[name.substr(10)];  // Remove "conv.mlp2." prefix
        }
        
        // Legacy format: conv1-3.*
        if (name.compare(0, 6, "conv1.") == 0) {
            return conv1[name.substr(6)];
        }
        if (name.compare(0, 6, "conv2.") == 0) {
            return conv2[name.substr(6)];
        }
        if (name.compare(0, 6, "conv3.") == 0) {
            return conv3[name.substr(6)];
        }
        
        if (name.compare(0, 5, "fstn.") == 0)  return fstn[name.substr(5)];

        throw std::runtime_error("Unknown parameter: " + name);
    }
};

// PointNetSegment class - yanx27 semantic segmentation structure
class PointNetSegment : public NeuralNet
{
    PointNetEncoder encoder;    // PointNet encoder: [N, 9] → [N, 1024]
    MaxPooling1DNode maxpool;   // Global max pooling: [N, 1024] → [1024]
    ReShapeNode reshape_global; // Reshape: [1024] → [1, 1024]
    BroadcastNode broadcast;    // Broadcast global feature: [1, 1024] → [N, 1024]
    ConcatNode concat;          // Concatenate: [N, 64] + [N, 1024] → [N, 1088]
    MLPSequence<4> segHead;     // Segmentation head: [N, 1088] → [N, 512] → [N, 256] → [N, 128] → [N, numClasses]

    uint32_t numClasses;

public:
    using u_ptr = std::unique_ptr<PointNetSegment>;

    PointNetSegment(Device& device, uint32_t numClasses, uint32_t channel = 3)
    : NeuralNet(device, 1, 1)
    , numClasses(numClasses)
    , encoder(channel)
    , maxpool()
    , reshape_global({1, 1024})  // Reshape [1024] → [1, 1024]
    , broadcast()
    , concat()
    , segHead({1088, 512, 256, 128, numClasses})  // 64 (local feat) + 1024 (global) = 1088
    {
        // yanx27 Semantic Segmentation Architecture:
        //
        // Input [N, 9] (x,y,z + RGB + normalized coords)
        //   ↓
        // encoder → [N, 1024] features
        //   ├───→ maxpool → [1024] (global feature)
        //   │         ↓
        //   │    reshape → [1, 1024]
        //   │         ↓
        //   │    broadcast → [N, 1024] (replicated global)
        //   │
        //   │  NOTE: yanx27 concatenates [pointfeat(64) + global(1024)] = 1088
        //   │        pointfeat is conv1 output (after first MLP layer)
        //   │        For now we use full encoder output [N, 1024]
        //   │
        //   └───→ concat → [N, 1088] (local 64 + global 1024)
        //             ↓
        //         Conv1d(1088→512) + BN + ReLU
        //         Conv1d(512→256) + BN + ReLU
        //         Conv1d(256→128) + BN + ReLU
        //         Conv1d(128→13)
        //             ↓
        //         output [N, 13]
        
        // 1. Input → Encoder (produces two outputs)
        input(0) - encoder;
        
        // 2. Split encoder outputs:
        //    Path A: encoder.out0 (pointfeat [N,64]) → concat.in0
        //    Path B: encoder.out1 (full [N,1024]) → maxpool → reshape → broadcast → concat.in1
        
        // Path A: Local features [N, 64] directly to concat
        encoder / "out0" - "in0" / concat;                 // pointfeat [N,64] to concat.in0
        
        // Path B: Global features [N, 1024] → pooled and broadcasted
        encoder / "out1" - maxpool;                        // [N,1024] → [1024]
        maxpool - reshape_global;                          // [1024] → [1,1024]
        reshape_global - broadcast;                        // [1,1024] to broadcast.in0
        encoder / "out1" - "in1" / broadcast;              // [N,1024] shape reference for broadcast
        broadcast - "in1" / concat;                        // Broadcasted [N,1024] to concat.in1
        
        // 3. Segmentation head: [N,1088] → [N,512] → [N,256] → [N,128] → [N,numClasses]
        concat - segHead - output(0);
    }

    Tensor& operator[](const std::string& name)
    {
        // yanx27 key mapping:
        // feat.* → encoder.*
        // conv1-4.* → segHead.mlp0-3.*
        if (name.compare(0, 5, "feat.") == 0)
            return encoder[name.substr(5)];  // feat.stn.* → encoder.stn.*
        
        // Segmentation head: conv1-4.* → segHead.mlp0-3.*
        // Note: conv1.weight, conv1.bn_bias, etc. all map to mlp0.*
        if (name.size() >= 6 && name.compare(0, 4, "conv") == 0 && 
            name[4] >= '1' && name[4] <= '4' && name[5] == '.') {
            int idx = name[4] - '1';  // conv1 → 0, conv2 → 1, etc.
            return segHead["mlp" + std::to_string(idx) + name.substr(5)];
        }
        
        throw std::runtime_error("Unknown parameter: " + name);
    }
    
    // Public accessors for testing
    PointNetEncoder& getEncoder() { return encoder; }
    MLPSequence<4>& getSegHead() { return segHead; }
};


} // namespace networks

#endif // POINTNET_HPP


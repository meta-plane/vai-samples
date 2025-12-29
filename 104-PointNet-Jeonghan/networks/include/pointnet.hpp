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
    , reshape_in({outDim, 1})   // Reshape FC output to [C, 1] (PyTorch convention)
    , bn(outDim)                // BatchNorm needs [C, N] format
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
// Input: [N, inputDim] point cloud
// Output: [outDim, outDim] transformation matrix
class TNetBlock : public NodeGroup
{
    uint32_t outDim;
    
    // Transformation matrix generation pipeline
    MLPSequence<3> mlp;              // [N, inputDim] → [N, 64] → [N, 128] → [N, 1024]
    MaxPooling1DNode maxpool;        // [N, 1024] → [1024]
    FCBNSequence<3> fc;              // [1024] → [512] → [256] → [outDim²]
    ReShapeNode reshape_matrix;      // [outDim²] → [outDim, outDim]
    AddIdentityNode add_identity;    // [outDim, outDim] + I → [outDim, outDim]

public:
    // inputDim: input feature channels
    // outDim: output matrix size (default = inputDim)
    TNetBlock(uint32_t inputDim, uint32_t outDim = 0)
    : NodeGroup()
    , outDim(outDim == 0 ? inputDim : outDim)
    , mlp({inputDim, 64, 128, 1024})
    , maxpool()
    , fc({1024, 512, 256, this->outDim * this->outDim})
    , reshape_matrix({this->outDim, this->outDim})
    , add_identity()
    {
        // Generate transformation matrix
        // Input [N, inputDim] → MLP → MaxPool → FC → Reshape → AddIdentity → Output [outDim, outDim]
        
        // Connect pipeline
        mlp - maxpool - fc - reshape_matrix - add_identity;
        
        defineSlot("in0", mlp.slot("in0"));  // Input: point cloud [N, inputDim]
        defineSlot("out0", add_identity.slot("out0"));  // Output: transform matrix [outDim, outDim]
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
    uint32_t channel;          // Input channel dimension (3, 6, or 9)
    
    IdentityNode split_for_stn;    // Split input: one for STN, one for transform
    IdentityNode split_for_slice;  // Split input: one for xyz, one for rest
    SliceNode slice_xyz;           // Extract xyz [0:3]
    SliceNode slice_rest;          // Extract rest [3:channel] if channel > 3
    ConcatNode concat_after_transform; // Concat transformed xyz + rest
    IdentityNode split2;           // Split conv1 output for FSTN dual-path
    
public:  // Make public for direct access
    TNetBlock stn;             // input transform: channel → 3x3 matrix (STN3d)
    MatMulNode matmul1;        // apply transform: [N, 3] @ [3, 3] → [N, 3]
    MLPSequence<1> conv1;      // (channel → 64) - first convolution with ReLU
    TNetBlock fstn;            // feature transform: generates 64x64 matrix (STNkd)
    MatMulNode matmul2;        // apply transform: [N,64] @ [64,64] → [N,64]
    MLPSequence<1> conv2;      // (64 → 128) - second convolution with ReLU
    PointWiseConvNode conv3;   // (128 → 1024) - third convolution (Conv+BN, NO ReLU!)

public:
    PointNetEncoder(uint32_t channel = 3)
    : NodeGroup()
    , channel(channel)
    , split_for_stn()
    , split_for_slice()
    , slice_xyz(0, 3)                      // Extract xyz [0:3]
    , slice_rest(3, channel > 3 ? channel : 4)  // Extract rest [3:channel]
    , concat_after_transform()
    , split2()
    , stn(channel, 3)                      // STN3d: full channel input → 3x3 matrix
    , matmul1()
    , conv1({channel, 64})             // First conv sequence: channel→64→64
    , fstn(64, 64)                         // STNkd: 64-dim input → 64x64 matrix
    , matmul2()
    , conv2({64, 128})                 // Second conv sequence: 64→64→128
    , conv3(128, 1024)                     // Third conv layer (Conv+BN, NO ReLU)
    {
        // yanx27 PointNet Encoder Architecture (exact implementation):
        //
        // Input [N, channel]
        //   ↓
        // split_for_stn
        //   ├─ out0 → STN (전체 channel 입력) → [3, 3] matrix ──────┐
        //   │                                                        │
        //   └─ out1 → split_for_slice                                │
        //             ├─ out0 → slice_xyz [N, 3] ─────────────────┤─ MatMul → [N, 3]
        //             │                                             │
        //             └─ out1 → slice_rest [N, channel-3] ─────────────────┐
        //                                                                   │
        //                                      MatMul output [N, 3] ───────┤─ Concat
        //                                                                   │
        //                                              [N, channel-3] ──────┘
        //                                                     ↓
        //                                              [N, channel] → Conv1
        
        // Single external input
        defineSlot("in0", split_for_stn.slot("in0"));
        
        if (channel > 3) {
            // For channel > 3: STN uses full channel, but transform only xyz
            split_for_stn / "out0" - stn - "in0" / matmul1;  // Full channel → STN → [3,3] → MatMul.in0
            split_for_stn / "out1" - split_for_slice;        // Full channel → split
            
            split_for_slice / "out0" - slice_xyz - "in1" / matmul1;  // xyz → MatMul.in1 [N,3]
            split_for_slice / "out1" - slice_rest - "in1" / concat_after_transform;  // rest → Concat
            
            matmul1 - "in0" / concat_after_transform;  // transformed xyz → Concat
            concat_after_transform - conv1;            // [N, channel] → Conv1
        } else {
            // For channel = 3: direct transformation
            split_for_stn / "out0" - "in1" / matmul1;        // Input → MatMul.in1 [N,3]
            split_for_stn / "out1" - stn - "in0" / matmul1;  // Input → STN → MatMul.in0 [3,3]
            matmul1 - conv1;
        }
        
        // Path 2: FSTN (64x64 transformation) on conv1 output
        conv1 - split2;
        split2 / "out0" - "in1" / matmul2;         // Conv1 output → MatMul2.in1 [N,64]
        split2 / "out1" - fstn - "in0" / matmul2;  // Conv1 output → FSTN → MatMul2.in0 [64,64]
        
        // Apply remaining convolutions
        matmul2 - conv2 - conv3;  // [N,64] → [N,128] → [N,1024]

        // Dual outputs (yanx27 structure):
        // out0: pointfeat [N, 64] from matmul2 (AFTER FSTN transform!)
        // out1: full features [N, 1024] from conv3
        defineSlot("out0", matmul2.slot("out0"));  // pointfeat [N,64] - AFTER FSTN!
        defineSlot("out1", conv3.slot("out0"));    // full features [N,1024]
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
    MaxPooling1DNode maxpool;   // Global max pooling: [1024, N] → [1024]
    ReShapeNode reshape_global; // Reshape: [1024] → [1024, 1]
    BroadcastNode broadcast;    // Broadcast global feature: [1024, 1] → [1024, N]
    ConcatNode concat;          // Concatenate: [N, 64] + [N, 1024] → [N, 1088]
    MLPSequence<3> segHead;     // Segmentation head: conv1-3 with BN+ReLU
    PointWiseLinearNode conv4;  // Final layer: conv4 (NO BN, NO ReLU) - matches PyTorch

    uint32_t numClasses;

public:
    using u_ptr = std::unique_ptr<PointNetSegment>;

    PointNetSegment(Device& device, uint32_t numClasses, uint32_t channel = 3)
    : NeuralNet(device, 1, 1)
    , numClasses(numClasses)
    , encoder(channel)
    , maxpool()
    , reshape_global({1024, 1})  // Reshape [1024] → [1024, 1]
    , broadcast()
    , concat()
    , segHead({1088, 512, 256, 128})  // 64 (local feat) + 1024 (global) = 1088 → 128
    , conv4(128, numClasses)          // Final layer: 128 → numClasses (NO BN, NO ReLU)
    {
        // yanx27 Semantic Segmentation Architecture:
        //
        // Input [N, 9] (x,y,z + RGB + normalized coords)
        //   ↓
        // encoder → [N, 1024] features
        //   ├───→ maxpool → [1024] (global feature)
        //   │         ↓
        //   │    reshape → [1024, 1]
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
        
        // Path A: Global features [1024, N] → pooled and broadcasted → concat.in0
        // (PyTorch order: [global, pointfeat] = [1024, 64])
        encoder / "out1" - maxpool;                        // [1024,N] → [1024]
        maxpool - reshape_global;                          // [1024] → [1024,1]
        reshape_global - broadcast;                        // [1024,1] to broadcast.in0
        encoder / "out1" - "in1" / broadcast;              // [1024,N] shape reference for broadcast
        broadcast - "in0" / concat;                        // Broadcasted [1024,N] to concat.in0

        // Path B: Local features [64, N] directly to concat.in1
        encoder / "out0" - "in1" / concat;                 // pointfeat [64,N] to concat.in1
        
        // 3. Segmentation head: [N,1088] → [N,512] → [N,256] → [N,128] → [N,numClasses]
        // segHead: conv1-3 with BN+ReLU
        // conv4: final layer without BN/ReLU (matches PyTorch)
        concat - segHead - conv4 - output(0);
    }

    Tensor& operator[](const std::string& name)
    {
        // yanx27 key mapping:
        // feat.* → encoder.*
        // conv1-3.* → segHead.mlp0-2.*
        // conv4.* → conv4.*
        if (name.compare(0, 5, "feat.") == 0)
            return encoder[name.substr(5)];  // feat.stn.* → encoder.stn.*

        // Segmentation head: conv1-3.* → segHead.mlp0-2.* (with BN+ReLU)
        if (name.size() >= 6 && name.compare(0, 4, "conv") == 0 &&
            name[4] >= '1' && name[4] <= '3' && name[5] == '.') {
            int idx = name[4] - '1';  // conv1 → 0, conv2 → 1, conv3 → 2
            return segHead["mlp" + std::to_string(idx) + name.substr(5)];
        }

        // conv4.* → conv4 node directly (no BN)
        if (name.compare(0, 6, "conv4.") == 0) {
            return conv4[name.substr(6)];  // conv4.weight → conv4["weight"]
        }

        throw std::runtime_error("Unknown parameter: " + name);
    }

    // Public accessors for testing
    PointNetEncoder& getEncoder() { return encoder; }
    MLPSequence<3>& getSegHead() { return segHead; }
    PointWiseLinearNode& getConv4() { return conv4; }
};


} // namespace networks

#endif // POINTNET_HPP


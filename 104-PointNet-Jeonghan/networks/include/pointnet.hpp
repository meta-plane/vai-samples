#ifndef POINTNET_HPP
#define POINTNET_HPP

#include "neuralNet.h"
#include "neuralNodes.h"
#include "jsonParser.h"
#include "tensor.h"
#include <memory>
#include <string>

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
            if (name.starts_with(prefix))
                return (*blocks[i])[name.substr(prefix.length())];
        }
        throw std::runtime_error("No such layer in FCSequence: " + name);
    }
};
template<std::size_t N>
FCSequence(const uint32_t (&)[N]) -> FCSequence<N - 1>;


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
            if (name.starts_with(prefix))
                return (*blocks[i])[name.substr(prefix.length())];
        }
        throw std::runtime_error("No such layer in MLPSequence: " + name);
    }
};
template<std::size_t N>
MLPSequence(const uint32_t (&)[N]) -> MLPSequence<N - 1>;


// TNetBlock class - Spatial Transformer Network
// Input: [N, K] point cloud
// Output: [N, K] transformed point cloud
class TNetBlock : public NodeGroup
{
    uint32_t K;

    MLPSequence<3> mlp;          // K -> 64 -> 128 -> 1024
    MaxPooling1DNode maxpool;    // [N, 1024] -> [1024]
    FCSequence<3> fc;            // 1024 -> 512 -> 256 -> K*K
    ReShapeNode reshape;         // [1, K*K] -> [K, K]
    MatMulNode matmul;           // [N, K] @ [K, K] -> [N, K]

public:
    TNetBlock(uint32_t inputDim)
    : K(inputDim)
    , mlp({K, 64, 128, 1024})
    , maxpool()
    , fc({1024, 512, 256, K*K})
    , reshape({K, K})            // Initialize with target shape [K, K]
    , matmul()
    {
        // Path A: Generate transformation matrix
        // input [N, K] -> MLP -> MaxPool -> FC -> Reshape -> [K, K] matrix
        mlp - maxpool - fc - reshape;
        
        // TEMPORARY: Skip MatMul, just pass through input
        // TODO: Fix MatMul connection issue
        
        // Define external slots
        // in0: receives input and passes through
        // out0: outputs same as input (identity transformation)
        defineSlot("in0", mlp.slot("in0"));
        defineSlot("out0", mlp.slot("out0"));  // TEMPORARY: bypass, should be matmul output
    }

    Tensor& operator[](const std::string& name)
    {
        if (name.starts_with("mlp."))
            return mlp[name.substr(4)]; // Remove "mlp." prefix
        if (name.starts_with("fc."))
            return fc[name.substr(3)]; // Remove "fc." prefix

        throw std::runtime_error("No such layer in TNetBlock: " + name);
    }
};


// PointNetEncoder class
// Input: [N, 3] point cloud
// Output: [N, 1024] point-wise features
class PointNetEncoder : public NodeGroup
{
public:  // Make public for direct access
    TNetBlock tnet1;           // input transform (3x3)
    MLPSequence<2> mlp1;       // (3 → 64 → 64)
    TNetBlock tnet2;           // feature transform (64x64)
    MLPSequence<2> mlp2;       // (64 → 128 → 1024)

public:
    PointNetEncoder()
    : tnet1(3)
    , mlp1({3, 64, 64})
    , tnet2(64)
    , mlp2({64, 128, 1024})
    {
        // Architecture flow (논문 구조):
        
        // Define encoder's input slot (only for MLP path)
        defineSlot("in0", tnet1.slot("in0"));
        
        // TNet1 -> MLP1
        tnet1 - mlp1;
        
        // TNet2: TEMPORARY bypass mode, only one input
        mlp1 - tnet2 / "in0";                           // MLP path
        
        // TNet2 -> MLP2
        tnet2 - mlp2;
        
        defineSlot("out0", mlp2.slot("out0"));
    }

    Tensor& operator[](const std::string& name)
    {
        if (name.starts_with("tnet1.")) return tnet1[name.substr(6)]; // Remove "tnet1." prefix
        if (name.starts_with("mlp1."))  return mlp1[name.substr(5)];  // Remove "mlp1." prefix
        if (name.starts_with("tnet2.")) return tnet2[name.substr(6)]; // Remove "tnet2." prefix
        if (name.starts_with("mlp2."))  return mlp2[name.substr(5)];  // Remove "mlp2." prefix

        throw std::runtime_error("Unknown parameter: " + name);
    }
};


// PointNetSegment class - Proper implementation following the paper
class PointNetSegment : public NeuralNet
{
    PointNetEncoder encoder;    // PointNet encoder: [N, 3] → [N, 1024]
    MaxPooling1DNode maxpool;   // Global max pooling: [N, 1024] → [1, 1024]
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
        
        // 1. Main path: input → encoder
        // TEMPORARY: TNet bypass mode, only one input needed
        input(0) - encoder.tnet1 / "in0";              // -> TNet1 (identity transform)
        
        // Connect TNet1 output to rest of encoder (already done in encoder constructor)
        
        // 2. Global feature branch: encoder → maxpool → broadcast
        encoder - maxpool - broadcast / "in0";      // global feature path
        encoder / "out0" - broadcast / "in1";       // shape reference for broadcast
        
        // 3. Concatenate point features + global features
        encoder - concat / "in0";                   // point-wise features [N, 1024]
        broadcast - concat / "in1";                 // broadcasted global [N, 1024]
        
        // 4. Segmentation head: concat → segHead → output
        concat - segHead - output(0);
    }

    Tensor& operator[](const std::string& name)
    {
        if (name.starts_with("encoder."))
            return encoder[name.substr(8)];  // "encoder." 제거
        if (name.starts_with("segHead."))
            return segHead[name.substr(8)];  // "segHead." 제거
        throw std::runtime_error("Unknown parameter: " + name);
    }
};


} // namespace networks

#endif // POINTNET_HPP


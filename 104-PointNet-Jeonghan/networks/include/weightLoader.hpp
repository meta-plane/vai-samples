#ifndef WEIGHT_LOADER_HPP
#define WEIGHT_LOADER_HPP

/**
 * WeightLoader - PyTorch state_dict를 Vulkan 네트워크에 로딩
 *
 * PyTorch 키를 직접 사용하여 변환 스크립트의 복잡도를 제거.
 * SafeTensors 파일은 Conv1d squeeze만 수행된 상태로 가정.
 */

#include "safeTensorsParser.h"
#include "tensor.h"
#include "pointnet.hpp"
#include <string>
#include <iostream>

namespace networks {

class WeightLoader {
    const SafeTensorsParser& weights;
    bool verbose;

public:
    WeightLoader(const SafeTensorsParser& w, bool verbose = true)
        : weights(w), verbose(verbose) {}

    // =========================================
    // High-level API: 네트워크 구조 단위 로딩
    // =========================================

    /**
     * TNetBlock 로딩 (STN3d or STNkd)
     *
     * PyTorch 구조:
     *   {prefix}.conv1/2/3 + {prefix}.bn1/2/3  (MLP layers)
     *   {prefix}.fc1/2 + {prefix}.bn4/5        (FC layers with BN)
     *   {prefix}.fc3                           (FC layer without BN)
     *
     * Vulkan 구조:
     *   tnet.mlp.mlp0/1/2     (MLPSequence<3>)
     *   tnet.fc.block0/1      (FCBNSequence<3>)
     *   tnet.fc.lastBlock     (FullyConnectedNode)
     */
    void loadTNet(TNetBlock& tnet, const std::string& prefix) {
        if (verbose) std::cout << "  Loading TNet: " << (prefix.empty() ? "(root)" : prefix) << "\n";

        // Handle prefix: add "." only if prefix is not empty
        std::string p = prefix.empty() ? "" : prefix + ".";

        // MLP: conv1/2/3 + bn1/2/3 → mlp.mlp0/1/2
        for (int i = 0; i < 3; i++) {
            std::string convKey = p + "conv" + std::to_string(i+1);
            std::string bnKey = p + "bn" + std::to_string(i+1);
            std::string vulkanKey = "mlp.mlp" + std::to_string(i);

            tnet[vulkanKey + ".weight"] = Tensor(weights[convKey + ".weight"]);
            tnet[vulkanKey + ".bias"] = Tensor(weights[convKey + ".bias"]);
            tnet[vulkanKey + ".bn_mean"] = Tensor(weights[bnKey + ".running_mean"]);
            tnet[vulkanKey + ".bn_var"] = Tensor(weights[bnKey + ".running_var"]);
            tnet[vulkanKey + ".bn_gamma"] = Tensor(weights[bnKey + ".weight"]);
            tnet[vulkanKey + ".bn_beta"] = Tensor(weights[bnKey + ".bias"]);
        }

        // FC: fc1/2 + bn4/5 → fc.block0/1
        for (int i = 0; i < 2; i++) {
            std::string fcKey = p + "fc" + std::to_string(i+1);
            std::string bnKey = p + "bn" + std::to_string(i+4);
            std::string vulkanKey = "fc.block" + std::to_string(i);

            tnet[vulkanKey + ".weight"] = Tensor(weights[fcKey + ".weight"]);
            tnet[vulkanKey + ".bias"] = Tensor(weights[fcKey + ".bias"]);
            tnet[vulkanKey + ".mean"] = Tensor(weights[bnKey + ".running_mean"]);
            tnet[vulkanKey + ".var"] = Tensor(weights[bnKey + ".running_var"]);
            tnet[vulkanKey + ".gamma"] = Tensor(weights[bnKey + ".weight"]);
            tnet[vulkanKey + ".beta"] = Tensor(weights[bnKey + ".bias"]);
        }

        // FC: fc3 (no BN) → fc.lastBlock
        std::string fc3Key = p + "fc3";
        tnet["fc.lastBlock.weight"] = Tensor(weights[fc3Key + ".weight"]);
        tnet["fc.lastBlock.bias"] = Tensor(weights[fc3Key + ".bias"]);
    }

    /**
     * PointNetEncoder 로딩
     *
     * PyTorch 구조:
     *   {prefix}.stn.*         (STN3d)
     *   {prefix}.conv1 + bn1   (channel → 64)
     *   {prefix}.fstn.*        (STNkd, k=64)
     *   {prefix}.conv2 + bn2   (64 → 128)
     *   {prefix}.conv3 + bn3   (128 → 1024, no ReLU)
     */
    void loadEncoder(PointNetEncoder& enc, const std::string& prefix = "") {
        if (verbose) std::cout << "Loading PointNetEncoder...\n";

        std::string p = prefix.empty() ? "" : prefix + ".";

        // STN3d
        loadTNet(enc.stn, p + "stn");

        // Conv1: channel → 64 (MLPSequence<1>)
        loadMLPSequence1(enc.conv1, p + "conv1", p + "bn1");
        if (verbose) std::cout << "  ✓ conv1\n";

        // STNkd (k=64)
        loadTNet(enc.fstn, p + "fstn");

        // Conv2: 64 → 128 (MLPSequence<1>)
        loadMLPSequence1(enc.conv2, p + "conv2", p + "bn2");
        if (verbose) std::cout << "  ✓ conv2\n";

        // Conv3: 128 → 1024 (PointWiseConvNode, no ReLU)
        loadPointWiseConv(enc.conv3, p + "conv3", p + "bn3");
        if (verbose) std::cout << "  ✓ conv3\n";
    }

    /**
     * PointNetSegment 전체 로딩
     *
     * PyTorch 구조:
     *   feat.*                   (PointNetEncoder)
     *   conv1/2/3 + bn1/2/3      (Segmentation head, with BN+ReLU)
     *   conv4                    (Final layer, NO BN, NO ReLU)
     */
    void loadSegment(PointNetSegment& net) {
        if (verbose) std::cout << "Loading PointNetSegment...\n";

        // Encoder
        loadEncoder(net.getEncoder(), "feat");

        // Segmentation head: conv1-3 → segHead.mlp0-2 (with BatchNorm)
        if (verbose) std::cout << "Loading Segmentation Head...\n";
        auto& segHead = net.getSegHead();
        for (int i = 0; i < 3; i++) {
            std::string convKey = "conv" + std::to_string(i+1);
            std::string bnKey = "bn" + std::to_string(i+1);
            std::string vulkanKey = "mlp" + std::to_string(i);

            segHead[vulkanKey + ".weight"] = Tensor(weights[convKey + ".weight"]);
            segHead[vulkanKey + ".bias"] = Tensor(weights[convKey + ".bias"]);
            segHead[vulkanKey + ".bn_mean"] = Tensor(weights[bnKey + ".running_mean"]);
            segHead[vulkanKey + ".bn_var"] = Tensor(weights[bnKey + ".running_var"]);
            segHead[vulkanKey + ".bn_gamma"] = Tensor(weights[bnKey + ".weight"]);
            segHead[vulkanKey + ".bn_beta"] = Tensor(weights[bnKey + ".bias"]);

            if (verbose) std::cout << "  ✓ " << convKey << "\n";
        }

        // conv4 → conv4 node directly (NO BatchNorm - final output layer)
        auto& conv4Node = net.getConv4();
        conv4Node["weight"] = Tensor(weights["conv4.weight"]);
        conv4Node["bias"] = Tensor(weights["conv4.bias"]);
        if (verbose) std::cout << "  ✓ conv4 (no BN)\n";
    }

private:
    // =========================================
    // Low-level helpers
    // =========================================

    /**
     * MLPSequence<1> 로딩 (단일 PointWiseMLPNode)
     */
    template<uint32_t N>
    void loadMLPSequence1(MLPSequence<N>& mlp, const std::string& conv, const std::string& bn) {
        mlp["mlp0.weight"] = Tensor(weights[conv + ".weight"]);
        mlp["mlp0.bias"] = Tensor(weights[conv + ".bias"]);
        mlp["mlp0.bn_mean"] = Tensor(weights[bn + ".running_mean"]);
        mlp["mlp0.bn_var"] = Tensor(weights[bn + ".running_var"]);
        mlp["mlp0.bn_gamma"] = Tensor(weights[bn + ".weight"]);
        mlp["mlp0.bn_beta"] = Tensor(weights[bn + ".bias"]);
    }

    /**
     * PointWiseConvNode 로딩 (Conv + BN, no ReLU)
     * Template version to support both PointWiseConvNode and FusedPointWiseConvNode
     */
    template<typename ConvNodeType>
    void loadPointWiseConv(ConvNodeType& node, const std::string& conv, const std::string& bn) {
        node["weight"] = Tensor(weights[conv + ".weight"]);
        node["bias"] = Tensor(weights[conv + ".bias"]);
        node["bn_mean"] = Tensor(weights[bn + ".running_mean"]);
        node["bn_var"] = Tensor(weights[bn + ".running_var"]);
        node["bn_gamma"] = Tensor(weights[bn + ".weight"]);
        node["bn_beta"] = Tensor(weights[bn + ".bias"]);
    }
};

} // namespace networks

#endif // WEIGHT_LOADER_HPP

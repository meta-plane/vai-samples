// SuffleNet public interface and building blocks
#pragma once

#include "neuralNet.h"
#include "neuralNodes.h"
#include <memory>
#include <optional>
#include <vector>
#include <array>

// ConvBlock: Conv -> ReLU -> MaxPool
class ConvBlock : public NodeGroup
{
    uint32_t C, F, K;
    ConvolutionNode conv;
    ReluNode relu;
    MaxPoolingNode maxpool;

public:
    ConvBlock(uint32_t inChannels, uint32_t outChannels, uint32_t kernel)
    : C(inChannels), F(outChannels), K(kernel),
      conv(inChannels, outChannels, kernel), maxpool(2)
    {
        conv - relu - maxpool;
        defineSlot("in0", conv.slot("in0"));
        defineSlot("out0", maxpool.slot("out0"));
    }

    Tensor& operator[](const std::string& name)
    {
        return conv[name];
    }
};

// ConvSequence: sequence of ConvBlocks
template<uint32_t nBlocks>
class ConvSequence : public NodeGroup
{
    uint32_t K;
    uint32_t C[nBlocks + 1];
    std::unique_ptr<ConvBlock> blocks[nBlocks];
    
public:
    ConvSequence(const uint32_t(&channels)[nBlocks + 1], uint32_t kernel)
    : K(kernel)
    {
        for (uint32_t i = 0; i <= nBlocks; ++i)
            C[i] = channels[i];

        for (uint32_t i = 0; i < nBlocks; ++i)
            blocks[i] = std::make_unique<ConvBlock>(C[i], C[i + 1], K);

        for (uint32_t i = 0; i < nBlocks - 1; ++i)
            *blocks[i] - *blocks[i + 1];

        defineSlot("in0", blocks[0]->slot("in0"));
        defineSlot("out0", blocks[nBlocks - 1]->slot("out0"));
    }

    Tensor& operator[](const std::string& name)
    {
        for (uint32_t i = 0; i < nBlocks; ++i)
            if (name.starts_with("conv" + std::to_string(i) + "."))
                return (*blocks[i])[name.substr(6)];
        throw std::runtime_error("No such layer in ConvSequence: " + name);
    }
};

template<std::size_t N>
ConvSequence(const uint32_t (&)[N], uint32_t) -> ConvSequence<N - 1>;

// Squeeze-and-Excitation layer (skeleton)
class SELayer : public NodeGroup
{
    uint32_t C, F, K;
    AdaptiveAvgPoolingNode aap;
    ConvolutionNode conv_1;
    BatchNormNode bn;
    ReluNode relu;
    ConvolutionNode conv_2;
    HSigmoidNode hsig;
    MultiplyNode mul;

public:
    SELayer(uint32_t inChannels)
    : C(inChannels),
      aap(1), conv_1(inChannels, inChannels/4, 1),
      bn(inChannels/4), conv_2(inChannels/4, inChannels, 1)
    {
        aap - conv_1 - bn - relu - conv_2 - hsig - "atten" / mul;
        defineSlot("in0", aap.slot("in0"));
        defineSlot("in1", mul.slot("in0"));
        defineSlot("out0", mul.slot("out0"));
    }

    Tensor& operator[](const std::string& name)
    {
        // Support: conv1.*, bn.*, conv2.*
        if (name.starts_with("conv1.")) return conv_1[name.substr(6)];
        if (name.starts_with("bn."))    return bn[name.substr(3)];
        if (name.starts_with("conv2.")) return conv_2[name.substr(6)];
        return conv_1[name]; // fallback: weight & bias
    }
};

// Activation selector
enum class Act { ReLU, HS };

// ShuffleNetV2 basic unit
// - Matches python blocks.Shufflenet (non-Xception)
class ShuffleUnit : public NodeGroup
{
    // parameters
    uint32_t inp;   // for stride==1, this is half of full input channels; for stride==2, full input channels
    uint32_t oup;   // block output channels
    uint32_t mid;   // base_mid_channels (== oup/2 per python)
    uint32_t k;     // kernel size for depthwise (approximated by normal conv)
    uint32_t s;     // stride (1 or 2)
    bool useSE;
    Act act;

    // main branch
    ConvolutionNode pw1;  // inp -> mid (1x1)
    BatchNormNode   bn1;
    ReluNode        relu1;
    HSNode          hs1;
    ConvolutionNode dw;   // mid -> mid (kxk, s)
    BatchNormNode   bn2;
    ConvolutionNode pw2;  // mid -> (oup - full_inp)
    BatchNormNode   bn3;
    ReluNode        relu2;
    HSNode          hs2;
    std::unique_ptr<SELayer> se; // optional, only when act==HS and useSE

    // projection branch (only when s==2)
    ConvolutionNode dw_proj;   // inp -> inp (kxk, s)
    BatchNormNode   bn_proj1;
    ConvolutionNode pw_proj;   // inp -> inp (1x1)
    BatchNormNode   bn_proj2;
    ReluNode        relu_proj;
    HSNode          hs_proj;

    // utilities
    ChannelShuffleNode cs;   // only used when s==1; C = 2*inp
    SplitNode         split; // only used when s==2; fan-out input to proj/main
    ConcatNode         concat;

public:
    ShuffleUnit(uint32_t inp_, uint32_t oup_, uint32_t base_mid_channels,
                uint32_t ksize, uint32_t stride_, Act activation, bool useSE_)
    : inp(inp_), oup(oup_), mid(base_mid_channels), k(ksize), s(stride_), useSE(useSE_), act(activation)
    , pw1(inp_, base_mid_channels, 1), bn1(base_mid_channels)
    , dw(base_mid_channels, base_mid_channels, ksize, stride_, int(ksize/2)), bn2(base_mid_channels)
    , pw2(base_mid_channels, (oup_ > inp_ ? (oup_ - inp_) : 0), 1), bn3((oup_ > inp_ ? (oup_ - inp_) : 0))
    , dw_proj(inp_, inp_, ksize, stride_, int(ksize/2)), bn_proj1(inp_)
    , pw_proj(inp_, inp_, 1), bn_proj2(inp_)
    , cs(2 * inp_)
    {
        // Optional SE only when HS is used (to mirror python assertion)
        if (act == Act::HS && useSE && (oup > inp))
            se = std::make_unique<SELayer>(oup - inp);

        if (s == 1)
        {
            // stride==1: split input channels via channel shuffle, main operates on half (out_odd)
            // cs: in0 -> out_even (x_proj), out_odd (x)
            // main: x -> pw1 -> bn1 -> act -> dw -> bn2 -> pw2 -> bn3 -> act2 [-> SE]
            cs / "out_odd" - pw1 - bn1;
            if (act == Act::HS) { bn1 - hs1 - dw; } else { bn1 - relu1 - dw; }
            dw - bn2 - pw2 - bn3;

            if (act == Act::HS) { bn3 - hs2; }
            else { bn3 - relu2; }

            // concat: [x_proj, main]
            (cs / "out_even") - ("in0" / concat);

            if (se)
            {
                // send post-activation to both se.in0 (for pooling) and se.in1 (for multiply)
                if (act == Act::HS)
                {
                    hs2 - ("in0" / *se);
                    hs2 - ("in1" / *se);
                    (*se) - ("in1" / concat);
                }
                else
                {
                    // should not reach due to condition above, but keep safe path
                    relu2 - ("in0" / *se);
                    relu2 - ("in1" / *se);
                    (*se) - ("in1" / concat);
                }
            }
            else
            {
                if (act == Act::HS) hs2 - ("in1" / concat); else relu2 - ("in1" / concat);
            }

            // expose group IO
            defineSlot("in0", cs.slot("in0"));
            defineSlot("out0", concat.slot("out0"));
        }
        else
        {
            // stride==2: both branches consume the same input (fan-out internally)
            // proj: x -> dw_proj -> bn -> pw_proj -> bn -> act
            split / "out0" - dw_proj - bn_proj1 - pw_proj - bn_proj2;
            if (act == Act::HS) bn_proj2 - hs_proj; else bn_proj2 - relu_proj;

            // main: x -> pw1 -> bn1 -> act -> dw -> bn2 -> pw2 -> bn3 -> act2 [-> SE]
            split / "out1" - pw1 - bn1;
            if (act == Act::HS) { bn1 - hs1; hs1 - dw; } else { bn1 - relu1; relu1 - dw; }
            dw - bn2 - pw2 - bn3;
            if (act == Act::HS) bn3 - hs2; else bn3 - relu2;

            // concat: [proj, main]
            if (se)
            {
                if (act == Act::HS)
                {
                    hs2 - ("in0" / *se);
                    hs2 - ("in1" / *se);
                    (*se) - ("in1" / concat);
                }
                else
                {
                    relu2 - ("in0" / *se);
                    relu2 - ("in1" / *se);
                    (*se) - ("in1" / concat);
                }
            }
            else
            {
                if (act == Act::HS) hs2 - ("in1" / concat); else relu2 - ("in1" / concat);
            }

            if (act == Act::HS) ("out0" / hs_proj) - ("in0" / concat); else ("out0" / relu_proj) - ("in0" / concat);

            // expose single input slot (fan-out inside)
            defineSlot("in0", split.slot("in0"));
            defineSlot("out0", concat.slot("out0"));
        }
    }

    // Optional parameter routing helper
    Tensor& operator[](const std::string& name)
    {
        if (name.starts_with("pw1.")) return pw1[name.substr(4)];
        if (name.starts_with("dw."))  return dw[name.substr(3)];
        if (name.starts_with("pw2.")) return pw2[name.substr(4)];
        if (name.starts_with("bn1.")) return bn1[name.substr(4)];
        if (name.starts_with("bn2.")) return bn2[name.substr(4)];
        if (name.starts_with("bn3.")) return bn3[name.substr(4)];
        if (name.starts_with("proj.dw.")) return dw_proj[name.substr(8)];
        if (name.starts_with("proj.pw.")) return pw_proj[name.substr(8)];
        if (name.starts_with("proj.bn1.")) return bn_proj1[name.substr(9)];
        if (name.starts_with("proj.bn2.")) return bn_proj2[name.substr(9)];
        if (se && name.starts_with("se.")) return (*se)[name.substr(3)];
        throw std::runtime_error("No such param in ShuffleUnit: " + name);
    }

};


// ShuffleNetV2 Xception unit
// - Matches python blocks.Shuffle_Xception (with sane stride handling: first dw uses stride, others use 1)
class ShuffleXception : public NodeGroup
{
    // parameters
    uint32_t inp;   // for stride==1, half of full; for stride==2, full
    uint32_t oup;
    uint32_t mid;
    uint32_t s;
    bool useSE;
    Act act;

    // main branch
    ConvolutionNode dw1;  // inp -> inp (3x3, s)
    BatchNormNode   bn1;
    ConvolutionNode pw1;  // inp -> mid (1x1)
    BatchNormNode   bn1p;
    ReluNode        relu1;
    HSNode          hs1;

    ConvolutionNode dw2;  // mid -> mid (3x3, 1)
    BatchNormNode   bn2;
    ConvolutionNode pw2;  // mid -> mid (1x1)
    BatchNormNode   bn2p;
    ReluNode        relu2;
    HSNode          hs2;

    ConvolutionNode dw3;  // mid -> mid (3x3, 1)
    BatchNormNode   bn3;
    ConvolutionNode pw3;  // mid -> (oup - full_inp)
    BatchNormNode   bn3p;
    ReluNode        relu3;
    HSNode          hs3;
    std::unique_ptr<SELayer> se;

    // projection branch (s==2)
    ConvolutionNode dwp;  // inp -> inp (3x3, s)
    BatchNormNode   bnp1;
    ConvolutionNode pwp;  // inp -> inp (1x1)
    BatchNormNode   bnp2;
    ReluNode        relu_proj;
    HSNode          hs_proj;

    // utils
    ChannelShuffleNode cs; // only when s==1 (C=2*inp)
    SplitNode         split; // only when s==2; fan-out input
    ConcatNode         concat;

public:
    ShuffleXception(uint32_t inp_, uint32_t oup_, uint32_t base_mid_channels,
                    uint32_t stride_, Act activation, bool useSE_)
    : inp(inp_), oup(oup_), mid(base_mid_channels), s(stride_), useSE(useSE_), act(activation)
    , dw1(inp_, inp_, 3, stride_, 1), bn1(inp_)
    , pw1(inp_, base_mid_channels, 1), bn1p(base_mid_channels)
    // Match Python Shuffle_Xception: dw1, dw2, dw3 all use `stride` as in blocks.py
    , dw2(base_mid_channels, base_mid_channels, 3, s, 1), bn2(base_mid_channels)
    , pw2(base_mid_channels, base_mid_channels, 1), bn2p(base_mid_channels)
    , dw3(base_mid_channels, base_mid_channels, 3, s, 1), bn3(base_mid_channels)
    , pw3(base_mid_channels, (oup_ > inp_ ? (oup_ - inp_) : 0), 1), bn3p((oup_ > inp_ ? (oup_ - inp_) : 0))
    , dwp(inp_, inp_, 3, stride_, 1), bnp1(inp_)
    , pwp(inp_, inp_, 1), bnp2(inp_)
    , cs(2 * inp_)
    {
        if (act == Act::HS && useSE && (oup > inp))
            se = std::make_unique<SELayer>(oup - inp);

        if (s == 1)
        {
            // x_proj, x = channel_shuffle(old_x); main operates on x (half channels)
            cs / "out_odd" - dw1 - bn1 - pw1 - bn1p;
            if (act == Act::HS) { bn1p - hs1; hs1 - dw2; } else { bn1p - relu1; relu1 - dw2; }
            dw2 - bn2 - pw2 - bn2p;
            if (act == Act::HS) { bn2p - hs2; hs2 - dw3; } else { bn2p - relu2; relu2 - dw3; }
            dw3 - bn3 - pw3 - bn3p;
            if (act == Act::HS) bn3p - hs3; else bn3p - relu3;

            cs / "out_even" - "in0" / concat;

            if (se)
            {
                if (act == Act::HS) {
                    hs3 - ("in0" / *se); hs3 - ("in1" / *se); (*se) - ("in1" / concat);
                } else {
                    relu3 - ("in0" / *se); relu3 - ("in1" / *se); (*se) - ("in1" / concat);
                }
            }
            else
            {
                if (act == Act::HS) hs3 - ("in1" / concat); else relu3 - ("in1" / concat);
            }

            defineSlot("in0", cs.slot("in0"));
            defineSlot("out0", concat.slot("out0"));
        }
        else
        {
            // stride==2: proj and main both take the same input (fan-out internally)
            split / "out0" - dwp - bnp1 - pwp - bnp2;
            if (act == Act::HS) bnp2 - hs_proj; else bnp2 - relu_proj;

            split / "out1" - dw1 - bn1 - pw1 - bn1p;
            if (act == Act::HS) { bn1p - hs1; hs1 - dw2; } else { bn1p - relu1; relu1 - dw2; }
            dw2 - bn2 - pw2 - bn2p;
            if (act == Act::HS) { bn2p - hs2; hs2 - dw3; } else { bn2p - relu2; relu2 - dw3; }
            dw3 - bn3 - pw3 - bn3p;
            if (act == Act::HS) bn3p - hs3; else bn3p - relu3;

            if (se)
            {
                if (act == Act::HS) { hs3 - ("in0" / *se); hs3 - ("in1" / *se); (*se) - ("in1" / concat); }
                else { relu3 - ("in0" / *se); relu3 - ("in1" / *se); (*se) - ("in1" / concat); }
            }
            else
            {
                if (act == Act::HS) hs3 - ("in1" / concat); else relu3 - ("in1" / concat);
            }

            if (act == Act::HS) ("out0" / hs_proj) - ("in0" / concat); else ("out0" / relu_proj) - ("in0" / concat);

            defineSlot("in0", split.slot("in0"));
            defineSlot("out0", concat.slot("out0"));
        }
    }

    Tensor& operator[](const std::string& name)
    {
        // Order matters: check more specific/longer prefixes first to avoid accidental matches
        if (name.starts_with("proj.dw.")) return dwp[name.substr(8)];
        if (name.starts_with("proj.pw.")) return pwp[name.substr(8)];
        if (name.starts_with("proj.bn1.")) return bnp1[name.substr(9)];
        if (name.starts_with("proj.bn2.")) return bnp2[name.substr(9)];

        if (name.starts_with("dw3."))   return dw3[name.substr(4)];
        if (name.starts_with("pw3."))   return pw3[name.substr(4)];
        if (name.starts_with("bn3p."))  return bn3p[name.substr(5)];
        if (name.starts_with("bn3."))   return bn3[name.substr(4)];

        if (name.starts_with("dw2."))   return dw2[name.substr(4)];
        if (name.starts_with("pw2."))   return pw2[name.substr(4)];
        if (name.starts_with("bn2p."))  return bn2p[name.substr(5)];
        if (name.starts_with("bn2."))   return bn2[name.substr(4)];

        if (name.starts_with("dw1."))   return dw1[name.substr(4)];
        if (name.starts_with("pw1."))   return pw1[name.substr(4)];
        if (name.starts_with("bn1p."))  return bn1p[name.substr(5)];
        if (name.starts_with("bn1."))   return bn1[name.substr(4)];

        if (se && name.starts_with("se.")) return (*se)[name.substr(3)];
        throw std::runtime_error("No such param in ShuffleXception: " + name);
    }

};

// SuffleNet model
class SuffleNetV2 : public NeuralNet
{
    // first conv: 3 -> 16, k3 s2 p1, BN, HS
    ConvolutionNode first_conv{3, 16, 3, 2, 1};
    BatchNormNode   first_bn{16};
    HSNode          first_hs;
    // feature blocks (owning containers)
    std::vector<std::unique_ptr<ShuffleUnit>>     blocks_su;
    std::vector<std::unique_ptr<ShuffleXception>> blocks_xc;
    // last conv: C -> 1280, BN, HS
    std::unique_ptr<ConvolutionNode> conv_last; // depends on last stage channels
    std::unique_ptr<BatchNormNode>   bn_last;   // 1280
    HSNode                           last_hs;
    // head: AvgPool2d(7,7, no padding, tail discard) -> SE(1280) -> Flatten -> FC(1280->1280) -> HS -> FC(1280->num_class)
    AveragePoolingNode     gap{7};
    SELayer                lastSE{1280};
    FlattenNode            flatten;
    FullyConnectedNode     fc1{1280, 1280};
    HSNode                 fc_hs;
    FullyConnectedNode     classifier; // 1280 -> num_class
    static constexpr std::array<uint32_t, 4> stage_repeats{4, 4, 8, 4};
    // Small only: [-1, 16, 36, 104, 208, 416, 1280]
    static constexpr std::array<uint32_t, 7> stage_out_channels_small{static_cast<uint32_t>(-1), 16, 36, 104, 208, 416, 1280};

    struct FeatureRef { bool isXc; ShuffleUnit* su; ShuffleXception* xc; };
    std::vector<FeatureRef> feature_order;

public:
    SuffleNetV2(Device& device, const std::vector<int>& architecture, uint32_t num_class = 1000)
    : NeuralNet(device, 1, 1)
    , classifier(1280, num_class)
    {
        // Stem: input -> conv -> bn -> HS
        input(0) - first_conv - first_bn - first_hs;
        std::vector<NodeGroup*> feature_chain;
        uint32_t input_channel = stage_out_channels_small[1]; // 16
        size_t archIndex = 0;
        for (size_t idxstage = 0; idxstage < stage_repeats.size(); ++idxstage)
        {
            uint32_t numrepeat = stage_repeats[idxstage];
            uint32_t output_channel = stage_out_channels_small[idxstage + 2];
            Act activation = (idxstage >= 1) ? Act::HS : Act::ReLU;
            bool useSE = (idxstage >= 2);
            for (uint32_t i = 0; i < numrepeat; ++i)
            {
                uint32_t inp = (i == 0) ? input_channel : (input_channel / 2);
                uint32_t stride = (i == 0) ? 2u : 1u;
                _ASSERT(archIndex < architecture.size());
                int blockIndex = architecture[archIndex++];
                if (blockIndex == 0 || blockIndex == 1 || blockIndex == 2)
                {
                    uint32_t ksize = (blockIndex == 0 ? 3u : (blockIndex == 1 ? 5u : 7u));
                    auto& blk = *blocks_su.emplace_back(std::make_unique<ShuffleUnit>(inp, output_channel, output_channel / 2, ksize, stride, activation, useSE));
                    feature_chain.push_back(&blk);
                    feature_order.push_back({false, &blk, nullptr});
                }
                else if (blockIndex == 3)
                {
                    auto& blk = *blocks_xc.emplace_back(std::make_unique<ShuffleXception>(inp, output_channel, output_channel / 2, stride, activation, useSE));
                    feature_chain.push_back(&blk);
                    feature_order.push_back({true, nullptr, &blk});
                }
                else
                {
                    throw std::runtime_error("Invalid block index for ShuffleNetV2: " + std::to_string(blockIndex));
                }
                input_channel = output_channel; // update
            }
        }
        // Tail: conv_last(C->1280)->bn->HS
        conv_last = std::make_unique<ConvolutionNode>(input_channel, 1280, 1, 1, 0);
        bn_last   = std::make_unique<BatchNormNode>(1280);

        // Link chain: first_hs -> first block -> ... -> last block -> conv_last -> bn_last -> last_hs
        if (!feature_chain.empty())
        {
            (NodeFlow)first_hs - *feature_chain.front();
            for (size_t k = 0; k + 1 < feature_chain.size(); ++k)
                *feature_chain[k] - *feature_chain[k + 1];
            *feature_chain.back() - *conv_last - *bn_last - last_hs;
        }
        else
        {
            // No blocks (theoretically shouldn't happen for ShuffleNetV2 Small), wire stem -> tail
            (NodeFlow)first_hs - *conv_last - *bn_last - last_hs;
        }
        // AvgPool2d(7) -> SE (feed same tensor to both in0 and in1)
        (NodeFlow)last_hs - gap;
        (NodeFlow)gap - ("in0" / lastSE);
        // PyTorch applies SE after global average pooling: both feature and atten are 1x1
        (NodeFlow)gap - ("in1" / lastSE);
        // Flatten -> FC -> HS -> Classifier -> output
        ("out0" / lastSE) - flatten - fc1 - fc_hs - classifier - output(0);
    }

    Tensor& debug_first_hs() { return first_hs["out0"]; }
    Tensor& debug_first_conv_out() { return first_conv["out0"]; }
    Tensor& debug_first_bn_out() { return first_bn["out0"]; }
    uint32_t debug_feature_count() const { return static_cast<uint32_t>(feature_order.size()); }
    Tensor& debug_feature_out(uint32_t idx)
    {
        _ASSERT(idx < feature_order.size());
        if (feature_order[idx].isXc) {
            _ASSERT(feature_order[idx].xc);
            return feature_order[idx].xc->slot("out0").getValueRef();
        } else {
            _ASSERT(feature_order[idx].su);
            return feature_order[idx].su->slot("out0").getValueRef();
        }
    }
    Tensor& debug_conv_last_out() { return (*conv_last)["out0"]; }
    Tensor& debug_bn_last_out() { return (*bn_last)["out0"]; }
    Tensor& debug_last_hs_out() { return last_hs["out0"]; }

    // Optional parameter routing
    Tensor& operator[](const std::string& name)
    {
        if (name.starts_with("stem.conv.")) return first_conv[name.substr(10)];
        if (name.starts_with("stem.bn."))   return first_bn[name.substr(8)];
        if (name.starts_with("lastSE."))    return lastSE[name.substr(7)];
        if (name.starts_with("last.conv.")) return (*conv_last)[name.substr(10)];
        if (name.starts_with("last.bn."))   return (*bn_last)[name.substr(8)];
        if (name == "fc1.weight" || name == "fc1.bias") return fc1[name.substr(4)];
        if (name == "classifier.weight" || name == "classifier.bias") return classifier[name.substr(11)];
        if (name.starts_with("features."))
        {
            // features.<idx>.<inner>
            size_t p = std::string("features.").size();
            size_t dot = name.find('.', p);
            if (dot == std::string::npos) throw std::runtime_error("Invalid feature name: " + name);
            uint32_t idx = static_cast<uint32_t>(std::stoul(name.substr(p, dot - p)));
            std::string inner = name.substr(dot + 1);
            _ASSERT(idx < feature_order.size());
            if (!feature_order[idx].isXc && feature_order[idx].su)
                return (*feature_order[idx].su)[inner];
            if (feature_order[idx].isXc && feature_order[idx].xc)
                return (*feature_order[idx].xc)[inner];
            throw std::runtime_error("Invalid feature route for: " + name);
        }
        throw std::runtime_error("No such parameter in SuffleNetV2");
    }


};

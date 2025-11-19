#ifndef GPT2_WEIGHTS_H
#define GPT2_WEIGHTS_H

#include "gpt2Net.h"
#include "../core/binaryWeightsReader.h"
#include <iostream>
#include <stdexcept>

/**
 * Load pretrained GPT-2 weights into GPT2Net
 *
 * Weight mapping from OpenAI format to our model:
 * - embedding.token → embedding["token_weight"]
 * - embedding.position → embedding["pos_weight"]
 * - block.{i}.attn.wq → transformerBlocks[i]["attn_wq"]
 * - block.{i}.attn.wk → transformerBlocks[i]["attn_wk"]
 * - block.{i}.attn.wv → transformerBlocks[i]["attn_wv"]
 * - block.{i}.attn.wout → transformerBlocks[i]["attn_wout"]
 * - block.{i}.ff.w1 → transformerBlocks[i]["ff_w1"]
 * - block.{i}.ff.w2 → transformerBlocks[i]["ff_w2"]
 * - block.{i}.ln1.weight → transformerBlocks[i]["norm1_scale"]
 * - block.{i}.ln1.bias → transformerBlocks[i]["norm1_shift"]
 * - block.{i}.ln2.weight → transformerBlocks[i]["norm2_scale"]
 * - block.{i}.ln2.bias → transformerBlocks[i]["norm2_shift"]
 * - finalNorm.weight → finalNorm["scale"]
 * - finalNorm.bias → finalNorm["shift"]
 * - lmHead.weight → lmHead["weight"]
 *
 * Note: Bias terms are in the weights file but not loaded (our implementation doesn't use bias)
 */
inline void loadGPT2Weights(GPT2Net& model, const std::string& weights_file)
{
    BinaryWeightsReader reader;
    reader.load(weights_file);

    const GPT2Config& config = model.getConfig();

    std::cout << "\nLoading weights into GPT2Net..." << std::endl;

    // Load embeddings
    std::cout << "  Loading embeddings..." << std::endl;
    {
        const auto& tokenEmbData = reader.getTensor("wte");
        const auto& tokenEmbShape = reader.getShape("wte");

        const auto& posEmbData = reader.getTensor("wpe");
        const auto& posEmbShape = reader.getShape("wpe");

        if (tokenEmbShape.size() != 2 || tokenEmbShape[0] != config.vocab_size || tokenEmbShape[1] != config.d_model) {
            throw std::runtime_error("Unexpected embedding.token shape");
        }
        if (posEmbShape.size() != 2 || posEmbShape[0] != config.max_seq_len || posEmbShape[1] != config.d_model) {
            throw std::runtime_error("Unexpected embedding.position shape");
        }

        // Set token embeddings
        Tensor tokenEmbTensor = Tensor(config.vocab_size, config.d_model).set(tokenEmbData).setConstant();
        model["embedding.token"] = tokenEmbTensor;

        // Set position embeddings
        model["embedding.position"] = Tensor(config.max_seq_len, config.d_model).set(posEmbData).setConstant();

        // Weight tying: LM head uses token embeddings
        model.lmHead["weight"] = tokenEmbTensor;
    }

    // Load transformer blocks
    std::cout << "  Loading " << config.num_layers << " transformer blocks..." << std::endl;
    for (uint32_t i = 0; i < config.num_layers; ++i) {
        std::string filePrefix = "h." + std::to_string(i) + ".";

        // Load attention weights (Q, K, V, Out)
        {
            auto loadAttnWeight = [&](const std::string& name, const std::string& targetKey) {
                const auto& data = reader.getTensor(filePrefix + name);
                const auto& shape = reader.getShape(filePrefix + name);

                if (shape.size() != 2 || shape[0] != config.d_model || shape[1] != config.d_model) {
                    throw std::runtime_error("Unexpected shape for " + filePrefix + name);
                }

                model["block." + std::to_string(i) + "." + targetKey] = Tensor(shape[0], shape[1]).set(data).setConstant();
            };

            loadAttnWeight("attn.W_query.weight", "attn.wq");
            loadAttnWeight("attn.W_key.weight", "attn.wk");
            loadAttnWeight("attn.W_value.weight", "attn.wv");
            loadAttnWeight("attn.out_proj.weight", "attn.wout");
        }

        // Load feedforward weights
        {
            // First layer: d_model → 4*d_model
            {
                const auto& data = reader.getTensor(filePrefix + "ff.layers.0.weight");
                const auto& shape = reader.getShape(filePrefix + "ff.layers.0.weight");

                if (shape.size() != 2 || shape[0] != 4 * config.d_model || shape[1] != config.d_model) {
                    throw std::runtime_error("Unexpected shape for " + filePrefix + "ff.layers.0.weight");
                }

                model["block." + std::to_string(i) + ".ff.w1"] = Tensor(shape[0], shape[1]).set(data).setConstant();
            }

            // Second layer: 4*d_model → d_model
            {
                const auto& data = reader.getTensor(filePrefix + "ff.layers.2.weight");
                const auto& shape = reader.getShape(filePrefix + "ff.layers.2.weight");

                if (shape.size() != 2 || shape[0] != config.d_model || shape[1] != 4 * config.d_model) {
                    throw std::runtime_error("Unexpected shape for " + filePrefix + "ff.layers.2.weight");
                }

                model["block." + std::to_string(i) + ".ff.w2"] = Tensor(shape[0], shape[1]).set(data).setConstant();
            }
        }

        // Load layer norms
        {
            auto loadNorm = [&](const std::string& name, const std::string& param, const std::string& targetKey) {
                const auto& data = reader.getTensor(filePrefix + name + "." + param);
                const auto& shape = reader.getShape(filePrefix + name + "." + param);

                if (shape.size() != 1 || shape[0] != config.d_model) {
                    throw std::runtime_error("Unexpected shape for " + filePrefix + name + "." + param);
                }

                model["block." + std::to_string(i) + "." + targetKey] = Tensor(shape[0]).set(data).setConstant();
            };

            loadNorm("norm1", "scale", "ln1.weight");
            loadNorm("norm1", "shift", "ln1.bias");
            loadNorm("norm2", "scale", "ln2.weight");
            loadNorm("norm2", "shift", "ln2.bias");
        }

        if ((i + 1) % 3 == 0 || i == 0) {
            std::cout << "    Loaded block " << (i + 1) << "/" << config.num_layers << std::endl;
        }
    }

    // Load final layer norm
    std::cout << "  Loading final layer norm..." << std::endl;
    {
        const auto& scaleData = reader.getTensor("final_norm.scale");
        const auto& scaleShape = reader.getShape("final_norm.scale");

        const auto& shiftData = reader.getTensor("final_norm.shift");
        const auto& shiftShape = reader.getShape("final_norm.shift");

        if (scaleShape.size() != 1 || scaleShape[0] != config.d_model) {
            throw std::runtime_error("Unexpected finalNorm.weight shape");
        }
        if (shiftShape.size() != 1 || shiftShape[0] != config.d_model) {
            throw std::runtime_error("Unexpected finalNorm.bias shape");
        }

        model["finalNorm.weight"] = Tensor(scaleShape[0]).set(scaleData).setConstant();
        model["finalNorm.bias"] = Tensor(shiftShape[0]).set(shiftData).setConstant();
    }

    std::cout << "✓ All weights loaded into model!" << std::endl;
}

#endif // GPT2_WEIGHTS_H

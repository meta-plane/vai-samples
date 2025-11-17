#include "gpt2.h"
#include "../core/error.h"
#include <cmath>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <ctime>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

static ComputePipeline requestPipeline(Device& device, const char* src)
{
    static std::unordered_map<const char*, ComputePipeline> pipelineCache;

    auto [it, inserted] = pipelineCache.try_emplace(src);
    if (inserted)
        it->second = device.createComputePipeline({src});
    return it->second;
}

// ============================================================================
// Language Modeling Head Shader: Y = X @ W^T
// Projects from d_model to vocab_size
// ============================================================================

static const char* src_lm_head = R"(
#version 450
layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) buffer Output { float y[]; };    // [B*S, V]
layout(set = 0, binding = 1) buffer Input { float x[]; };     // [B*S, D]
layout(set = 0, binding = 2) buffer Weight { float w[]; };    // [V, D]

layout(push_constant) uniform PushConstants {
    int BS;  // batch * seq_len
    int D;   // d_model
    int V;   // vocab_size
};

void main() {
    int bs = int(gl_GlobalInvocationID.x);
    int v = int(gl_GlobalInvocationID.y);

    if (bs >= BS || v >= V) return;

    // Y[bs, v] = sum_d(X[bs, d] * W[v, d])
    float sum = 0.0;
    for (int d = 0; d < D; ++d) {
        sum += x[bs * D + d] * w[v * D + d];
    }
    y[bs * V + v] = sum;
}
)";

// ============================================================================
// GPT-2 Configuration Presets
// ============================================================================

GPT2Config GPT2SmallConfig() {
    return GPT2Config{
        .vocab_size = 50257,
        .max_seq_len = 1024,
        .d_model = 768,
        .num_heads = 12,
        .num_layers = 12,
        .dropout = 0.1f
    };
}

GPT2Config GPT2MediumConfig() {
    return GPT2Config{
        .vocab_size = 50257,
        .max_seq_len = 1024,
        .d_model = 1024,
        .num_heads = 16,
        .num_layers = 24,
        .dropout = 0.1f
    };
}

GPT2Config GPT2LargeConfig() {
    return GPT2Config{
        .vocab_size = 50257,
        .max_seq_len = 1024,
        .d_model = 1280,
        .num_heads = 20,
        .num_layers = 36,
        .dropout = 0.1f
    };
}

GPT2Config GPT2XLConfig() {
    return GPT2Config{
        .vocab_size = 50257,
        .max_seq_len = 1024,
        .d_model = 1600,
        .num_heads = 25,
        .num_layers = 48,
        .dropout = 0.1f
    };
}

// ============================================================================
// GPT-2 Implementation
// ============================================================================

GPT2::GPT2(Device& device, DescriptorPool& descPool, const GPT2Config& config)
    : config(config), device(device), descPool(descPool), net(device, 1, 1)
{
    buildModel();
    initializeWeights();
}

GPT2::~GPT2()
{
    delete embedding;
    // transformerBlocks are now unique_ptr, automatically cleaned up
    delete finalNorm;
    delete lmHead;
}

void GPT2::buildModel()
{
    // 1. Embedding layer (token + positional)
    embedding = new GPTEmbeddingNode(config.vocab_size, config.max_seq_len, config.d_model);

    // 2. Transformer blocks
    transformerBlocks.reserve(config.num_layers);
    for (uint32_t i = 0; i < config.num_layers; ++i) {
        transformerBlocks.push_back(
            std::make_unique<TransformerBlockNode>(config.d_model, config.num_heads)
        );
    }

    // 3. Final layer normalization
    finalNorm = new LayerNormNode(config.d_model);

    // 4. Language modeling head (projects to vocabulary)
    lmHead = new LMHeadNode(config.d_model, config.vocab_size);

    // 5. Build neural network graph
    // Input -> Embedding
    net.input(0) - *embedding;

    // Embedding -> TransformerBlocks (sequential)
    // TransformerBlockNode internally handles residual connections
    *embedding - *transformerBlocks[0];
    for (size_t i = 1; i < transformerBlocks.size(); ++i) {
        *transformerBlocks[i - 1] - *transformerBlocks[i];
    }

    // Last TransformerBlock -> Final LayerNorm -> LM Head -> Output
    *transformerBlocks.back() - *finalNorm - *lmHead - net.output(0);
}

void GPT2::initializeWeights()
{
    // Initialize embedding weights if not already set
    Tensor& token_emb = (*embedding)["token_weight"];
    Tensor& pos_emb = (*embedding)["pos_weight"];

    if (!token_emb.validShape()) {
        // Initialize with small random values (normally would load pretrained)
        std::vector<float> token_data(config.vocab_size * config.d_model, 0.01f);
        token_emb = Tensor(config.vocab_size, config.d_model).set(token_data);
    }
    if (!pos_emb.validShape()) {
        // Initialize positional embeddings with small values
        std::vector<float> pos_data(config.max_seq_len * config.d_model, 0.01f);
        pos_emb = Tensor(config.max_seq_len, config.d_model).set(pos_data);
    }

    // Initialize transformer block weights
    for (uint32_t i = 0; i < config.num_layers; ++i) {
        auto* block = transformerBlocks[i].get();

        // LayerNorm1
        if (!(*block)["norm1_scale"].validShape()) {
            std::vector<float> scale_data(config.d_model, 1.0f);
            (*block)["norm1_scale"] = Tensor(config.d_model).set(scale_data);
        }
        if (!(*block)["norm1_shift"].validShape()) {
            std::vector<float> shift_data(config.d_model, 0.0f);
            (*block)["norm1_shift"] = Tensor(config.d_model).set(shift_data);
        }

        // Attention weights
        if (!(*block)["attn_wq"].validShape()) {
            std::vector<float> wq_data(config.d_model * config.d_model, 0.01f);
            (*block)["attn_wq"] = Tensor(config.d_model, config.d_model).set(wq_data);
        }
        if (!(*block)["attn_wk"].validShape()) {
            std::vector<float> wk_data(config.d_model * config.d_model, 0.01f);
            (*block)["attn_wk"] = Tensor(config.d_model, config.d_model).set(wk_data);
        }
        if (!(*block)["attn_wv"].validShape()) {
            std::vector<float> wv_data(config.d_model * config.d_model, 0.01f);
            (*block)["attn_wv"] = Tensor(config.d_model, config.d_model).set(wv_data);
        }
        if (!(*block)["attn_wout"].validShape()) {
            std::vector<float> wout_data(config.d_model * config.d_model, 0.01f);
            (*block)["attn_wout"] = Tensor(config.d_model, config.d_model).set(wout_data);
        }

        // LayerNorm2
        if (!(*block)["norm2_scale"].validShape()) {
            std::vector<float> scale_data(config.d_model, 1.0f);
            (*block)["norm2_scale"] = Tensor(config.d_model).set(scale_data);
        }
        if (!(*block)["norm2_shift"].validShape()) {
            std::vector<float> shift_data(config.d_model, 0.0f);
            (*block)["norm2_shift"] = Tensor(config.d_model).set(shift_data);
        }

        // FeedForward weights
        if (!(*block)["ff_w1"].validShape()) {
            std::vector<float> w1_data(4 * config.d_model * config.d_model, 0.01f);
            (*block)["ff_w1"] = Tensor(4 * config.d_model, config.d_model).set(w1_data);
        }
        if (!(*block)["ff_w2"].validShape()) {
            std::vector<float> w2_data(config.d_model * 4 * config.d_model, 0.01f);
            (*block)["ff_w2"] = Tensor(config.d_model, 4 * config.d_model).set(w2_data);
        }
    }

    // Initialize final LayerNorm weights
    if (!(*finalNorm)["scale"].validShape()) {
        std::vector<float> scale_data(config.d_model, 1.0f);
        (*finalNorm)["scale"] = Tensor(config.d_model).set(scale_data);
    }
    if (!(*finalNorm)["shift"].validShape()) {
        std::vector<float> shift_data(config.d_model, 0.0f);
        (*finalNorm)["shift"] = Tensor(config.d_model).set(shift_data);
    }

    // Weight tying: LM head shares weights with token embedding
    (*lmHead)["weight"] = (*embedding)["token_weight"];
}

Tensor GPT2::forward(const Tensor& input_ids)
{
    _ASSERT(input_ids.shape().size() == 2);  // [batch, seq_len]

    uint32_t B = input_ids.shape()[0];
    uint32_t S = input_ids.shape()[1];
    _ASSERT(S <= config.max_seq_len);

    // Run full network: embedding -> transformers -> layernorm -> lm_head
    // Output is logits [B, S, vocab_size] on GPU
    std::vector<Tensor> outputs = net(input_ids);

    // Return GPU tensor directly (like 10-mnist example)
    return outputs[0];
}

void GPT2::loadWeights(const std::string& weights_file)
{
    std::cout << "Loading GPT-2 weights from: " << weights_file << std::endl;

    FILE* f = fopen(weights_file.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("Failed to open weights file: " + weights_file);
    }

    // Read number of weights
    uint32_t num_weights = 0;
    fread(&num_weights, sizeof(uint32_t), 1, f);
    std::cout << "Number of weight tensors: " << num_weights << std::endl;

    std::unordered_map<std::string, std::vector<float>> weights_map;
    std::unordered_map<std::string, std::vector<uint32_t>> shapes_map;

    // Read all weights into memory
    for (uint32_t i = 0; i < num_weights; ++i) {
        // Read name
        uint32_t name_len = 0;
        fread(&name_len, sizeof(uint32_t), 1, f);

        std::vector<char> name_buf(name_len + 1, 0);
        fread(name_buf.data(), 1, name_len, f);
        std::string name(name_buf.data());

        // Read shape
        uint32_t num_dims = 0;
        fread(&num_dims, sizeof(uint32_t), 1, f);

        std::vector<uint32_t> shape(num_dims);
        fread(shape.data(), sizeof(uint32_t), num_dims, f);

        // Calculate total elements
        uint32_t total_elements = 1;
        for (uint32_t dim : shape) {
            total_elements *= dim;
        }

        // Read data
        std::vector<float> data(total_elements);
        fread(data.data(), sizeof(float), total_elements, f);

        weights_map[name] = std::move(data);
        shapes_map[name] = std::move(shape);

        std::cout << "  [" << (i + 1) << "/" << num_weights << "] " << name << " ";
        for (size_t j = 0; j < shapes_map[name].size(); ++j) {
            std::cout << (j > 0 ? "x" : "") << shapes_map[name][j];
        }
        std::cout << std::endl;
    }

    fclose(f);

    std::cout << "\nMapping weights to model..." << std::endl;

    // Helper function to transpose 2D weights
    auto transpose = [](const std::vector<float>& data, uint32_t rows, uint32_t cols) {
        std::vector<float> transposed(data.size());
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t c = 0; c < cols; ++c) {
                transposed[c * rows + r] = data[r * cols + c];
            }
        }
        return transposed;
    };

    // Load embeddings
    if (weights_map.count("wte.weight")) {
        (*embedding)["token_weight"] = Tensor(config.vocab_size, config.d_model)
            .set(weights_map["wte.weight"]);
        std::cout << "  Loaded token embeddings" << std::endl;
    }

    if (weights_map.count("wpe.weight")) {
        (*embedding)["pos_weight"] = Tensor(config.max_seq_len, config.d_model)
            .set(weights_map["wpe.weight"]);
        std::cout << "  Loaded positional embeddings" << std::endl;
    }

    // Load transformer blocks
    for (uint32_t i = 0; i < config.num_layers; ++i) {
        std::string prefix = "transformer.h." + std::to_string(i) + ".";
        TransformerBlockNode* block = transformerBlocks[i].get();

        // LayerNorm 1
        std::string ln1_weight = prefix + "ln_1.weight";
        std::string ln1_bias = prefix + "ln_1.bias";
        if (weights_map.count(ln1_weight)) {
            (*block)["norm1_scale"] = Tensor(config.d_model).set(weights_map[ln1_weight]);
        }
        if (weights_map.count(ln1_bias)) {
            (*block)["norm1_shift"] = Tensor(config.d_model).set(weights_map[ln1_bias]);
        }

        // Attention: c_attn contains concatenated Q, K, V weights
        std::string attn_weight = prefix + "attn.c_attn.weight";
        if (weights_map.count(attn_weight)) {
            const auto& qkv_data = weights_map[attn_weight];
            // Shape: [d_model, 3*d_model] in HuggingFace format
            // Need to split into Q, K, V and transpose

            uint32_t d = config.d_model;
            std::vector<float> q_data(d * d);
            std::vector<float> k_data(d * d);
            std::vector<float> v_data(d * d);

            for (uint32_t r = 0; r < d; ++r) {
                for (uint32_t c = 0; c < d; ++c) {
                    q_data[c * d + r] = qkv_data[r * (3 * d) + c];       // Q: columns 0 to d-1, transposed
                    k_data[c * d + r] = qkv_data[r * (3 * d) + d + c];   // K: columns d to 2d-1, transposed
                    v_data[c * d + r] = qkv_data[r * (3 * d) + 2*d + c]; // V: columns 2d to 3d-1, transposed
                }
            }

            (*block)["attn_wq"] = Tensor(d, d).set(q_data);
            (*block)["attn_wk"] = Tensor(d, d).set(k_data);
            (*block)["attn_wv"] = Tensor(d, d).set(v_data);
        }

        // Attention output projection
        std::string attn_proj = prefix + "attn.c_proj.weight";
        if (weights_map.count(attn_proj)) {
            auto wout_t = transpose(weights_map[attn_proj], config.d_model, config.d_model);
            (*block)["attn_wout"] = Tensor(config.d_model, config.d_model).set(wout_t);
        }

        // LayerNorm 2
        std::string ln2_weight = prefix + "ln_2.weight";
        std::string ln2_bias = prefix + "ln_2.bias";
        if (weights_map.count(ln2_weight)) {
            (*block)["norm2_scale"] = Tensor(config.d_model).set(weights_map[ln2_weight]);
        }
        if (weights_map.count(ln2_bias)) {
            (*block)["norm2_shift"] = Tensor(config.d_model).set(weights_map[ln2_bias]);
        }

        // FeedForward: first linear layer
        std::string mlp_fc = prefix + "mlp.c_fc.weight";
        if (weights_map.count(mlp_fc)) {
            // Shape: [d_model, 4*d_model] -> transpose to [4*d_model, d_model]
            auto w1_t = transpose(weights_map[mlp_fc], config.d_model, 4 * config.d_model);
            (*block)["ff_w1"] = Tensor(4 * config.d_model, config.d_model).set(w1_t);
        }

        // FeedForward: second linear layer
        std::string mlp_proj = prefix + "mlp.c_proj.weight";
        if (weights_map.count(mlp_proj)) {
            // Shape: [4*d_model, d_model] -> transpose to [d_model, 4*d_model]
            auto w2_t = transpose(weights_map[mlp_proj], 4 * config.d_model, config.d_model);
            (*block)["ff_w2"] = Tensor(config.d_model, 4 * config.d_model).set(w2_t);
        }

        std::cout << "  Loaded transformer block " << i << std::endl;
    }

    // Load final layer norm
    if (weights_map.count("transformer.ln_f.weight")) {
        (*finalNorm)["scale"] = Tensor(config.d_model)
            .set(weights_map["transformer.ln_f.weight"]);
    }
    if (weights_map.count("transformer.ln_f.bias")) {
        (*finalNorm)["shift"] = Tensor(config.d_model)
            .set(weights_map["transformer.ln_f.bias"]);
    }
    std::cout << "  Loaded final layer norm" << std::endl;

    std::cout << "\nSuccessfully loaded all GPT-2 weights!" << std::endl;
}

// Helper function: Sample from logits with temperature and top-k
static int sampleToken(const std::vector<float>& logits, float temperature, int top_k)
{
    std::vector<float> probs = logits;

    // Apply temperature
    if (temperature != 1.0f) {
        for (float& p : probs) {
            p /= temperature;
        }
    }

    // Softmax: subtract max for numerical stability
    float max_logit = *std::max_element(probs.begin(), probs.end());
    for (float& p : probs) {
        p = std::exp(p - max_logit);
    }

    float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    for (float& p : probs) {
        p /= sum;
    }

    // Top-k filtering
    if (top_k > 0 && top_k < (int)probs.size()) {
        // Create indices
        std::vector<std::pair<float, int>> prob_idx;
        for (size_t i = 0; i < probs.size(); ++i) {
            prob_idx.push_back({probs[i], (int)i});
        }

        // Sort by probability (descending)
        std::sort(prob_idx.begin(), prob_idx.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        // Zero out probabilities outside top-k
        for (size_t i = 0; i < probs.size(); ++i) {
            probs[i] = 0.0f;
        }
        for (int i = 0; i < top_k; ++i) {
            probs[prob_idx[i].second] = prob_idx[i].first;
        }

        // Renormalize
        sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
        for (float& p : probs) {
            p /= sum;
        }
    }

    // Sample from categorical distribution
    float r = (float)rand() / RAND_MAX;
    float cumsum = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        cumsum += probs[i];
        if (r < cumsum) {
            return (int)i;
        }
    }

    // Fallback to last token
    return (int)probs.size() - 1;
}

std::vector<int> GPT2::generate(
    const std::vector<int>& prompt_ids,
    uint32_t max_new_tokens,
    float temperature,
    int top_k)
{
    std::cout << "\nGenerating text..." << std::endl;
    std::cout << "  Prompt length: " << prompt_ids.size() << " tokens" << std::endl;
    std::cout << "  Max new tokens: " << max_new_tokens << std::endl;
    std::cout << "  Temperature: " << temperature << std::endl;
    std::cout << "  Top-k: " << (top_k > 0 ? std::to_string(top_k) : "disabled") << std::endl;

    // Start with prompt
    std::vector<int> generated = prompt_ids;

    // Generate tokens autoregressively
    for (uint32_t i = 0; i < max_new_tokens; ++i) {
        // Prepare input tensor from current sequence
        uint32_t seq_len = (uint32_t)generated.size();

        // Take last max_seq_len tokens if sequence is too long
        uint32_t start_idx = 0;
        if (seq_len > config.max_seq_len) {
            start_idx = seq_len - config.max_seq_len;
            seq_len = config.max_seq_len;
        }

        std::vector<float> input_data(seq_len);
        for (uint32_t j = 0; j < seq_len; ++j) {
            input_data[j] = (float)generated[start_idx + j];
        }

        Tensor input = Tensor(1, seq_len).set(input_data);

        // Forward pass - returns logits on GPU [1, seq_len, vocab_size]
        Tensor gpu_logits = forward(input);

        // Get GPU buffer first (following 10-mnist pattern)
        Buffer gpu_buffer = gpu_logits.buffer();

        // Copy GPU logits to CPU
        uint32_t logits_size = seq_len * config.vocab_size * sizeof(float);
        Buffer cpu_buffer = device.createBuffer({
            .size = logits_size,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        });

        // Copy from GPU to CPU
        device.newCommandBuffer(queue_compute)
            .begin()
            .copyBuffer(cpu_buffer, gpu_buffer)
            .end()
            .submit()
            .wait();

        // Map and extract logits data
        std::vector<float> logits_data(seq_len * config.vocab_size);
        memcpy(logits_data.data(), cpu_buffer.map(), logits_size);
        cpu_buffer.unmap();

        // Get logits for last token
        std::vector<float> last_token_logits(config.vocab_size);
        uint32_t last_token_offset = (seq_len - 1) * config.vocab_size;
        for (uint32_t j = 0; j < config.vocab_size; ++j) {
            last_token_logits[j] = logits_data[last_token_offset + j];
        }

        // Sample next token
        int next_token = sampleToken(last_token_logits, temperature, top_k);

        // Append to sequence
        generated.push_back(next_token);

        // Progress indicator
        if ((i + 1) % 10 == 0 || i == 0) {
            std::cout << "  Generated " << (i + 1) << "/" << max_new_tokens << " tokens..." << std::endl;
        }
    }

    std::cout << "  Generation complete! Total tokens: " << generated.size() << std::endl;
    return generated;
}

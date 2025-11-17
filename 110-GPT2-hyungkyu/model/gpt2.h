#ifndef GPT2_H
#define GPT2_H

#include "../core/neuralNet.h"
#include "../core/vulkanApp.h"
#include "embedding/embeddingNode.h"
#include "transformerBlock/transformerNode.h"
#include "lmHeadNode.h"

using namespace vk;

/**
 * GPT-2 Language Model Configuration
 */
struct GPT2Config {
    uint32_t vocab_size;      // Vocabulary size (50257 for GPT-2)
    uint32_t max_seq_len;     // Maximum sequence length (1024 for GPT-2)
    uint32_t d_model;         // Model dimension (768 for GPT-2 small, 1024 for medium, etc.)
    uint32_t num_heads;       // Number of attention heads (12 for small, 16 for medium, etc.)
    uint32_t num_layers;      // Number of transformer blocks (12 for small, 24 for medium, etc.)
    float dropout;            // Dropout rate (not used in inference)
};

/**
 * GPT-2 Language Model
 *
 * Architecture:
 * Input: [batch, seq_len] token IDs
 * 1. Token Embedding + Positional Embedding → [batch, seq_len, d_model]
 * 2. N × TransformerBlock (with residual connections)
 * 3. Final LayerNorm
 * 4. Language Modeling Head (linear projection to vocab)
 * Output: [batch, seq_len, vocab_size] logits
 */
class GPT2
{
private:
    GPT2Config config;
    Device& device;
    DescriptorPool& descPool;

    NeuralNet net;

    // Nodes
    GPTEmbeddingNode* embedding;
    std::vector<std::unique_ptr<TransformerBlockNode>> transformerBlocks;
    LayerNormNode* finalNorm;
    LMHeadNode* lmHead;

    void buildModel();
    void initializeWeights();

public:
    GPT2(Device& device, DescriptorPool& descPool, const GPT2Config& config);
    ~GPT2();

    // Forward pass - returns logits [batch, seq_len, vocab_size]
    Tensor forward(const Tensor& input_ids);

    // Generate text autoregressively
    // prompt_ids: [1, prompt_len] - initial tokens
    // max_new_tokens: number of tokens to generate
    // temperature: sampling temperature (higher = more random)
    // top_k: if > 0, only sample from top k tokens
    // Returns: [1, prompt_len + max_new_tokens] - generated token IDs
    std::vector<int> generate(
        const std::vector<int>& prompt_ids,
        uint32_t max_new_tokens = 50,
        float temperature = 1.0f,
        int top_k = 0
    );

    // Load pretrained weights from HuggingFace format
    void loadWeights(const std::string& weights_file);

    // Access to underlying network (for loading weights, etc.)
    NeuralNet& getNetwork() { return net; }
    const GPT2Config& getConfig() const { return config; }
};

// Helper function to create common GPT-2 configurations
GPT2Config GPT2SmallConfig();   // 124M parameters
GPT2Config GPT2MediumConfig();  // 355M parameters
GPT2Config GPT2LargeConfig();   // 774M parameters
GPT2Config GPT2XLConfig();      // 1.5B parameters

#endif // GPT2_H

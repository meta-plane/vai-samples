#ifndef SHADERS_H
#define SHADERS_H

// ============================================================================
// GPT-2 Shader Sources
// ============================================================================
// This file declares all compute shader sources used in the GPT-2 model.
// Actual shader code is defined in their respective model component files.

// Attention shaders (attentionNode.cpp)
extern const char* src_linear;
extern const char* src_softmax;
extern const char* src_qkv_projection;
extern const char* src_attention_scores;
extern const char* src_attention_scores_cached;
extern const char* src_causal_mask;
extern const char* src_causal_mask_cached;
extern const char* src_weighted_sum;
extern const char* src_weighted_sum_cached;
extern const char* src_combine_heads;
extern const char* src_reshape_to_heads;
extern const char* src_update_cache;
extern const char* src_concatenate_kv;

// Transformer block shaders (transformer.cpp)
extern const char* src_layer_norm;
extern const char* src_gelu;
extern const char* src_linear_ff;
extern const char* src_add;

// Embedding shaders (embeddingNode.cpp)
extern const char* src_token_embedding;
extern const char* src_positional_embedding;
extern const char* src_add_embeddings;

// LM head shaders (lmHeadNode.cpp)
extern const char* src_lm_head;

#endif // SHADERS_H

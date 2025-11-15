#include "embedding.h"
#include "../../core/error.h"
#include <iostream>
#include <cmath>

// ==================== TokenEmbedding ====================

TokenEmbedding::TokenEmbedding(int vocab_size, int embedding_dim)
    : vocab_size(vocab_size), embedding_dim(embedding_dim) {
    ASSERT_(vocab_size > 0);
    ASSERT_(embedding_dim > 0);

    // Initialize embedding table
    weight.resize(vocab_size, std::vector<float>(embedding_dim));
    initialize_weights();

    std::cout << "TokenEmbedding initialized: vocab_size=" << vocab_size
              << ", embedding_dim=" << embedding_dim << std::endl;
}

void TokenEmbedding::initialize_weights() {
    // Random initialization with small values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.02f);  // Mean=0, Std=0.02

    for (auto& row : weight) {
        for (auto& val : row) {
            val = dist(gen);
        }
    }
}

std::vector<float> TokenEmbedding::get_token_embedding(int token_id) const {
    ASSERT_(token_id >= 0 && token_id < vocab_size);
    return weight[token_id];
}

Tensor3D TokenEmbedding::forward(const std::vector<std::vector<int>>& token_ids) const {
    int batch_size = token_ids.size();
    ASSERT_(batch_size > 0);

    int seq_length = token_ids[0].size();
    ASSERT_(seq_length > 0);

    // Output: [batch_size, seq_length, embedding_dim]
    Tensor3D embeddings(batch_size,
        std::vector<std::vector<float>>(seq_length,
            std::vector<float>(embedding_dim)));

    // Lookup embeddings for each token
    for (int b = 0; b < batch_size; b++) {
        ASSERT_(token_ids[b].size() == static_cast<size_t>(seq_length));

        for (int s = 0; s < seq_length; s++) {
            int token_id = token_ids[b][s];
            embeddings[b][s] = get_token_embedding(token_id);
        }
    }

    return embeddings;
}

// ==================== PositionalEmbedding ====================

PositionalEmbedding::PositionalEmbedding(int max_length, int embedding_dim)
    : max_length(max_length), embedding_dim(embedding_dim) {
    ASSERT_(max_length > 0);
    ASSERT_(embedding_dim > 0);

    // Initialize positional embedding table
    weight.resize(max_length, std::vector<float>(embedding_dim));
    initialize_weights();

    std::cout << "PositionalEmbedding initialized: max_length=" << max_length
              << ", embedding_dim=" << embedding_dim << std::endl;
}

void PositionalEmbedding::initialize_weights() {
    // Random initialization with small values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.02f);

    for (auto& row : weight) {
        for (auto& val : row) {
            val = dist(gen);
        }
    }
}

std::vector<float> PositionalEmbedding::get_position_embedding(int position) const {
    ASSERT_(position >= 0 && position < max_length);
    return weight[position];
}

Tensor2D PositionalEmbedding::forward(int seq_length) const {
    ASSERT_(seq_length > 0 && seq_length <= max_length);

    // Output: [seq_length, embedding_dim]
    Tensor2D pos_embeddings(seq_length, std::vector<float>(embedding_dim));

    // Get embeddings for positions 0 to seq_length-1
    for (int pos = 0; pos < seq_length; pos++) {
        pos_embeddings[pos] = get_position_embedding(pos);
    }

    return pos_embeddings;
}

// ==================== GPTEmbedding ====================

GPTEmbedding::GPTEmbedding(int vocab_size, int max_length, int embedding_dim)
    : token_emb(vocab_size, embedding_dim)
    , pos_emb(max_length, embedding_dim) {
    std::cout << "GPTEmbedding initialized successfully" << std::endl;
}

Tensor3D GPTEmbedding::add_embeddings(const Tensor3D& token_emb,
                                       const Tensor2D& pos_emb) const {
    int batch_size = token_emb.size();
    int seq_length = token_emb[0].size();
    int emb_dim = token_emb[0][0].size();

    // Result: [batch_size, seq_length, embedding_dim]
    Tensor3D result(batch_size,
        std::vector<std::vector<float>>(seq_length,
            std::vector<float>(emb_dim)));

    // Add token and positional embeddings element-wise
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_length; s++) {
            for (int e = 0; e < emb_dim; e++) {
                result[b][s][e] = token_emb[b][s][e] + pos_emb[s][e];
            }
        }
    }

    return result;
}

Tensor3D GPTEmbedding::forward(const std::vector<std::vector<int>>& token_ids) const {
    // Get token embeddings
    Tensor3D tok_emb = token_emb.forward(token_ids);

    // Get positional embeddings
    int seq_length = token_ids[0].size();
    Tensor2D pos_emb_seq = pos_emb.forward(seq_length);

    // Combine: token + position
    return add_embeddings(tok_emb, pos_emb_seq);
}

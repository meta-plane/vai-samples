#pragma once

#include <vector>
#include <string>
#include <random>

// 3D tensor: [batch_size, seq_length, embedding_dim]
using Tensor3D = std::vector<std::vector<std::vector<float>>>;

// 2D tensor: [size, embedding_dim]
using Tensor2D = std::vector<std::vector<float>>;

// Token Embedding Layer
// Converts token IDs to dense vectors
class TokenEmbedding {
public:
    // Constructor: initialize embedding table
    // vocab_size: number of unique tokens
    // embedding_dim: dimension of embedding vectors
    TokenEmbedding(int vocab_size, int embedding_dim);

    // Forward pass: convert token IDs to embeddings
    // Input: token_ids [batch_size, seq_length]
    // Output: embeddings [batch_size, seq_length, embedding_dim]
    Tensor3D forward(const std::vector<std::vector<int>>& token_ids) const;

    // Get embedding for a single token
    std::vector<float> get_token_embedding(int token_id) const;

    // Getters
    int get_vocab_size() const { return vocab_size; }
    int get_embedding_dim() const { return embedding_dim; }

private:
    int vocab_size;
    int embedding_dim;
    Tensor2D weight;  // Embedding table [vocab_size, embedding_dim]

    // Initialize weights with random values
    void initialize_weights();
};

// Positional Embedding Layer
// Adds position information to token embeddings
class PositionalEmbedding {
public:
    // Constructor: initialize positional embedding table
    // max_length: maximum sequence length (context length)
    // embedding_dim: dimension of embedding vectors
    PositionalEmbedding(int max_length, int embedding_dim);

    // Forward pass: get positional embeddings for a sequence
    // Input: seq_length (length of current sequence)
    // Output: pos_embeddings [seq_length, embedding_dim]
    Tensor2D forward(int seq_length) const;

    // Get embedding for a single position
    std::vector<float> get_position_embedding(int position) const;

    // Getters
    int get_max_length() const { return max_length; }
    int get_embedding_dim() const { return embedding_dim; }

private:
    int max_length;
    int embedding_dim;
    Tensor2D weight;  // Positional embedding table [max_length, embedding_dim]

    // Initialize weights with random values
    void initialize_weights();
};

// Combined Embedding: Token + Positional
// This is the complete input embedding for GPT
class GPTEmbedding {
public:
    // Constructor
    GPTEmbedding(int vocab_size, int max_length, int embedding_dim);

    // Forward pass: token embeddings + positional embeddings
    // Input: token_ids [batch_size, seq_length]
    // Output: embeddings [batch_size, seq_length, embedding_dim]
    Tensor3D forward(const std::vector<std::vector<int>>& token_ids) const;

    // Getters
    int get_vocab_size() const { return token_emb.get_vocab_size(); }
    int get_max_length() const { return pos_emb.get_max_length(); }
    int get_embedding_dim() const { return token_emb.get_embedding_dim(); }

private:
    TokenEmbedding token_emb;
    PositionalEmbedding pos_emb;

    // Add two tensors element-wise
    Tensor3D add_embeddings(const Tensor3D& token_emb, const Tensor2D& pos_emb) const;
};

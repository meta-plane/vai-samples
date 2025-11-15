#include "bpeTokenizer.h"
#include "../core/error.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <regex>
#include <algorithm>
#include <iostream>

using json = nlohmann::json;

BPETokenizer::BPETokenizer(const std::string& vocab_file, const std::string& merges_file) {
    initByteEncoder();
    loadVocab(vocab_file);
    loadMerges(merges_file);
}

void BPETokenizer::initByteEncoder() {
    // GPT-2 byte encoder: maps bytes to unicode characters
    // Ranges that map directly: '!'=0x21 to '~'=0x7E, 0xA1 to 0xAC, 0xAE to 0xFF
    // Others map to 256+offset
    std::vector<int> bytes_list;
    std::vector<int> chars_list;
    
    // Add printable ASCII (! to ~)
    for (int b = 0x21; b <= 0x7E; b++) {
        bytes_list.push_back(b);
        chars_list.push_back(b);
    }
    
    // Add high bytes (161-172, 174-255)
    for (int b = 0xA1; b <= 0xAC; b++) {
        bytes_list.push_back(b);
        chars_list.push_back(b);
    }
    for (int b = 0xAE; b <= 0xFF; b++) {
        bytes_list.push_back(b);
        chars_list.push_back(b);
    }
    
    // Add remaining bytes (control chars, space, etc.) mapped to 256+offset
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (std::find(bytes_list.begin(), bytes_list.end(), b) == bytes_list.end()) {
            bytes_list.push_back(b);
            chars_list.push_back(256 + n);
            n++;
        }
    }
    
    // Create byte_encoder and byte_decoder mappings
    for (size_t i = 0; i < bytes_list.size(); i++) {
        unsigned char byte_val = static_cast<unsigned char>(bytes_list[i]);
        int char_code = chars_list[i];
        
        // Convert char_code to UTF-8 string
        std::string utf8_str;
        if (char_code < 0x80) {
            utf8_str += static_cast<char>(char_code);
        } else if (char_code < 0x800) {
            utf8_str += static_cast<char>(0xC0 | (char_code >> 6));
            utf8_str += static_cast<char>(0x80 | (char_code & 0x3F));
        } else {
            utf8_str += static_cast<char>(0xE0 | (char_code >> 12));
            utf8_str += static_cast<char>(0x80 | ((char_code >> 6) & 0x3F));
            utf8_str += static_cast<char>(0x80 | (char_code & 0x3F));
        }
        
        byte_encoder[byte_val] = utf8_str;
        byte_decoder[utf8_str] = byte_val;
    }
}

void BPETokenizer::loadVocab(const std::string& vocab_file) {
    std::ifstream file(vocab_file);
    ASSERT_(file.is_open());
    
    json j;
    file >> j;
    
    for (auto& [key, value] : j.items()) {
        encoder[key] = value.get<int>();
        decoder[value.get<int>()] = key;
    }
    
    std::cout << "Loaded vocabulary with " << encoder.size() << " tokens" << std::endl;
}

void BPETokenizer::loadMerges(const std::string& merges_file) {
    std::ifstream file(merges_file);
    ASSERT_(file.is_open());
    
    std::string line;
    // Skip header line
    std::getline(file, line);
    
    int rank = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        size_t space_pos = line.find(' ');
        if (space_pos != std::string::npos) {
            std::string first = line.substr(0, space_pos);
            std::string second = line.substr(space_pos + 1);
            bpe_ranks[{first, second}] = rank++;
        }
    }
    
    std::cout << "Loaded " << bpe_ranks.size() << " BPE merge rules" << std::endl;
}

std::set<std::pair<size_t, size_t>> BPETokenizer::get_pairs(const std::vector<std::string>& word) {
    std::set<std::pair<size_t, size_t>> pairs;
    
    for (size_t i = 0; i < word.size() - 1; i++) {
        pairs.insert({i, i + 1});
    }
    
    return pairs;
}

std::vector<std::string> BPETokenizer::bpe(const std::string& token) {
    if (token.empty()) {
        return {};
    }
    
    // Split token into UTF-8 characters (not bytes!)
    std::vector<std::string> word;
    size_t i = 0;
    while (i < token.length()) {
        unsigned char c = token[i];
        
        // Determine UTF-8 character length
        int char_len = 1;
        if ((c & 0x80) == 0) {
            // 1-byte (ASCII)
            char_len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            // 2-byte UTF-8
            char_len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte UTF-8
            char_len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte UTF-8
            char_len = 4;
        }
        
        // Extract the full UTF-8 character
        if (i + char_len <= token.length()) {
            word.push_back(token.substr(i, char_len));
            i += char_len;
        } else {
            // Malformed UTF-8, just take one byte
            word.push_back(token.substr(i, 1));
            i++;
        }
    }
    
    if (word.size() == 1) {
        return word;
    }
    
    // Apply BPE merges
    while (word.size() > 1) {
        // Find the pair with lowest rank
        int min_rank = INT_MAX;
        int best_i = -1;
        
        for (size_t i = 0; i < word.size() - 1; i++) {
            auto pair = std::make_pair(word[i], word[i + 1]);
            auto it = bpe_ranks.find(pair);
            if (it != bpe_ranks.end() && it->second < min_rank) {
                min_rank = it->second;
                best_i = i;
            }
        }
        
        if (best_i == -1) {
            break;  // No more merges possible
        }
        
        // Merge the best pair
        std::vector<std::string> new_word;
        size_t i = 0;
        while (i < word.size()) {
            if ((int)i == best_i) {
                new_word.push_back(word[i] + word[i + 1]);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i++;
            }
        }
        word = new_word;
    }
    
    return word;
}

std::vector<std::string> splitGPT2Pattern(const std::string& text) {
    // GPT-2 pattern: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    // Implementation:
    // - Contractions ('s, 't, 're, 've, 'm, 'll, 'd)
    // - Optional spaces + letters/numbers/punctuation
    // - Trailing whitespace
    // Key: Spaces are attached to the FOLLOWING token, not separate

    std::vector<std::string> words;
    size_t i = 0;

    while (i < text.length()) {
        std::string match;
        char c = text[i];

        // Check for contractions first
        if (c == '\'' && i + 1 < text.length()) {
            char next = text[i + 1];
            if (next == 's' || next == 't' || next == 'm' || next == 'd') {
                match = text.substr(i, 2);
                i += 2;
                words.push_back(match);
                continue;
            } else if (i + 2 < text.length()) {
                std::string two_char = text.substr(i + 1, 2);
                if (two_char == "re" || two_char == "ve" || two_char == "ll") {
                    match = text.substr(i, 3);
                    i += 3;
                    words.push_back(match);
                    continue;
                }
            }
        }

        // Consume optional leading spaces (any number of them)
        while (i < text.length() && (text[i] == ' ' || text[i] == '\n' || text[i] == '\t')) {
            match += text[i++];
        }

        if (i >= text.length()) {
            // Only whitespace remaining (trailing)
            if (!match.empty()) {
                words.push_back(match);
            }
            break;
        }

        c = text[i];

        // Letters
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
            while (i < text.length() &&
                   ((text[i] >= 'a' && text[i] <= 'z') || (text[i] >= 'A' && text[i] <= 'Z'))) {
                match += text[i++];
            }
            words.push_back(match);
        }
        // Numbers
        else if (c >= '0' && c <= '9') {
            while (i < text.length() && text[i] >= '0' && text[i] <= '9') {
                match += text[i++];
            }
            words.push_back(match);
        }
        // Punctuation/special chars (not whitespace)
        else {
            // Single punctuation character
            match += c;
            i++;
            words.push_back(match);
        }
    }

    return words;
}

std::vector<int> BPETokenizer::encode(const std::string& text) {
    std::vector<int> token_ids;
    
    // Split text using GPT-2 pattern
    auto words = splitGPT2Pattern(text);
    
    // Process each word/token
    for (const auto& word : words) {
        // Convert to bytes and then to unicode using byte_encoder
        std::string byte_encoded;
        for (unsigned char c : word) {
            auto it = byte_encoder.find(c);
            if (it != byte_encoder.end()) {
                byte_encoded += it->second;
            }
        }
        
        // Apply BPE
        auto bpe_tokens = bpe(byte_encoded);
        
        // Convert to IDs
        for (const auto& token : bpe_tokens) {
            auto it = encoder.find(token);
            if (it != encoder.end()) {
                token_ids.push_back(it->second);
            }
        }
    }
    
    return token_ids;
}

std::string BPETokenizer::decode(const std::vector<int>& tokens) {
    std::string text;
    
    // Convert token IDs to token strings
    for (int token_id : tokens) {
        auto it = decoder.find(token_id);
        if (it != decoder.end()) {
            text += it->second;
        }
    }
    
    // Decode the concatenated token string back to bytes
    std::string result;
    size_t i = 0;
    while (i < text.length()) {
        // Try to match multi-byte UTF-8 sequences first (3-byte, 2-byte, 1-byte)
        bool found = false;
        
        // Try 3-byte UTF-8
        if (i + 2 < text.length()) {
            std::string three_byte = text.substr(i, 3);
            auto it = byte_decoder.find(three_byte);
            if (it != byte_decoder.end()) {
                result += static_cast<char>(it->second);
                i += 3;
                found = true;
                continue;
            }
        }
        
        // Try 2-byte UTF-8
        if (!found && i + 1 < text.length()) {
            std::string two_byte = text.substr(i, 2);
            auto it = byte_decoder.find(two_byte);
            if (it != byte_decoder.end()) {
                result += static_cast<char>(it->second);
                i += 2;
                found = true;
                continue;
            }
        }
        
        // Try 1-byte
        if (!found) {
            std::string one_byte = text.substr(i, 1);
            auto it = byte_decoder.find(one_byte);
            if (it != byte_decoder.end()) {
                result += static_cast<char>(it->second);
            } else {
                // Fallback: keep original character
                result += text[i];
            }
            i++;
        }
    }
    
    return result;
}


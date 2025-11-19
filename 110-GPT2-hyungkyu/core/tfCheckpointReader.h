#ifndef TF_CHECKPOINT_READER_H
#define TF_CHECKPOINT_READER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <stdexcept>
#include <cstring>

/**
 * Simple TensorFlow Checkpoint Reader
 * Reads weights from TensorFlow checkpoint files (specifically GPT-2 format)
 */
class TFCheckpointReader
{
    std::unordered_map<std::string, std::vector<float>> tensors;
    std::unordered_map<std::string, std::vector<uint32_t>> shapes;

public:
    /**
     * Load TensorFlow checkpoint from directory
     * Reads model.ckpt.data-00000-of-00001 and model.ckpt.index
     */
    void load(const std::string& checkpoint_dir);

    /**
     * Get tensor by name
     */
    const std::vector<float>& getTensor(const std::string& name) const;

    /**
     * Get tensor shape
     */
    const std::vector<uint32_t>& getShape(const std::string& name) const;

    /**
     * Check if tensor exists
     */
    bool hasTensor(const std::string& name) const;

    /**
     * Get all tensor names
     */
    std::vector<std::string> getTensorNames() const;

    /**
     * Transpose 2D tensor
     */
    static std::vector<float> transpose(const std::vector<float>& data, uint32_t rows, uint32_t cols);

    /**
     * Split tensor along last axis into N parts
     */
    static std::vector<std::vector<float>> split(const std::vector<float>& data,
                                                   const std::vector<uint32_t>& shape,
                                                   uint32_t num_splits);
};

#endif // TF_CHECKPOINT_READER_H

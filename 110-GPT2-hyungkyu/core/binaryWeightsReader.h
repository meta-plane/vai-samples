#ifndef BINARY_WEIGHTS_READER_H
#define BINARY_WEIGHTS_READER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <iostream>

/**
 * Binary Weights Reader
 * Reads weights from binary format created by convert_openai_weights.py
 *
 * Binary format:
 * - uint32_t: number of tensors
 * - For each tensor:
 *   - uint32_t: name length
 *   - char[]: name (UTF-8)
 *   - uint32_t: number of dimensions
 *   - uint32_t[]: shape dimensions
 *   - float[]: data (row-major)
 */
class BinaryWeightsReader
{
    std::unordered_map<std::string, std::vector<float>> tensors;
    std::unordered_map<std::string, std::vector<uint32_t>> shapes;

public:
    /**
     * Load weights from binary file
     */
    void load(const std::string& weights_file)
    {
        std::ifstream file(weights_file, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open weights file: " + weights_file);
        }

        // Read number of tensors
        uint32_t num_tensors = 0;
        file.read(reinterpret_cast<char*>(&num_tensors), sizeof(uint32_t));

        std::cout << "Loading " << num_tensors << " tensors from " << weights_file << std::endl;

        for (uint32_t i = 0; i < num_tensors; ++i) {
            // Read name
            uint32_t name_length = 0;
            file.read(reinterpret_cast<char*>(&name_length), sizeof(uint32_t));

            std::string name(name_length, '\0');
            file.read(&name[0], name_length);

            // Read shape
            uint32_t num_dims = 0;
            file.read(reinterpret_cast<char*>(&num_dims), sizeof(uint32_t));

            std::vector<uint32_t> shape(num_dims);
            file.read(reinterpret_cast<char*>(shape.data()), num_dims * sizeof(uint32_t));

            // Calculate number of elements
            uint32_t num_elements = 1;
            for (uint32_t dim : shape) {
                num_elements *= dim;
            }

            // Read data
            std::vector<float> data(num_elements);
            file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(float));

            // Store
            tensors[name] = std::move(data);
            shapes[name] = std::move(shape);

            if ((i + 1) % 20 == 0 || i == 0) {
                std::cout << "  Loaded " << (i + 1) << "/" << num_tensors << " tensors..." << std::endl;
            }
        }

        std::cout << "✓ All weights loaded successfully" << std::endl;
    }

    /**
     * Get tensor data by name
     */
    const std::vector<float>& getTensor(const std::string& name) const
    {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            throw std::runtime_error("Tensor not found: " + name);
        }
        return it->second;
    }

    /**
     * Get tensor shape by name
     */
    const std::vector<uint32_t>& getShape(const std::string& name) const
    {
        auto it = shapes.find(name);
        if (it == shapes.end()) {
            throw std::runtime_error("Shape not found for tensor: " + name);
        }
        return it->second;
    }

    /**
     * Check if tensor exists
     */
    bool hasTensor(const std::string& name) const
    {
        return tensors.find(name) != tensors.end();
    }

    /**
     * Get all tensor names
     */
    std::vector<std::string> getTensorNames() const
    {
        std::vector<std::string> names;
        names.reserve(tensors.size());
        for (const auto& [name, _] : tensors) {
            names.push_back(name);
        }
        return names;
    }
};

#endif // BINARY_WEIGHTS_READER_H

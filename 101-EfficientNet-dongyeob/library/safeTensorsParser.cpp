// @chay116 - ViT Start: SafeTensors parser (from 83-ViT)
#include "safeTensorsParser.h"
#include <nlohmann/json.hpp>
#include <fstream>


struct SafeTensorsParserImpl
{
    nlohmann::json metadata;
    std::vector<char> tensorData;

    SafeTensorsParserImpl(const char* safetensorsFilePath)
    {
        std::ifstream file(safetensorsFilePath, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("Could not open SafeTensors file: " + std::string(safetensorsFilePath));

        // 1. Read header length (first 8 bytes)
        uint64_t headerLength = 0;
        file.read(reinterpret_cast<char*>(&headerLength), sizeof(headerLength));
        if (file.gcount() != sizeof(headerLength))
            throw std::runtime_error("Failed to read header length.");

        // 2. Read JSON header
        std::string headerStr(headerLength, '\0');
        file.read(&headerStr[0], headerLength);
        if (file.gcount() != headerLength)
            throw std::runtime_error("Failed to read header string.");
        metadata = nlohmann::json::parse(headerStr);

        // 3. Read the rest of the file as binary tensor data
        file.seekg(0, std::ios::end);
        size_t endPos = file.tellg();
        size_t startOfData = sizeof(headerLength) + headerLength;
        size_t dataSize = endPos - startOfData;

        tensorData.resize(dataSize);
        file.seekg(startOfData);
        file.read(tensorData.data(), dataSize);
        if (file.gcount() != dataSize)
            throw std::runtime_error("Failed to read tensor data buffer.");
    }
};


struct SafeTensorsParserRefImpl
{
    const nlohmann::json& metadata;
    const std::vector<char>& tensorData;

    SafeTensorsParserRefImpl(const nlohmann::json& meta, const std::vector<char>& tensors) : metadata(meta), tensorData(tensors) {}
};


SafeTensorsParser::~SafeTensorsParser() = default;
SafeTensorsParser::SafeTensorsParser(const char* safetensorsFilePath)
    : pImpl(std::make_unique<SafeTensorsParserImpl>(safetensorsFilePath))
{
}


std::vector<std::string> SafeTensorsParser::getTensorNames() const
{
    std::vector<std::string> names;
    if (!pImpl->metadata.is_object()) return names;

    for (auto const& [key, val] : pImpl->metadata.items()) {
        if (key != "__metadata__") { // Ignore metadata key
            names.push_back(key);
        }
    }
    return names;
}


SafeTensorsParserRef::~SafeTensorsParserRef() = default;
SafeTensorsParserRef::SafeTensorsParserRef(std::unique_ptr<SafeTensorsParserRefImpl> impl)
    : pImpl(std::move(impl))
{
}


SafeTensorsParserRef SafeTensorsParser::operator[](std::string_view key) const
{
    if (!pImpl->metadata.is_object())
        throw std::runtime_error("Invalid SafeTensors format: top-level metadata is not an object.");

    auto it = pImpl->metadata.find(key);
    if (it == pImpl->metadata.end() || it->is_null()) // Safetensors can have a `__metadata__` key.
        throw std::out_of_range("Tensor key not found in SafeTensors metadata: " + std::string(key));

    return SafeTensorsParserRef(std::make_unique<SafeTensorsParserRefImpl>(*it, pImpl->tensorData));
}


SafeTensorsParserRef SafeTensorsParserRef::operator[](std::string_view key) const
{
    if (!pImpl->metadata.is_object())
        throw std::runtime_error("Invalid SafeTensors format: top-level metadata is not an object.");

    auto it = pImpl->metadata.find(key);
    if (it == pImpl->metadata.end() || it->is_null()) // Safetensors can have a `__metadata__` key.
        throw std::out_of_range("Tensor key not found in SafeTensors metadata: " + std::string(key));

    return SafeTensorsParserRef(std::make_unique<SafeTensorsParserRefImpl>(*it, pImpl->tensorData));
}


std::vector<uint32_t> SafeTensorsParserRef::getShape() const
{
    return pImpl->metadata.at("shape").get<std::vector<uint32_t>>();
}


std::string SafeTensorsParserRef::getDataType() const
{
    return pImpl->metadata.at("dtype").get<std::string>();
}


std::vector<float> SafeTensorsParserRef::parseNDArray() const
{
    const auto shape = getShape();
    const auto dtypeStr = getDataType();
    const auto offsets = pImpl->metadata.at("data_offsets").get<std::vector<size_t>>();
    const size_t startOffset = offsets[0];
    const size_t endOffset = offsets[1];

    const size_t numElements = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    const size_t byteSize = endOffset - startOffset;
    
    if (endOffset > pImpl->tensorData.size())
        throw std::runtime_error("Tensor offset out of bounds: endOffset=" + std::to_string(endOffset) + ", dataSize=" + std::to_string(pImpl->tensorData.size()));

    if (dtypeStr != "F32")
        throw std::runtime_error("Unsupported dtype: " + dtypeStr + " (expected F32)");

    std::vector<float> result;
    result.reserve(numElements);

    const char* bufferStart = pImpl->tensorData.data() + startOffset;

    if (byteSize != numElements * sizeof(float))
        throw std::runtime_error("Data size mismatch for F32 tensor.");
    const float* dataPtr = reinterpret_cast<const float*>(bufferStart);
    result.assign(dataPtr, dataPtr + numElements);

    return result;
}
// @chay116 - ViT End

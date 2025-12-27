// @chay116 - ViT Start: SafeTensors parser (from 83-ViT)
#ifndef SAFETENSORSPARSER_H
#define SAFETENSORSPARSER_H

#include <vector>
#include <memory>
#include <string>
#include <string_view>
#include <cstdint>


struct SafeTensorsParserImpl;
struct SafeTensorsParserRefImpl;
class SafeTensorsParserRef;


class SafeTensorsParser
{
    std::unique_ptr<SafeTensorsParserImpl> pImpl;

public:
    SafeTensorsParser(const char* safetensorsFilePath);
    ~SafeTensorsParser();

    SafeTensorsParserRef operator[](std::string_view key) const;

    // Check layer tensors names.
    std::vector<std::string> getTensorNames() const;
};


class SafeTensorsParserRef
{
    friend class SafeTensorsParser;
    std::unique_ptr<SafeTensorsParserRefImpl> pImpl;

public:
    SafeTensorsParserRef(std::unique_ptr<SafeTensorsParserRefImpl> impl);
    ~SafeTensorsParserRef();

    SafeTensorsParserRef operator[](uint32_t index) const;
    SafeTensorsParserRef operator[](std::string_view key) const;

    std::vector<float> parseNDArray(std::vector<uint32_t>& outShape) const;
    std::vector<float> parseNDArray() const;

    std::vector<uint32_t> getShape() const;
    std::string getDataType() const;
};


#endif // SAFETENSORSPARSER_H
// @chay116 - ViT End

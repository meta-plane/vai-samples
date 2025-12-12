#pragma once

#include "../library/Tensor.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <cstdint>
#include <cstring>     // std::memcpy
#include <limits>

// ONNX Proto Field IDs (공식 스펙 기준)
namespace OnnxIDs {
    // ModelProto
    const int MODEL_GRAPH = 7;  // optional GraphProto graph = 7;

    // GraphProto
    const int GRAPH_INITIALIZER = 5;  // repeated TensorProto initializer = 5;

    // TensorProto
    const int TENSOR_DIMS = 1;  // repeated int64 dims = 1;
    const int TENSOR_DATA_TYPE = 2;  // int32 data_type = 2;
    const int TENSOR_NAME = 8;  // string name = 8;
    const int TENSOR_RAW_DATA = 9;  // bytes raw_data = 9;
    // (필요하면 float_data=4 등도 나중에 추가)
}

// -------------------------
// Helper: Protobuf Varint
// -------------------------
class ProtoUtil {
public:
    static bool ReadVarint64(std::istream& is, uint64_t& out) {
        out = 0;
        int shift = 0;
        uint8_t byte = 0;

        while (shift < 64) {
            if (!is.read(reinterpret_cast<char*>(&byte), 1)) {
                return false; // EOF or error
            }
            out |= (uint64_t)(byte & 0x7Fu) << shift;
            if ((byte & 0x80u) == 0)
                return true;
            shift += 7;
        }
        return false; // malformed
    }

    static bool ReadVarint32(std::istream& is, uint32_t& out) {
        uint64_t tmp = 0;
        if (!ReadVarint64(is, tmp)) return false;
        if (tmp > std::numeric_limits<uint32_t>::max()) return false;
        out = static_cast<uint32_t>(tmp);
        return true;
    }

    static bool SkipField(std::istream& is, int wireType) {
        switch (wireType) {
        case 0: { // Varint
            uint64_t dummy;
            return ReadVarint64(is, dummy);
        }
        case 1: { // 64-bit
            is.seekg(8, std::ios::cur);
            return is.good();
        }
        case 2: { // Length-delimited
            uint64_t len = 0;
            if (!ReadVarint64(is, len)) return false;
            is.seekg(static_cast<std::streamoff>(len), std::ios::cur);
            return is.good();
        }
        case 5: { // 32-bit
            is.seekg(4, std::ios::cur);
            return is.good();
        }
        default:
            return false; // unsupported
        }
    }
};

// -------------------------------------
// ONNX Weight Loader
// -------------------------------------
class OnnxWeightLoader {
public:
    static std::map<std::string, Tensor> Load(const std::string& path) {
        std::map<std::string, Tensor> weights;

        std::ifstream fs(path, std::ios::binary);
        if (!fs.is_open()) {
            std::cerr << "Error: Could not open ONNX file: " << path << std::endl;
            return weights;
        }

        // ModelProto 파싱: graph(field 7) 찾기
        while (fs.peek() != EOF && fs.good()) {
            uint64_t tag = 0;
            if (!ProtoUtil::ReadVarint64(fs, tag)) break;

            int fieldNum = static_cast<int>(tag >> 3);
            int wireType = static_cast<int>(tag & 0x07);

            if (fieldNum == OnnxIDs::MODEL_GRAPH && wireType == 2) {
                // length-delimited GraphProto
                uint64_t length = 0;
                if (!ProtoUtil::ReadVarint64(fs, length)) break;

                std::streamoff lenOff = static_cast<std::streamoff>(length);
                std::streampos startPos = fs.tellg();
                if (startPos == std::streampos(-1)) break;
                std::streampos endPos = startPos + lenOff;

                ParseGraph(fs, startPos, endPos, weights);

                // 하나의 graph만 있다고 가정하면 바로 종료
                break;
            }
            else {
                if (!ProtoUtil::SkipField(fs, wireType)) break;
            }
        }

        return weights;
    }

private:
    // GraphProto 파싱: initializer만 관심
    static void ParseGraph(std::ifstream& fs,
        std::streampos startPos,
        std::streampos endPos,
        std::map<std::string, Tensor>& outWeights)
    {
        fs.seekg(startPos);
        while (fs.good() && fs.peek() != EOF) {
            std::streampos curPos = fs.tellg();
            if (curPos == std::streampos(-1) || curPos >= endPos) break;

            uint64_t tag = 0;
            if (!ProtoUtil::ReadVarint64(fs, tag)) break;

            int fieldNum = static_cast<int>(tag >> 3);
            int wireType = static_cast<int>(tag & 0x07);

            if (fieldNum == OnnxIDs::GRAPH_INITIALIZER && wireType == 2) {
                uint64_t length = 0;
                if (!ProtoUtil::ReadVarint64(fs, length)) break;

                std::streamoff lenOff = static_cast<std::streamoff>(length);
                std::streampos tensorStart = fs.tellg();
                if (tensorStart == std::streampos(-1)) break;
                std::streampos tensorEnd = tensorStart + lenOff;

                ParseTensor(fs, tensorStart, tensorEnd, outWeights);

                // TensorProto 끝으로 점프(혹시 덜 읽었어도 정렬)
                fs.seekg(tensorEnd);
            }
            else {
                if (!ProtoUtil::SkipField(fs, wireType)) break;
            }
        }

        fs.seekg(endPos);
    }

    // TensorProto 파싱해서 Tensor 생성
    static void ParseTensor(std::ifstream& fs,
        std::streampos startPos,
        std::streampos endPos,
        std::map<std::string, Tensor>& outWeights)
    {
        fs.seekg(startPos);

        std::string name;
        int dataType = 0;
        std::vector<uint64_t> dims64;   // ONNX dims = int64
        std::vector<char> rawData;
        std::vector<float> floatDataField; // float_data 지원용

        while (fs.good() && fs.peek() != EOF) {
            std::streampos curPos = fs.tellg();
            if (curPos == std::streampos(-1) || curPos >= endPos) break;

            uint64_t tag = 0;
            if (!ProtoUtil::ReadVarint64(fs, tag)) break;

            int fieldNum = static_cast<int>(tag >> 3);
            int wireType = static_cast<int>(tag & 0x07);

            if (fieldNum == OnnxIDs::TENSOR_DIMS) {   // repeated int64 dims = 1;
                if (wireType == 0) {
                    uint64_t d = 0;
                    if (!ProtoUtil::ReadVarint64(fs, d)) break;
                    dims64.push_back(d);
                }
                else if (wireType == 2) {
                    // packed dims
                    uint64_t len = 0;
                    if (!ProtoUtil::ReadVarint64(fs, len)) break;
                    std::streampos packedEnd = fs.tellg();
                    if (packedEnd == std::streampos(-1)) break;
                    packedEnd += static_cast<std::streamoff>(len);

                    while (fs.good()) {
                        std::streampos p = fs.tellg();
                        if (p == std::streampos(-1) || p >= packedEnd) break;
                        uint64_t d = 0;
                        if (!ProtoUtil::ReadVarint64(fs, d)) break;
                        dims64.push_back(d);
                    }
                    fs.seekg(packedEnd);
                }
                else {
                    if (!ProtoUtil::SkipField(fs, wireType)) break;
                }
            }
            else if (fieldNum == OnnxIDs::TENSOR_DATA_TYPE) {  // int32 data_type = 2;
                if (wireType != 0) {
                    if (!ProtoUtil::SkipField(fs, wireType)) break;
                    continue;
                }
                uint32_t dt = 0;
                if (!ProtoUtil::ReadVarint32(fs, dt)) break;
                dataType = static_cast<int>(dt);
            }
            else if (fieldNum == OnnxIDs::TENSOR_NAME) {       // string name = 8;
                if (wireType != 2) {
                    if (!ProtoUtil::SkipField(fs, wireType)) break;
                    continue;
                }
                uint64_t len = 0;
                if (!ProtoUtil::ReadVarint64(fs, len)) break;
                name.resize(static_cast<std::size_t>(len));
                fs.read(&name[0], static_cast<std::streamsize>(len));
            }
            else if (fieldNum == OnnxIDs::TENSOR_RAW_DATA) {   // bytes raw_data = 9;
                if (wireType != 2) {
                    if (!ProtoUtil::SkipField(fs, wireType)) break;
                    continue;
                }
                uint64_t len = 0;
                if (!ProtoUtil::ReadVarint64(fs, len)) break;
                rawData.resize(static_cast<std::size_t>(len));
                fs.read(rawData.data(), static_cast<std::streamsize>(len));
            }
            else if (fieldNum == 4) { // float_data (repeated float = 4;)
                // 혹시 raw_data가 없을 때를 대비해서 처리 (length-delimited packed)
                if (wireType == 2) {
                    uint64_t len = 0;
                    if (!ProtoUtil::ReadVarint64(fs, len)) break;
                    if (len % sizeof(float) != 0) {
                        fs.seekg(static_cast<std::streamoff>(len), std::ios::cur);
                    }
                    else {
                        size_t cnt = static_cast<size_t>(len / sizeof(float));
                        floatDataField.resize(cnt);
                        fs.read(reinterpret_cast<char*>(floatDataField.data()),
                            static_cast<std::streamsize>(len));
                    }
                }
                else if (wireType == 5) {
                    // 32-bit 한 개 float
                    float f = 0;
                    fs.read(reinterpret_cast<char*>(&f), sizeof(float));
                    floatDataField.push_back(f);
                }
                else {
                    if (!ProtoUtil::SkipField(fs, wireType)) break;
                }
            }
            else {
                if (!ProtoUtil::SkipField(fs, wireType)) break;
            }
        }

        // 여기서 Parsed TensorProto → Tensor 변환

        // 1) FLOAT만 지원
        if (dataType != 1) {
            return;
        }
        if (dims64.empty()) {
            return;
        }

        // 2) dims 변환 및 numElements 계산
        std::vector<uint32_t> dims;
        dims.reserve(dims64.size());
        size_t numElements = 1;
        for (uint64_t d : dims64) {
            if (d == 0) return;
            if (d > std::numeric_limits<uint32_t>::max()) return;
            dims.push_back(static_cast<uint32_t>(d));
            numElements *= static_cast<size_t>(d);
        }

        // 3) 데이터 소스 선택: raw_data 우선, 없으면 float_data 사용
        std::vector<float> data;

        if (!rawData.empty()) {
            if (rawData.size() != numElements * sizeof(float)) {
                // 사이즈 안 맞으면 버림
                return;
            }
            data.resize(numElements);
            std::memcpy(data.data(), rawData.data(), rawData.size());
        }
        else if (!floatDataField.empty()) {
            if (floatDataField.size() != numElements) {
                return;
            }
            data = std::move(floatDataField);
        }
        else {
            // 데이터 없음
            return;
        }

        // 4) 이름 없는 텐서는 스킵
        if (name.empty()) {
            return;
        }

        // 5) Tensor 생성 후 맵에 저장
        Tensor t(dims);
        t.set(std::move(data));
        outWeights[name] = std::move(t);
    }
};

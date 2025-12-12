

#include <iostream>
#include <algorithm>

#include <vector>
#include <string>
#include <cstdint>
#include <fstream>

#include <iomanip>


#include "library/neuralNet.h"
#include "library/tensor.h"
#include "library/vulkanApp.h"

#include "safeTensor/safeTensorsParser.h"

#include "networks/include/unet.hpp"

#include "utills.hpp"

#define IMAGE_PATH "D:/VAI/images/image.png"

#define LABEL_PATH "D:/VAI/images/label.png"

#define ONNX_PATH "D:/VAI/vai-sample/100-UNet-seungwoo/pretrained/weight/U_Net_Model.onnx"

#define SAFE_TENSOR_PATH "D:/VAI/vai-sample/100-UNet-seungwoo/pretrained/weight/unet.safetensors"



struct BMPFileHeader {
    uint16_t bfType;      // 파일 타입, 항상 'BM' = 0x4D42
    uint32_t bfSize;      // 파일 전체 크기 (바이트)
    uint16_t bfReserved1; // 예약(0)
    uint16_t bfReserved2; // 예약(0)
    uint32_t bfOffBits;   // 픽셀 데이터 시작 오프셋
};

struct BMPInfoHeader {
    uint32_t biSize;          // 이 헤더의 크기 (40 바이트)
    int32_t  biWidth;         // 이미지 가로 (픽셀)
    int32_t  biHeight;        // 이미지 세로 (픽셀) (양수 = bottom-up)
    uint16_t biPlanes;        // 항상 1
    uint16_t biBitCount;      // 픽셀당 비트 수 (24 = RGB)
    uint32_t biCompression;   // 압축 방식 (0 = BI_RGB, 무압축)
    uint32_t biSizeImage;     // 이미지 데이터 크기 (바이트)
    int32_t  biXPelsPerMeter; // 가로 해상도 (옵션, 0)
    int32_t  biYPelsPerMeter; // 세로 해상도 (옵션, 0)
    uint32_t biClrUsed;       // 색상수 (0 = 전부)
    uint32_t biClrImportant;  // 중요한 색상수 (0)
};

void saveFloatToBMP_0_255(
    const std::vector<float>& data,
    uint32_t width,
    uint32_t height,
    const std::string& filename,
    bool normalize = false)
{
    if (data.size() != size_t(width) * height)
    {
        std::cerr << "Data size mismatch" << std::endl;
        return;
    }

    // --------------------------
    // 1. float → uint8 변환
    // --------------------------
    float minv = 0.f, maxv = 1.f;
    if (normalize)
    {
        minv = *std::min_element(data.begin(), data.end());
        maxv = *std::max_element(data.begin(), data.end());
        if (maxv - minv < 1e-12f) maxv = minv + 1e-12f;
    }

    std::vector<uint8_t> u8(width * height);

    for (size_t i = 0; i < data.size(); i++)
    {
        float v = data[i];

        // min-max normalize (옵션)
        if (normalize)
            v = (v - minv) / (maxv - minv);

        // 확실히 0~255에 clamp
        v = std::clamp(v, 0.0f, 1.0f);
        u8[i] = static_cast<uint8_t>(v * 255.0f);
    }

    // --------------------------
    // 2. BMP 헤더 구성
    // --------------------------
    const int bytesPerPixel = 3; // 24bit RGB
    uint32_t rowStride = (width * bytesPerPixel + 3) & ~3u;
    uint32_t imageSize = rowStride * height;
    uint32_t fileSize = 54 + imageSize; // file header(14) + info header(40)

    unsigned char fileHeader[14] = {
        'B', 'M',          // Signature
        0,0,0,0,           // File size
        0,0,0,0,           // Reserved
        54,0,0,0           // Pixel data offset
    };
    fileHeader[2] = (unsigned char)(fileSize);
    fileHeader[3] = (unsigned char)(fileSize >> 8);
    fileHeader[4] = (unsigned char)(fileSize >> 16);
    fileHeader[5] = (unsigned char)(fileSize >> 24);

    unsigned char infoHeader[40] = {
        40,0,0,0,          // Header size
        0,0,0,0,           // Width
        0,0,0,0,           // Height
        1,0,               // Planes
        24,0,              // Bits per pixel
        0,0,0,0,           // Compression
        0,0,0,0,           // Image size
        0,0,0,0,           // X pixels per meter
        0,0,0,0,           // Y pixels per meter
        0,0,0,0,           // Colors used
        0,0,0,0            // Important colors
    };

    // width
    infoHeader[4] = (unsigned char)(width);
    infoHeader[5] = (unsigned char)(width >> 8);
    infoHeader[6] = (unsigned char)(width >> 16);
    infoHeader[7] = (unsigned char)(width >> 24);

    // height
    infoHeader[8] = (unsigned char)(height);
    infoHeader[9] = (unsigned char)(height >> 8);
    infoHeader[10] = (unsigned char)(height >> 16);
    infoHeader[11] = (unsigned char)(height >> 24);

    // image size
    infoHeader[20] = (unsigned char)(imageSize);
    infoHeader[21] = (unsigned char)(imageSize >> 8);
    infoHeader[22] = (unsigned char)(imageSize >> 16);
    infoHeader[23] = (unsigned char)(imageSize >> 24);

    // --------------------------
    // 3. 실제 BMP 파일 쓰기 (BGR, bottom-up)
    // --------------------------

    std::ofstream ofs(filename, std::ios::binary);

    ofs.write((char*)fileHeader, 14);
    ofs.write((char*)infoHeader, 40);

    std::vector<uint8_t> rowBuf(rowStride);

    for (int y = 0; y < int(height); y++)
    {
        int srcY = height - 1 - y; // bottom-up

        for (uint32_t x = 0; x < width; x++)
        {
            uint8_t g = u8[srcY * width + x];
            size_t idx = x * 3;

            rowBuf[idx + 0] = g; // B
            rowBuf[idx + 1] = g; // G
            rowBuf[idx + 2] = g; // R
        }

        ofs.write((char*)rowBuf.data(), rowStride);
    }

    ofs.close();

    std::cout << "[OK] BMP 저장 완료 → " << filename << std::endl;
}

std::vector<float> sigmoid(const std::vector<float>& input)
{
    std::vector<float> output;
    output.reserve(input.size());

    for (float v : input)
    {
        float s = 1.0f / (1.0f + std::exp(-v));
        output.push_back(s);
    }

    return output;
}
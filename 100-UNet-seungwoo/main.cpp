#include "main.hpp"

using namespace std;
using namespace networks;
using namespace utils;

typedef struct 
{
    uint32_t channels;
    uint32_t height;
    uint32_t width;
}ImageInfo;

std::vector<float> pre_processing(const ImageInfo& info, const std::vector<uint8_t>& srcImage) 
{
    std::vector<float> dstImage(info.channels * info.height * info.width);
    
    for (size_t i = 0; i < srcImage.size(); ++i)
        dstImage[i] = (srcImage[i] / 255.0);

    return dstImage;
}

bool set_Weight(Unet::w_ptr net, SafeTensorsParser& parser) 
{
    auto unet = net.lock();
    
    unet->setWeight(parser);

    return true;
}

Tensor eval_unet(Unet::w_ptr net, const ImageInfo& info, const std::vector<float>& srcImage)
{
    auto unet = net.lock();

    //Tensor H, W, C
    Tensor result;
    Tensor inputTensor = Tensor(info.height, info.width, info.channels).set(srcImage);

    result = (*unet)(inputTensor)[0];

    return result;
}

//void saveFloatGrayscaleToBMP(
//    const std::vector<float>& data,
//    uint32_t width,
//    uint32_t height,
//    const std::string& filename)
//{
//    // data 크기 체크 (C=1 가정)
//    if (data.size() != static_cast<size_t>(width) * height) {
//        std::cerr << "[Error] saveFloatGrayscaleToBMP: data size mismatch" << std::endl;
//        return;
//    }
//
//    const int bytesPerPixel = 3; // 24bit RGB
//    // 각 라인은 4바이트 배수로 패딩해야 함
//    uint32_t rowStride = (width * bytesPerPixel + 3) & ~3u; // ((...+3)/4)*4
//
//    uint32_t imageSize = rowStride * height;
//    uint32_t fileSize = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + imageSize;
//
//    BMPFileHeader fileHeader{};
//    fileHeader.bfType = 0x4D42; // 'BM'
//    fileHeader.bfSize = fileSize;
//    fileHeader.bfReserved1 = 0;
//    fileHeader.bfReserved2 = 0;
//    fileHeader.bfOffBits = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);
//
//    BMPInfoHeader infoHeader{};
//    infoHeader.biSize = sizeof(BMPInfoHeader);
//    infoHeader.biWidth = static_cast<int32_t>(width);
//    infoHeader.biHeight = static_cast<int32_t>(height); // 양수 = bottom-up 저장
//    infoHeader.biPlanes = 1;
//    infoHeader.biBitCount = 24;      // 24bit RGB
//    infoHeader.biCompression = 0;       // BI_RGB
//    infoHeader.biSizeImage = imageSize;
//    infoHeader.biXPelsPerMeter = 0;
//    infoHeader.biYPelsPerMeter = 0;
//    infoHeader.biClrUsed = 0;
//    infoHeader.biClrImportant = 0;
//
//    std::ofstream ofs(filename, std::ios::binary);
//    if (!ofs) {
//        std::cerr << "[Error] saveFloatGrayscaleToBMP: failed to open file: " << filename << std::endl;
//        return;
//    }
//
//    // 헤더 쓰기
//    ofs.write(reinterpret_cast<const char*>(&fileHeader), sizeof(fileHeader));
//    ofs.write(reinterpret_cast<const char*>(&infoHeader), sizeof(infoHeader));
//
//    // 한 줄씩(하단 → 상단) 쓰기 (BMP는 bottom-up)
//    std::vector<uint8_t> rowBuffer(rowStride);
//
//    for (int y = 0; y < static_cast<int>(height); ++y)
//    {
//        // BMP는 bottom-up 이므로 data 의 (height - 1 - y) 번째 행을 사용
//        int srcY = height - 1 - y;
//
//        for (uint32_t x = 0; x < width; ++x)
//        {
//            size_t idxFloat = static_cast<size_t>(srcY) * width + x;
//            float v = data[idxFloat];
//
//            // 0~1 범위로 가정하고 clamp 후 0~255로 변환
//            v = std::clamp(v, 0.0f, 1.0f);
//            uint8_t g = static_cast<uint8_t>(v * 255.0f);
//
//            // BMP는 BGR 순서
//            size_t idxByte = x * bytesPerPixel;
//            rowBuffer[idxByte + 0] = g; // B
//            rowBuffer[idxByte + 1] = g; // G
//            rowBuffer[idxByte + 2] = g; // R
//        }
//
//        // 패딩 영역은 0으로 채워진 상태 (rowBuffer 초기화 시 자동)
//        ofs.write(reinterpret_cast<const char*>(rowBuffer.data()), rowStride);
//    }
//
//    ofs.close();
//
//    std::cout << "[OK] BMP 저장 완료: " << filename << std::endl;
//}

void saveFloatGrayscaleToBMP(
    const std::vector<float>& data,
    uint32_t width,
    uint32_t height,
    const std::string& filename)
{
    if (data.size() != static_cast<size_t>(width) * height) {
        std::cerr << "[Error] saveFloatGrayscaleToBMP: data size mismatch" << std::endl;
        return;
    }

    const int bytesPerPixel = 3; // 24bit RGB
    // 한 줄은 반드시 4바이트 배수로 패딩
    uint32_t rowStride = (width * bytesPerPixel + 3) & ~3u;
    uint32_t imageSize = rowStride * height;
    uint32_t fileSize = 54 + imageSize;  // 14(File) + 40(Info) = 54

    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        std::cerr << "[Error] saveFloatGrayscaleToBMP: failed to open file: " << filename << std::endl;
        return;
    }

    // -------------------------
    // 14바이트 File Header
    // -------------------------
    unsigned char fileHeader[14] = {
        'B', 'M',          // 0: bfType = 'BM'
        0, 0, 0, 0,        // 2: bfSize (파일 전체 크기)
        0, 0, 0, 0,        // 6: bfReserved1, bfReserved2 (0)
        54, 0, 0, 0        // 10: bfOffBits = 54 (픽셀 데이터 시작 위치)
    };

    fileHeader[2] = (unsigned char)(fileSize);
    fileHeader[3] = (unsigned char)(fileSize >> 8);
    fileHeader[4] = (unsigned char)(fileSize >> 16);
    fileHeader[5] = (unsigned char)(fileSize >> 24);

    // -------------------------
    // 40바이트 Info Header (BITMAPINFOHEADER)
    // -------------------------
    unsigned char infoHeader[40] = {
        40, 0, 0, 0,       // 0: biSize = 40
        0, 0, 0, 0,        // 4: biWidth
        0, 0, 0, 0,        // 8: biHeight
        1, 0,              // 12: biPlanes = 1
        24, 0,             // 14: biBitCount = 24 (24bit RGB)
        0, 0, 0, 0,        // 16: biCompression = BI_RGB(0)
        0, 0, 0, 0,        // 20: biSizeImage (0 or imageSize)
        0, 0, 0, 0,        // 24: biXPelsPerMeter
        0, 0, 0, 0,        // 28: biYPelsPerMeter
        0, 0, 0, 0,        // 32: biClrUsed
        0, 0, 0, 0         // 36: biClrImportant
    };

    // width
    infoHeader[4] = (unsigned char)(width);
    infoHeader[5] = (unsigned char)(width >> 8);
    infoHeader[6] = (unsigned char)(width >> 16);
    infoHeader[7] = (unsigned char)(width >> 24);

    // height (양수 → bottom-up)
    infoHeader[8] = (unsigned char)(height);
    infoHeader[9] = (unsigned char)(height >> 8);
    infoHeader[10] = (unsigned char)(height >> 16);
    infoHeader[11] = (unsigned char)(height >> 24);

    // 원하면 biSizeImage도 채워주기 (필수는 아님)
    infoHeader[20] = (unsigned char)(imageSize);
    infoHeader[21] = (unsigned char)(imageSize >> 8);
    infoHeader[22] = (unsigned char)(imageSize >> 16);
    infoHeader[23] = (unsigned char)(imageSize >> 24);

    // 헤더 쓰기
    ofs.write(reinterpret_cast<const char*>(fileHeader), sizeof(fileHeader));
    ofs.write(reinterpret_cast<const char*>(infoHeader), sizeof(infoHeader));

    // -------------------------
    // 픽셀 데이터 쓰기 (bottom-up, BGR)
    // -------------------------
    std::vector<uint8_t> rowBuf(rowStride, 0);

    for (int y = 0; y < (int)height; ++y)
    {
        int srcY = height - 1 - y; // BMP는 아래에서 위로 저장

        for (uint32_t x = 0; x < width; ++x)
        {
            size_t idxFloat = (size_t)srcY * width + x;
            float v = data[idxFloat];

            v = std::clamp(v, 0.0f, 1.0f);
            uint8_t g = static_cast<uint8_t>(v * 255.0f);

            // BGR 순서
            size_t idxByte = x * bytesPerPixel;
            rowBuf[idxByte + 0] = g; // B
            rowBuf[idxByte + 1] = g; // G
            rowBuf[idxByte + 2] = g; // R
        }

        ofs.write(reinterpret_cast<const char*>(rowBuf.data()), rowStride);
    }

    ofs.close();
    std::cout << "[OK] BMP 저장 완료: " << filename << std::endl;
}

int main()
{
    Device device = VulkanApp::get().device();

    constexpr uint32_t channels = 3;
    auto [srcImage, width, height] = readImage<channels>(IMAGE_PATH);
    _ASSERT(width * height * channels == srcImage.size());

    ImageInfo info = { channels, height, width };

    Unet::s_ptr net = make_shared<Unet>(device, height, width, channels, 1, 1);

    auto preImage = pre_processing(info, srcImage);

    SafeTensorsParser parser(SAFE_TENSOR_PATH);
    set_Weight(net, parser);

    Tensor eval;
    eval = eval_unet(net, info, preImage);

    auto shape = eval.shape();   // {512, 512, 1}
    uint32_t H = shape[0];
    uint32_t W = shape[1];
    uint32_t C = shape[2];

    size_t elemCount = size_t(H) * size_t(W) * size_t(C);   // 512*512*1 = 262144
    size_t byteSize = elemCount * sizeof(float);

    vk::Buffer outBuffer = netGlobalDevice.createBuffer({
        byteSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        });

    vk::Buffer evalBuffer = eval.buffer();
    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuffer, evalBuffer)
        .end()
        .submit()
        .wait();

    std::vector<float> cpuData(elemCount);

    void* mapped = outBuffer.map();
    memcpy(cpuData.data(), mapped, byteSize);
    outBuffer.unmap();

    auto result = sigmoid(cpuData);

    saveFloatToBMP_0_255(result, W, H, "unet_output.bmp");

    std::cout << "다운로드 완료! 크기 = " << result.size() << " float" << std::endl;

    return 0;
}
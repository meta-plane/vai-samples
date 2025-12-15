#include "main.hpp"

using namespace std;
using namespace networks;
using namespace utils;

std::vector<float> pre_processing(const ImageInfo& info, const std::vector<uint8_t>& srcImage) 
{
    std::vector<float> dstImage(info.channels * info.height * info.width);
    
    for (size_t i = 0; i < srcImage.size(); ++i)
        dstImage[i] = (static_cast<float>(srcImage[i]) / 255.0);

    return dstImage;
}

std::vector<uint8_t> post_processing(const ImageInfo& info, const std::vector<float>& srcImage)
{
    std::vector<uint8_t> dstImage(info.channels * info.height * info.width);

    auto clamp = [](float v, float lo = 0, float hi = 255)
        {
            return v < lo ? lo : (v > hi ? hi : v);
		};

    for (size_t i = 0; i < srcImage.size(); ++i) 
    {
        dstImage[i] = static_cast<uint8_t>(clamp((srcImage[i] * 255)));
    }

    return dstImage;
}

Tensor eval_unet(Unet::w_ptr net, const ImageInfo& info, const std::vector<float>& srcImage)
{
    auto unet = net.lock();

    Tensor result;
    Tensor inputTensor = Tensor(info.height, info.width, info.channels).set(srcImage);
 
    result = (*unet)(inputTensor)[0];

    return result;
}

Tensor eval_unet(Unet::w_ptr net, const ImageInfo& info, const string& path)
{
    auto unet = net.lock();

    Tensor result;
    Tensor inputTensor = Tensor::fromFile(path);

    result = (*unet)(inputTensor)[0];

    return result;
}

Tensor eval_Testnet(std::weak_ptr<TestNet> net, const ImageInfo& info, const std::vector<float>& srcImage)
{
    auto testnet = net.lock();

    Tensor result;
    Tensor inputTensor = Tensor(info.height, info.width, info.channels).set(srcImage);

    result = (*testnet)(inputTensor)[0];

    return result;
}

float meanDifference(const std::vector<float>& v1,
    const std::vector<float>& v2)
{
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("벡터의 크기가 다릅니다.");
    }

    if (v1.empty()) {
        throw std::invalid_argument("벡터가 비어 있습니다.");
    }

    float sum = 0.0;

    for (size_t i = 0; i < v1.size(); ++i) {
        sum += std::abs(v1[i] - v2[i]);
    }

    float err = sum / v1.size();
    
    return err;
}

int main()
{
    Device device = VulkanApp::get().device();

    constexpr uint32_t channels = 3;
    auto [srcImage, width, height] = readImage<channels>(IMAGE_PATH);

    _ASSERT(width * height * channels == srcImage.size());
    ImageInfo info = { channels, height, width };
    auto preImage = pre_processing(info, srcImage);

    SafeTensorsParser parser(SAFE_TENSOR_PATH);
    Unet::s_ptr net = make_shared<Unet>(device, height, width, channels, 1, 1);
    net->setWeight(parser);

	TestNet::s_ptr testnet = make_shared<TestNet>(device, info, 1, 1);
	testnet->setWeight();

    Tensor eval;
    eval = eval_unet(net, info, "D:/VAI/vai-sample/100-UNet-seungwoo/python/image.bin");
    /*eval = eval_unet(net, info, preImage);*/
    /*eval = eval_Testnet(testnet, info, preImage);*/
    
	std::vector<float> cpuData = tensor_to_cpu(eval);

    /*Tensor resulte = Tensor::fromFile("D:\\VAI\\vai-sample\\100-UNet-seungwoo\\python\\workspace\\bin\\convTrans_result.bin");*/
    //Tensor resulte = Tensor::fromFile("D:\\VAI\\vai-sample\\100-UNet-seungwoo\\python\\debug_layers\\header_.bin");

    //auto s = resulte.shape(); // [b, c, h, w]
    //uint32_t b = s[0];
    //uint32_t c = s[1];
    //uint32_t h = s[2];
    //uint32_t w = s[3];

    //size_t elemCount = 1;
    //for (auto d : s) elemCount *= size_t(d);
    //
    //resulte.permute(0, 2, 3, 1);
    //
    //Tensor wt = resulte;
    //wt.reshape(h, w, c);

    //const float* p = wt.hostData();
    //std::vector<float> re(elemCount);
    //memcpy(re.data(), p, elemCount*sizeof(float));
    //
    //
    //float err = meanDifference(re, cpuData);
    //std:cout << "Error : " << err << std::endl;

    auto shape = eval.shape();// {512, 512, 1}
	ImageInfo outInfo = { shape[2], shape[1], shape[0] };

    auto postImage = post_processing(outInfo, cpuData);

    writeImage(OUTPUT_PATH, postImage, outInfo);

    return 0;
}
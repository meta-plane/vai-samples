#include "main.hpp"

using namespace std;
using namespace networks;
using namespace utils;

int main()
{
    Device device = VulkanApp::get().device();
    /*constexpr uint32_t channels = 3;

    auto [srcImage, width, height] = readImage<channels>(IMAGE_PATH);
    _ASSERT(width * height * channels == srcImage.size());

    Unet::u_ptr net = make_unique<Unet>(device, height, width, channels);*/

    Unet::u_ptr net = make_unique<Unet>(device, 512, 512, 3);

    return 0;
}
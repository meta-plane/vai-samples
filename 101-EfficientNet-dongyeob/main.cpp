#include <GLFW/glfw3.h>
#include "library/neuralNodes.h"
#include "networks/efficientNet.h"

const uint32_t WIDTH = 1200;
const uint32_t HEIGHT = 800;

GLFWwindow* createWindow()
{
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    return glfwCreateWindow(WIDTH, HEIGHT, "EfficientNet", nullptr, nullptr);
}

void test(const std::string& versionStr);

int main(int argc, char* argv[])
{
    glfwInit();
    
    std::string versionStr = "";
    if (argc > 1)
    {
        versionStr = argv[1];
    }

    test(versionStr);
    
    glfwTerminate();
    return 0;
}

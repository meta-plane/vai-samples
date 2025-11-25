#include <GLFW/glfw3.h>

const uint32_t WIDTH = 1200;
const uint32_t HEIGHT = 800;

GLFWwindow* createWindow()
{
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    return glfwCreateWindow(WIDTH, HEIGHT, "Vulkan App", nullptr, nullptr);
}

int main()
{
    glfwInit();
    //GLFWwindow* window = createWindow();
    
    void Run();
    Run();

    // void eval_adaptive_avgpool();
    // eval_adaptive_avgpool();

    // void eval_multiply();
    // eval_multiply();

    // void eval_hs();
    // eval_hs();

    // void eval_batchnorm2d();
    // eval_batchnorm2d();

    // void eval_channel_shuffle_concat();
    // eval_channel_shuffle_concat();

    // while (!glfwWindowShouldClose(window))
    // {
    //     glfwPollEvents();
    // }

    //glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

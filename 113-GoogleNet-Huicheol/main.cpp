#include <GLFW/glfw3.h>
#include <iostream>

void test(const char* imagePath = nullptr);

int main(int argc, char** argv)
{
    glfwInit();

    const char* imgPath = (argc > 1) ? argv[1] : nullptr;
    test(imgPath);

    glfwTerminate();
    return 0;
}

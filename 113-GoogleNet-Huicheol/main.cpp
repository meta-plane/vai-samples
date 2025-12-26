#include <GLFW/glfw3.h>
#include <iostream>

void test(const char* imagePath = nullptr, bool benchmark = false);

int main(int argc, char** argv)
{
    glfwInit();

    const char* imgPath = nullptr;
    bool benchmark = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--benchmark") {
            benchmark = true;
        } else {
            imgPath = argv[i];
        }
    }

    test(imgPath, benchmark);

    glfwTerminate();
    return 0;
}

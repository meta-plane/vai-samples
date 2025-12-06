#include <GLFW/glfw3.h>
#include <iostream>

int main()
{
    glfwInit();
    
    // Window creation might be handled in test() or vulkanApp.cpp depending on the exact pattern
    // For now, we keep the basic setup but delegate logic to test()
    
    void test();
    test();
    
    glfwTerminate();
    return 0;
}
#include <GLFW/glfw3.h>

#ifdef BENCHMARK_MODE
#include "utils/test_layers.h"
#include <iostream>
#include <string>

void printBenchmarkUsage()
{
    std::cout << "\n=== Benchmark Mode ===\n";
    std::cout << "To run specific benchmarks, modify main.cpp and uncomment the desired functions.\n";
    std::cout << "\nAvailable benchmarks:\n";
    std::cout << "\n[Basic Operation Nodes]\n";
    std::cout << "  - benchmarkBatchNorm()\n";
    std::cout << "  - benchmarkRelu6()\n";
    std::cout << "  - benchmarkAdd()\n";
    std::cout << "  - benchmarkMaxPooling()\n";
    std::cout << "  - benchmarkGlobalAvgPool()\n";
    std::cout << "  - benchmarkFullyConnected()\n";
    std::cout << "\n[Convolution Nodes]\n";
    std::cout << "  - benchmarkConvolution()\n";
    std::cout << "  - benchmarkDepthwiseConv()\n";
    std::cout << "  - benchmarkPointwiseConv()\n";
    std::cout << "\n[Composite Nodes]\n";
    std::cout << "  - benchmarkConvBnReLU6()\n";
    std::cout << "  - benchmarkInvertedResidualBlock()\n";
    std::cout << "\n[Run All]\n";
    std::cout << "  - benchmarkAllLayers()  (runs all benchmarks)\n\n";
}

int main()
{
    printBenchmarkUsage();

    // targeted benchmarks
    benchmarkAllLayers();
    // benchmarkDepthwiseConv();
    // benchmarkPointwiseConv();
    // benchmarkConvBnReLU6();
    // benchmarkInvertedResidualBlock();

    return 0;
}

#else
// Normal inference mode
int main()
{
    void test();
    test();

    return 0;
}
#endif
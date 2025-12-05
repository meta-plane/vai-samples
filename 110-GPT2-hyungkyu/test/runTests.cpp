/**
 * Test Runner - GPT-2 Layer Unit Tests
 */

#include "../core/globalContext.h"
#include "../core/vulkanApp.h"
#include "testBase.h"
#include "linearTest.h"
#include "layerNormTest.h"
#include <iostream>
#include <vector>
#include <memory>

using namespace vk;

void registerTests(std::vector<std::unique_ptr<ITest>>& tests) {
    tests.push_back(std::make_unique<LinearTest>("LinearNode - Small (2x3x4 -> 2x3x5)", 2, 3, 4, 5));
    tests.push_back(std::make_unique<LayerNormTest>("LayerNorm - Standard (2x3x8)", 2, 3, 8));
}

int main() {
    try {
        std::cout << "╔════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║  GPT-2 Unit Tests - Layer Testing                    ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════╝\n" << std::endl;

        std::vector<std::unique_ptr<ITest>> tests;
        registerTests(tests);

        int total_tests = 0;
        int passed_tests = 0;

        for (auto& test : tests) {
            if (test->execute()) {
                passed_tests++;
            }
            total_tests++;
        }

        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "OVERALL TEST SUMMARY" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "Total tests run: " << total_tests << std::endl;
        std::cout << "Tests passed: " << passed_tests << std::endl;
        std::cout << "Tests failed: " << (total_tests - passed_tests) << std::endl;

        if (passed_tests == total_tests) {
            std::cout << "\n✓ ALL TESTS PASSED!" << std::endl;
            return 0;
        } else {
            std::cout << "\n✗ SOME TESTS FAILED" << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cout << "\n✗ Fatal error: " << e.what() << std::endl;
        return 1;
    }
}

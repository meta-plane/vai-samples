/**
 * Test Runner - Simplified for debugging
 */

#include "../core/globalContext.h"
#include "../core/vulkanApp.h"
#include "linearTest.h"
#include <iostream>

using namespace vk;

int main() {
    try {
        std::cout << "╔════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║  GPT-2 Test Framework - Linear Test Only             ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════╝\n" << std::endl;

        int total_tests = 0;
        int passed_tests = 0;

        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "LinearNode Tests" << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        {
            LinearTest test("LinearNode - Small (2x3x4 -> 2x3x5)", 2, 3, 4, 5);
            if (test.execute()) passed_tests++;
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

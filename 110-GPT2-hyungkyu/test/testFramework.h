#ifndef TEST_FRAMEWORK_H
#define TEST_FRAMEWORK_H

#include "../core/neuralNet.h"
#include "../core/vulkanApp.h"
#include "../core/globalContext.h"
#include "testHelpers.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <functional>
#include <vector>
#include <string>
#include <cmath>

using namespace vk;

/**
 * Helper: Read tensor data to host memory
 */
inline std::vector<float> readTensorData(const Tensor& tensor) {
    if (!tensor.validShape()) {
        throw std::runtime_error("Invalid tensor shape");
    }

    uint32_t num_elements = tensor.numElements();

    // Create host-visible buffer
    Buffer outBuffer = netGlobalDevice.createBuffer({
        .size = num_elements * sizeof(float),
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .reqMemProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });

    // Copy tensor data to host buffer
    netGlobalDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuffer, tensor.buffer())
        .end()
        .submit()
        .wait();

    // Map and read data
    float* data = (float*)outBuffer.map();
    return std::vector<float>(data, data + num_elements);
}

/**
 * Test Result
 * Stores the outcome of a test execution
 */
struct TestResult {
    std::string test_name;
    bool passed = false;
    std::string error_message;
    double execution_time_ms = 0.0;

    void print() const {
        std::cout << (passed ? "✓ PASS" : "✗ FAIL") << ": " << test_name;
        if (execution_time_ms > 0) {
            std::cout << " (" << std::fixed << std::setprecision(2)
                      << execution_time_ms << " ms)";
        }
        std::cout << std::endl;

        if (!passed && !error_message.empty()) {
            std::cout << "  Error: " << error_message << std::endl;
        }
    }
};

/**
 * Test Assertions
 * Common assertion utilities for testing
 */
class TestAssert {
public:
    // Assert tensors are equal within tolerance
    static void assertEqual(const Tensor& actual, const Tensor& expected,
                           float tolerance = 1e-4f, const std::string& msg = "") {
        if (!actual.validShape() || !expected.validShape()) {
            throw std::runtime_error("Invalid tensor shape: " + msg);
        }

        if (actual.shape() != expected.shape()) {
            throw std::runtime_error("Shape mismatch: " + msg);
        }

        std::vector<float> actual_data = readTensorData(actual);
        std::vector<float> expected_data = readTensorData(expected);

        for (size_t i = 0; i < actual_data.size(); ++i) {
            float diff = std::abs(actual_data[i] - expected_data[i]);
            if (diff > tolerance) {
                throw std::runtime_error(
                    "Value mismatch at index " + std::to_string(i) +
                    ": expected " + std::to_string(expected_data[i]) +
                    ", got " + std::to_string(actual_data[i]) +
                    " (diff: " + std::to_string(diff) + ") " + msg
                );
            }
        }
    }

    // Assert condition is true
    static void assertTrue(bool condition, const std::string& msg = "") {
        if (!condition) {
            throw std::runtime_error("Assertion failed: " + msg);
        }
    }

    // Assert values are equal within tolerance
    static void assertClose(float actual, float expected,
                           float tolerance = 1e-4f, const std::string& msg = "") {
        float diff = std::abs(actual - expected);
        if (diff > tolerance) {
            throw std::runtime_error(
                "Value mismatch: expected " + std::to_string(expected) +
                ", got " + std::to_string(actual) +
                " (diff: " + std::to_string(diff) + ") " + msg
            );
        }
    }

    // Assert shape matches
    static void assertShape(const Tensor& tensor, const std::vector<uint32_t>& expected_shape,
                           const std::string& msg = "") {
        if (!tensor.validShape()) {
            throw std::runtime_error("Invalid tensor shape: " + msg);
        }

        if (tensor.shape() != expected_shape) {
            std::string actual_str = shapeToString(tensor.shape());
            std::string expected_str = shapeToString(expected_shape);
            throw std::runtime_error(
                "Shape mismatch: expected " + expected_str +
                ", got " + actual_str + " " + msg
            );
        }
    }

private:
    static std::string shapeToString(const std::vector<uint32_t>& shape) {
        std::string result = "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            result += std::to_string(shape[i]);
            if (i < shape.size() - 1) result += ", ";
        }
        result += "]";
        return result;
    }
};

/**
 * Base Test Class
 * Provides common test infrastructure
 */
class BaseTest {
protected:
    std::string test_name;

public:
    BaseTest(const std::string& name) : test_name(name) {}

    virtual ~BaseTest() = default;

    // Test lifecycle methods
    virtual void setup() {}
    virtual void run() = 0;  // Must be implemented by derived classes
    virtual void teardown() {}

    // Execute the test with timing and error handling
    TestResult execute() {
        TestResult result;
        result.test_name = test_name;

        std::cout << "\n========================================" << std::endl;
        std::cout << "Test: " << test_name << std::endl;
        std::cout << "========================================" << std::endl;

        try {
            auto start = std::chrono::high_resolution_clock::now();

            setup();
            run();
            teardown();

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            result.execution_time_ms = duration.count() / 1000.0;

            result.passed = true;
            std::cout << "✓ Test completed successfully" << std::endl;

        } catch (const std::exception& e) {
            result.passed = false;
            result.error_message = e.what();
            std::cout << "✗ Test failed: " << e.what() << std::endl;
        }

        return result;
    }
};

/**
 * Layer Test Class
 * Template for testing individual layers
 */
template<typename LayerType>
class LayerTest : public BaseTest {
protected:
    std::unique_ptr<LayerType> layer;

public:
    LayerTest(const std::string& name) : BaseTest(name) {}

    virtual ~LayerTest() = default;

    // Override these in derived classes
    virtual void createLayer() = 0;
    virtual void prepareInputs() = 0;
    virtual void runLayer() = 0;
    virtual void verifyOutputs() = 0;

    void run() override {
        createLayer();
        prepareInputs();
        runLayer();
        verifyOutputs();
    }
};

/**
 * Test Suite
 * Groups multiple tests together
 */
class TestSuite {
private:
    std::string suite_name;
    std::vector<std::function<TestResult()>> tests;

public:
    TestSuite(const std::string& name) : suite_name(name) {}

    // Add a test to the suite
    void addTest(std::function<TestResult()> test_func) {
        tests.push_back(test_func);
    }

    // Run all tests in the suite
    std::vector<TestResult> runAll() {
        std::cout << "\n╔════════════════════════════════════════╗" << std::endl;
        std::cout << "║  Test Suite: " << std::left << std::setw(24) << suite_name << "║" << std::endl;
        std::cout << "╚════════════════════════════════════════╝" << std::endl;

        std::vector<TestResult> results;

        for (auto& test : tests) {
            results.push_back(test());
        }

        // Print summary
        printSummary(results);

        return results;
    }

private:
    void printSummary(const std::vector<TestResult>& results) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Test Suite Summary: " << suite_name << std::endl;
        std::cout << "========================================" << std::endl;

        int passed = 0;
        int failed = 0;
        double total_time = 0.0;

        for (const auto& result : results) {
            result.print();
            if (result.passed) {
                passed++;
            } else {
                failed++;
            }
            total_time += result.execution_time_ms;
        }

        std::cout << "\nTotal: " << results.size() << " tests" << std::endl;
        std::cout << "Passed: " << passed << std::endl;
        std::cout << "Failed: " << failed << std::endl;
        std::cout << "Total time: " << std::fixed << std::setprecision(2)
                  << total_time << " ms" << std::endl;

        if (failed == 0) {
            std::cout << "\n✓ All tests passed!" << std::endl;
        } else {
            std::cout << "\n✗ Some tests failed" << std::endl;
        }
        std::cout << "========================================\n" << std::endl;
    }
};

#endif // TEST_FRAMEWORK_H

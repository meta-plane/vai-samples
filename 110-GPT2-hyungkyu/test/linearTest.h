#ifndef LINEAR_TEST_H
#define LINEAR_TEST_H

#include "testBase.h"
#include "../model/attention/attentionNode.h"

/**
 * LinearNode Test
 *
 * Tests the LinearNode layer (matrix multiplication Y = X @ W^T)
 * Uses smart pointer for automatic memory management
 *
 * Example usage:
 *   LinearTest test("LinearNode Basic Test", 2, 3, 4, 5);
 *   bool passed = test.execute();
 */
class LinearTest : public TestBase<LinearNode> {
private:
    // Test configuration
    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t inFeatures;
    uint32_t outFeatures;

public:
    LinearTest(const std::string& name,
               uint32_t B = 2,
               uint32_t S = 3,
               uint32_t in_feat = 4,
               uint32_t out_feat = 5);

    void createGraph() override;
    void setupInputs() override;
    void setupParameters() override;
    void setupExpectedOutputs() override;
};

#endif // LINEAR_TEST_H

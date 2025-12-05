#ifndef LAYER_NORM_TEST_H
#define LAYER_NORM_TEST_H

#include "testBase.h"
#include "../model/transformerBlock/transformer.h"

/**
 * LayerNormNode Test
 *
 * Tests the LayerNormNode layer (Layer Normalization)
 * Formula: output = scale * (x - mean) / sqrt(var + eps) + shift
 *
 * Example usage:
 *   LayerNormTest test("LayerNorm Basic Test", 2, 3, 8);
 *   bool passed = test.execute();
 */
class LayerNormTest : public TestBase<LayerNormNode> {
private:
    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t dModel;
    float eps;

public:
    LayerNormTest(const std::string& name,
                  uint32_t B = 2,
                  uint32_t S = 3,
                  uint32_t D = 8,
                  float eps = 1e-5f);

    void createGraph() override;
    void setupInputs() override;
    void setupParameters() override;
    void setupExpectedOutputs() override;
};

#endif // LAYER_NORM_TEST_H

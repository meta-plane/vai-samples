#include "linearTest.h"

LinearTest::LinearTest(const std::string& name,
                       uint32_t B,
                       uint32_t S,
                       uint32_t in_feat,
                       uint32_t out_feat)
    : TestBase<LinearNode>(name),
      batchSize(B),
      seqLen(S),
      inFeatures(in_feat),
      outFeatures(out_feat) {}

void LinearTest::createGraph() {
    targetGraph = std::make_unique<LinearNode>(inFeatures, outFeatures);
}

void LinearTest::setupInputs() {
    // Set CPU input tensor shape and data
    cpuInput.shape = {batchSize, seqLen, inFeatures};
    cpuInput.data.resize(batchSize * seqLen * inFeatures);

    // Fill with simple pattern for easy verification
    for (uint32_t b = 0; b < batchSize; ++b) {
        for (uint32_t s = 0; s < seqLen; ++s) {
            for (uint32_t i = 0; i < inFeatures; ++i) {
                uint32_t idx = b * (seqLen * inFeatures) + s * inFeatures + i;
                cpuInput.data[idx] = b * 100.0f + s * 10.0f + i * 1.0f;
            }
        }
    }
}

void LinearTest::setupParameters() {
    // Create weight parameter
    CPUTensorData weight;
    weight.slotName = "weight";
    weight.shape = {outFeatures, inFeatures};
    weight.data.resize(outFeatures * inFeatures);

    // Fill with simple pattern
    for (uint32_t o = 0; o < outFeatures; ++o) {
        for (uint32_t i = 0; i < inFeatures; ++i) {
            uint32_t idx = o * inFeatures + i;
            weight.data[idx] = (o * 10.0f + i) * 0.1f;
        }
    }

    cpuParameters.push_back(weight);
}

void LinearTest::setupExpectedOutputs() {
    // Set CPU expected output shape
    cpuExpectedOutput.shape = {batchSize, seqLen, outFeatures};
    cpuExpectedOutput.data.resize(batchSize * seqLen * outFeatures, 0.0f);

    // Compute Y = X @ W^T manually
    for (uint32_t b = 0; b < batchSize; ++b) {
        for (uint32_t s = 0; s < seqLen; ++s) {
            for (uint32_t o = 0; o < outFeatures; ++o) {
                float sum = 0.0f;
                for (uint32_t i = 0; i < inFeatures; ++i) {
                    float x_val = cpuInput.data[b * (seqLen * inFeatures) + s * inFeatures + i];
                    float w_val = cpuParameters[0].data[o * inFeatures + i];
                    sum += x_val * w_val;
                }
                cpuExpectedOutput.data[b * (seqLen * outFeatures) + s * outFeatures + o] = sum;
            }
        }
    }
}

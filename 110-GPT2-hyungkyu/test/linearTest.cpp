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
    std::cout << "Creating LinearNode graph..." << std::endl;
    std::cout << "  Input shape: [" << batchSize << ", " << seqLen << ", " << inFeatures << "]" << std::endl;
    std::cout << "  Output shape: [" << batchSize << ", " << seqLen << ", " << outFeatures << "]" << std::endl;

    // Create LinearNode with smart pointer
    targetGraph = std::make_unique<LinearNode>(inFeatures, outFeatures);
}

void LinearTest::setupInputs() {
    std::cout << "Setting up inputs..." << std::endl;

    // Create input tensor [B, S, inFeatures]
    inputData.resize(batchSize * seqLen * inFeatures);

    // Fill with simple pattern for easy verification
    for (uint32_t b = 0; b < batchSize; ++b) {
        for (uint32_t s = 0; s < seqLen; ++s) {
            for (uint32_t i = 0; i < inFeatures; ++i) {
                uint32_t idx = b * (seqLen * inFeatures) + s * inFeatures + i;
                inputData[idx] = b * 100.0f + s * 10.0f + i * 1.0f;
            }
        }
    }

    Tensor input = createInputTensor({batchSize, seqLen, inFeatures}, inputData);
    inputTensors.push_back(input);
    (*targetGraph)["in0"] = input;

    // Create weight tensor [outFeatures, inFeatures]
    weightData.resize(outFeatures * inFeatures);

    // Fill with simple pattern
    for (uint32_t o = 0; o < outFeatures; ++o) {
        for (uint32_t i = 0; i < inFeatures; ++i) {
            uint32_t idx = o * inFeatures + i;
            weightData[idx] = (o * 10.0f + i) * 0.1f;
        }
    }

    Tensor weight = createInputTensor({outFeatures, inFeatures}, weightData);
    (*targetGraph)["weight"] = weight;

    std::cout << "  Input and weight tensors set" << std::endl;
}

void LinearTest::setupExpectedOutputs() {
    std::cout << "Computing expected outputs..." << std::endl;

    // Compute Y = X @ W^T manually using stored data
    std::vector<float> expected_data(batchSize * seqLen * outFeatures, 0.0f);

    for (uint32_t b = 0; b < batchSize; ++b) {
        for (uint32_t s = 0; s < seqLen; ++s) {
            for (uint32_t o = 0; o < outFeatures; ++o) {
                float sum = 0.0f;
                for (uint32_t i = 0; i < inFeatures; ++i) {
                    float x_val = inputData[b * (seqLen * inFeatures) + s * inFeatures + i];
                    float w_val = weightData[o * inFeatures + i];
                    sum += x_val * w_val;
                }
                expected_data[b * (seqLen * outFeatures) + s * outFeatures + o] = sum;
            }
        }
    }

    Tensor expected = createExpectedTensor({batchSize, seqLen, outFeatures}, expected_data);
    expectedOutputTensors.push_back(expected);

    std::cout << "  Expected output computed" << std::endl;
}

void LinearTest::verifyResults() {
    std::cout << "Verifying results..." << std::endl;
    verifyAllOutputs();
    std::cout << "  All outputs match expected values" << std::endl;
}

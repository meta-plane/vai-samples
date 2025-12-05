#include "layerNormTest.h"
#include "../core/jsonParser.h"

LayerNormTest::LayerNormTest(const std::string& name,
                             uint32_t B,
                             uint32_t S,
                             uint32_t D,
                             float eps)
    : TestBase<LayerNormNode>(name),
      batchSize(B),
      seqLen(S),
      dModel(D),
      eps(eps) {}

void LayerNormTest::createGraph() {
    targetGraph = std::make_unique<LayerNormNode>(dModel, eps);
}

void LayerNormTest::setupInputs() {
    std::string testDataPath = PROJECT_CURRENT_DIR;
    testDataPath += "/assets/test_data/layer_norm_test_data.json";

    JsonParser json(testDataPath.c_str());

    std::vector<uint32_t> inputShape;
    cpuInput.data = json["input"].parseNDArray(inputShape);

    if (inputShape.size() != 3 || inputShape[0] != batchSize ||
        inputShape[1] != seqLen || inputShape[2] != dModel) {
        throw std::runtime_error("Test data dimensions don't match");
    }

    cpuInput.shape = {batchSize, seqLen, dModel};
}

void LayerNormTest::setupParameters() {
    std::string testDataPath = PROJECT_CURRENT_DIR;
    testDataPath += "/assets/test_data/layer_norm_test_data.json";

    JsonParser json(testDataPath.c_str());

    CPUTensorData scale;
    scale.slotName = "scale";
    std::vector<uint32_t> scaleShape;
    scale.data = json["scale"].parseNDArray(scaleShape);
    scale.shape = {dModel};

    CPUTensorData shift;
    shift.slotName = "shift";
    std::vector<uint32_t> shiftShape;
    shift.data = json["shift"].parseNDArray(shiftShape);
    shift.shape = {dModel};

    cpuParameters.push_back(scale);
    cpuParameters.push_back(shift);
}

void LayerNormTest::setupExpectedOutputs() {
    std::string testDataPath = PROJECT_CURRENT_DIR;
    testDataPath += "/assets/test_data/layer_norm_test_data.json";

    JsonParser json(testDataPath.c_str());

    std::vector<uint32_t> outputShape;
    cpuExpectedOutput.data = json["output"].parseNDArray(outputShape);
    cpuExpectedOutput.shape = {batchSize, seqLen, dModel};
}

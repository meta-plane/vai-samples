#pragma once
#include <random>
#include "../library/tensor.h"


Tensor makeRandomTensor(const std::vector<uint32_t>& shape, std::mt19937& gen, float stddev = 0.02f);
Tensor makeConstTensor(const std::vector<uint32_t>& shape, float val);
Tensor makeConstTensor(uint32_t size, float val);

bool allClose(const float* data, const float* expected, int n, float eps = 1e-3f);

void testRelu6();
void testDepthwiseConv();
void testPointwiseConv();
void testGlobalAvgPool();
void testConvBnReLU6();
void testInvertedResidualBlock();
void testSimplePipeline();
void testLoadWeights();
void testImageClassification();
void testFullClassification();
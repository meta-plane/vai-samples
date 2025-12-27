#pragma once
#include <random>
#include "../library/tensor.h"


Tensor makeRandomTensor(const std::vector<uint32_t>& shape, std::mt19937& gen, float stddev = 0.02f);
Tensor makeConstTensor(const std::vector<uint32_t>& shape, float val);
Tensor makeConstTensor(uint32_t size, float val);

bool allClose(const float* data, const float* expected, int n, float eps = 1e-3f);

// Benchmark functions - test cases defined internally
// Basic operation nodes
void benchmarkBatchNorm();
void benchmarkRelu6();
void benchmarkAdd();
void benchmarkMaxPooling();
void benchmarkGlobalAvgPool();
void benchmarkFullyConnected();

// Convolution nodes
void benchmarkConvolution();
void benchmarkDepthwiseConv();
void benchmarkPointwiseConv();

// Composite nodes
void benchmarkConvBnReLU6();
void benchmarkInvertedResidualBlock();

// Run all benchmarks
void benchmarkAllLayers();
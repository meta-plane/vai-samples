#include "../core/jsonParser.h"
#include "../core/tensor.h"

auto readImage(const char* filename) -> std::tuple<std::vector<uint8_t>, uint32_t, uint32_t>; // -> (image data, width, height)
Tensor makeConstTensor(const std::vector<uint32_t>& shape, float val);  
Tensor makeConstTensor(uint32_t size, float val);

void softmax(float* data, int size);
std::vector<int> getTopK(const float* data, int size, int k);

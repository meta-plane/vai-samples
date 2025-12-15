

#include <iostream>
#include <algorithm>
#include <vector>
#include <string>

#include <fstream>
#include <random>
#include <cmath>

#include "library/neuralNet.h"
#include "library/tensor.h"
#include "library/vulkanApp.h"

#include "safeTensor/safeTensorsParser.h"

#include "networks/include/unet.hpp"

#include "utils.hpp"

#define IMAGE_PATH "D:/VAI/images/image.png"

#define LABEL_PATH "D:/VAI/images/label.png"

#define OUTPUT_PATH "D:/VAI/images/output.png"

#define SAFE_TENSOR_PATH "D:/VAI/weight/unet.safetensors"
#include "../../library/neuralNet.h"
#include "../../library/neuralNodes.h"

class ConvBnRelu : public NodeGroup 
{
	uint32_t			C, F, K;

	ConvolutionNode		conv_;
	BatchNormNode		bn_;
	ReluNode			relu_;

public:
	ConvBnRelu(uint32_t inChannels, uint32_t outChannels, uint32_t kernel)
		: C(inChannels), F(outChannels), K(kernel), conv_(inChannels, outChannels, kernel)
	{
		conv_ - bn_ - relu_;
		defineSlot("in0", conv_.slot("in0"));
		defineSlot("out0", relu_.slot("out0"));
	}

	Tensor& operator[](const std::string& name)
	{
		return (*this)[name];
	}
};

class doubleConvBnRelu : public NodeGroup 
{
	uint32_t			C1, F1, K1, C2, F2, K2;
	ConvBnRelu			conv1_;
	ConvBnRelu			conv2_;

public:
	doubleConvBnRelu(uint32_t in_c1, uint32_t out_c1, uint32_t kernel_1, uint32_t in_c2, uint32_t out_c2, uint32_t kernel_2)
		: C1(in_c1), F1(out_c1), K1(kernel_1), C2(in_c2), F2(out_c2), K2(kernel_2),
		conv1_(in_c1, out_c1, kernel_1), conv2_(in_c2, out_c2, kernel_2)
	{
		conv1_ - conv2_;
		defineSlot("in0", conv1_.slot("in0"));
		defineSlot("out0", conv2_.slot("out0"));
	}

	Tensor& operator[](const std::string& name)
	{
		return (*this)[name];
	}
};
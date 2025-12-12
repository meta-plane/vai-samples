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
		// 1. Convolution 파라미터
		if (name == "weight")
		{
			return conv_["weight"];
		}
		else if (name == "bias")
		{
			// Conv에 bias가 있다면 반환, 없다면 예외 처리 필요
			return conv_["bias"];
		}

		//// 2. BatchNorm 파라미터 (일반적인 BN 파라미터 매핑)
		// Gamma (Scale Factor)
		else if (name == "gamma")
		{
			return bn_["gamma"]; // 보통 BN 노드 내부는 weight라는 키를 사용할 가능성이 높음
		}
		// Beta (Shift Factor)
		else if (name == "beta")
		{
			return bn_["beta"];   // 보통 BN 노드 내부는 bias라는 키를 사용할 가능성이 높음
		}
		// Running Mean
		else if (name == "mean")
		{
			return bn_["mean"];
		}
		// Running Variance
		else if (name == "variance")
		{
			return bn_["variance"];
		}

		// 3. 예외 처리: 알 수 없는 키가 들어온 경우
		throw std::runtime_error("ConvBnRelu: Unknown parameter name '" + name + "'");
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
		// 구분자 "." 의 위치를 찾음
		size_t delimiterPos = name.find(".");

		if (delimiterPos == std::string::npos)
		{
			throw std::runtime_error("DoubleConvBnRelu: Invalid key format. Use 'prefix.param' (e.g., '0.weight')");
		}

		// 접두어(Prefix)와 나머지 이름(SubName) 분리
		std::string prefix = name.substr(0, delimiterPos); // "0" or "1"
		std::string subName = name.substr(delimiterPos + 1); // "weight", "gamma" ...

		// 접두어에 따라 하위 블록의 operator[] 호출 (재귀적 구조)
		if (prefix == "0")
		{
			return conv1_[subName];
		}
		else if (prefix == "1")
		{
			return conv2_[subName];
		}
		else
		{
			throw std::runtime_error("DoubleConvBnRelu: Unknown block prefix '" + prefix + "'");
		};
	}
};
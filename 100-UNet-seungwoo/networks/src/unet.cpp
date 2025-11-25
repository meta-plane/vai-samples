#include "../include/unet.hpp"

using namespace networks;
using namespace std;


;
Unet::Unet(Device& device, uint32_t  height, uint32_t width, uint32_t channel, uint32_t numInputs, uint32_t numOutputs)
	: NeuralNet(device, numInputs, numOutputs)
	,device_(device), height_(height), width_(width), channel_(channel), numInputs_(numInputs), numOutputs_(numOutputs)
{
	//Encoder
	encoderConv1_ = std::make_unique<doubleConvBnRelu>(numInputs_, 64, 3, 64, 64, 3);
	pool1_ = std::make_unique<MaxPoolingNode>(2);

	encoderConv2_ = std::make_unique<doubleConvBnRelu>(64, 128, 3, 128, 128, 3);
	pool2_ = std::make_unique<MaxPoolingNode>(2);

	encoderConv3_ = std::make_unique<doubleConvBnRelu>(128, 256, 3, 256, 256, 3);
	pool3_ = std::make_unique<MaxPoolingNode>(2);

	encoderConv4_ = std::make_unique<doubleConvBnRelu>(256, 512, 3, 512, 512, 3);
	pool4_ = std::make_unique<MaxPoolingNode>(2);

	//bottleneck
	bottleneck_ = std::make_unique<doubleConvBnRelu>(512, 1024, 3, 1024, 1024, 3);

	//Decoder
	upConv4_ = std::make_unique<ConvTransposeNode>(1024, 512, 3);
	concat4_ = std::make_unique<ConCatNode>(0);
	decoderConv4_ = std::make_unique<doubleConvBnRelu>(1024, 512, 3, 512, 512, 3);

	upConv3_ = std::make_unique<ConvTransposeNode>(512, 256, 3);
	concat3_ = std::make_unique<ConCatNode>(0);
	decoderConv3_ = std::make_unique<doubleConvBnRelu>(512, 256, 3, 256, 256, 3);

	upConv2_ = std::make_unique<ConvTransposeNode>(256, 128, 3);
	concat2_ = std::make_unique<ConCatNode>(0);
	decoderConv2_ = std::make_unique<doubleConvBnRelu>(256, 128, 3, 128, 128, 3);

	upConv1_ = std::make_unique<ConvTransposeNode>(128, 64, 3);
	concat1_ = std::make_unique<ConCatNode>(0);
	decoderConv1_ = std::make_unique<doubleConvBnRelu>(128, 64, 3, 64, 64, 3);

	//Header
	header_ = std::make_unique<ConvolutionNode>(64, numOutputs_, 3);

	//model build
	input(0) - *encoderConv1_ - *pool1_ - *encoderConv2_ - *pool2_ - *encoderConv3_ - *pool3_ - *encoderConv4_ - *pool4_ - *bottleneck_;

	*bottleneck_ - *upConv4_ - ("in1" / *concat4_);
	*encoderConv4_ - ("in0" / *concat4_) - *decoderConv4_;

	*decoderConv4_ - *upConv3_ - ("in1" / *concat3_);
	*encoderConv3_ - ("in0" / *concat3_) - *decoderConv3_;

	*decoderConv3_ - *upConv2_ - ("in1" / *concat2_);
	*encoderConv2_ - ("in0" / *concat2_) - *decoderConv2_;

	*decoderConv2_ - *upConv1_ - ("in1" / *concat1_);
	*encoderConv1_ - ("in0" / *concat1_) - *decoderConv1_;

	*decoderConv1_ - *header_ - output(0);

	cout << "Unet Init Done" << endl;
}

Unet::~Unet() 
{

}

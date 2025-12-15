#include "../include/unet.hpp"

using namespace networks;
using namespace utils;
using namespace std;

Unet::Unet(Device& device, uint32_t  height, uint32_t width, uint32_t channel, uint32_t numInputs, uint32_t numOutputs)
	: NeuralNet(device, numInputs, numOutputs)
	,device_(device), height_(height), width_(width), channel_(channel), numInputs_(numInputs), numOutputs_(numOutputs)
{
	//Encoder
	encoderConv1_ = std::make_unique<doubleConvBnRelu>(3, 64, 3, 64, 64, 3);
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
	upConv4_ = std::make_unique<ConvTransposeNode>(1024, 512, 2);
	concat4_ = std::make_unique<ConcatNode>();
	decoderConv4_ = std::make_unique<doubleConvBnRelu>(1024, 512, 3, 512, 512, 3);

	upConv3_ = std::make_unique<ConvTransposeNode>(512, 256, 2);
	concat3_ = std::make_unique<ConcatNode>();
	decoderConv3_ = std::make_unique<doubleConvBnRelu>(512, 256, 3, 256, 256, 3);

	upConv2_ = std::make_unique<ConvTransposeNode>(256, 128, 2);
	concat2_ = std::make_unique<ConcatNode>();
	decoderConv2_ = std::make_unique<doubleConvBnRelu>(256, 128, 3, 128, 128, 3);

	upConv1_ = std::make_unique<ConvTransposeNode>(128, 64, 2);
	concat1_ = std::make_unique<ConcatNode>();
	decoderConv1_ = std::make_unique<doubleConvBnRelu>(128, 64, 3, 64, 64, 3);

	//Header
	header_ = std::make_unique<ConvolutionNode>(64, 1, 3);

	//Sigmoid
	sigmoid_ = std::make_unique<SigmoidNode>();

	//model build
	input(0) - *encoderConv1_ - *pool1_ - *encoderConv2_ - *pool2_ - *encoderConv3_ - *pool3_ - *encoderConv4_ - *pool4_ - *bottleneck_;

	*bottleneck_ - *upConv4_ - ("in0" / *concat4_);
	*encoderConv4_ - ("in1" / *concat4_);
	
	*concat4_ - *decoderConv4_;

	*decoderConv4_ - *upConv3_ - ("in0" / *concat3_);
	*encoderConv3_ - ("in1" / *concat3_);

	*concat3_ - *decoderConv3_;

	*decoderConv3_ - *upConv2_ - ("in0" / *concat2_);
	*encoderConv2_ - ("in1" / *concat2_);

	*concat2_ - *decoderConv2_;

	*decoderConv2_ - *upConv1_ - ("in0" / *concat1_);
	*encoderConv1_ - ("in1" / *concat1_);
	
	*concat1_ - *decoderConv1_;

	*decoderConv1_ - *header_ - *sigmoid_ - output(0);

	cout << "Unet Init Done" << endl;
}

void Unet::setWeight(SafeTensorsParser& parser) 
{
	/////////////////////////////////encoder 1/////////////////////////////////
	(*encoderConv1_)["0.weight"]	= Tensor(parser["encoderConv1_.ConvBnRelu_1.conv.weight"]);
	(*encoderConv1_)["0.bias"]		= Tensor(parser["encoderConv1_.ConvBnRelu_1.conv.bias"]);
	(*encoderConv1_)["0.gamma"]		= Tensor(parser["encoderConv1_.ConvBnRelu_1.bn.weight"]);
	(*encoderConv1_)["0.beta"]		= Tensor(parser["encoderConv1_.ConvBnRelu_1.bn.bias"]);
	(*encoderConv1_)["0.mean"]		= Tensor(parser["encoderConv1_.ConvBnRelu_1.bn.running_mean"]);
	(*encoderConv1_)["0.variance"]	= Tensor(parser["encoderConv1_.ConvBnRelu_1.bn.running_var"]);

	(*encoderConv1_)["1.weight"]	= Tensor(parser["encoderConv1_.ConvBnRelu_2.conv.weight"]);
	(*encoderConv1_)["1.bias"]		= Tensor(parser["encoderConv1_.ConvBnRelu_2.conv.bias"]);
	(*encoderConv1_)["1.gamma"]		= Tensor(parser["encoderConv1_.ConvBnRelu_2.bn.weight"]);
	(*encoderConv1_)["1.beta"]		= Tensor(parser["encoderConv1_.ConvBnRelu_2.bn.bias"]);
	(*encoderConv1_)["1.mean"]		= Tensor(parser["encoderConv1_.ConvBnRelu_2.bn.running_mean"]);
	(*encoderConv1_)["1.variance"]	= Tensor(parser["encoderConv1_.ConvBnRelu_2.bn.running_var"]);

	/////////////////////////////////encoder 2/////////////////////////////////
	(*encoderConv2_)["0.weight"]	= Tensor(parser["encoderConv2_.ConvBnRelu_1.conv.weight"]);
	(*encoderConv2_)["0.bias"]		= Tensor(parser["encoderConv2_.ConvBnRelu_1.conv.bias"]);
	(*encoderConv2_)["0.gamma"]		= Tensor(parser["encoderConv2_.ConvBnRelu_1.bn.weight"]);
	(*encoderConv2_)["0.beta"]		= Tensor(parser["encoderConv2_.ConvBnRelu_1.bn.bias"]);
	(*encoderConv2_)["0.mean"]		= Tensor(parser["encoderConv2_.ConvBnRelu_1.bn.running_mean"]);
	(*encoderConv2_)["0.variance"]	= Tensor(parser["encoderConv2_.ConvBnRelu_1.bn.running_var"]);

	(*encoderConv2_)["1.weight"]	= Tensor(parser["encoderConv2_.ConvBnRelu_2.conv.weight"]);
	(*encoderConv2_)["1.bias"]		= Tensor(parser["encoderConv2_.ConvBnRelu_2.conv.bias"]);
	(*encoderConv2_)["1.gamma"]		= Tensor(parser["encoderConv2_.ConvBnRelu_2.bn.weight"]);
	(*encoderConv2_)["1.beta"]		= Tensor(parser["encoderConv2_.ConvBnRelu_2.bn.bias"]);
	(*encoderConv2_)["1.mean"]		= Tensor(parser["encoderConv2_.ConvBnRelu_2.bn.running_mean"]);
	(*encoderConv2_)["1.variance"]	= Tensor(parser["encoderConv2_.ConvBnRelu_2.bn.running_var"]);

	/////////////////////////////////encoder 3/////////////////////////////////
	(*encoderConv3_)["0.weight"]	= Tensor(parser["encoderConv3_.ConvBnRelu_1.conv.weight"]);
	(*encoderConv3_)["0.bias"]		= Tensor(parser["encoderConv3_.ConvBnRelu_1.conv.bias"]);
	(*encoderConv3_)["0.gamma"]		= Tensor(parser["encoderConv3_.ConvBnRelu_1.bn.weight"]);
	(*encoderConv3_)["0.beta"]		= Tensor(parser["encoderConv3_.ConvBnRelu_1.bn.bias"]);
	(*encoderConv3_)["0.mean"]		= Tensor(parser["encoderConv3_.ConvBnRelu_1.bn.running_mean"]);
	(*encoderConv3_)["0.variance"]	= Tensor(parser["encoderConv3_.ConvBnRelu_1.bn.running_var"]);

	(*encoderConv3_)["1.weight"]	= Tensor(parser["encoderConv3_.ConvBnRelu_2.conv.weight"]);
	(*encoderConv3_)["1.bias"]		= Tensor(parser["encoderConv3_.ConvBnRelu_2.conv.bias"]);
	(*encoderConv3_)["1.gamma"]		= Tensor(parser["encoderConv3_.ConvBnRelu_2.bn.weight"]);
	(*encoderConv3_)["1.beta"]		= Tensor(parser["encoderConv3_.ConvBnRelu_2.bn.bias"]);
	(*encoderConv3_)["1.mean"]		= Tensor(parser["encoderConv3_.ConvBnRelu_2.bn.running_mean"]);
	(*encoderConv3_)["1.variance"]	= Tensor(parser["encoderConv3_.ConvBnRelu_2.bn.running_var"]);

	/////////////////////////////////encoder 4/////////////////////////////////
	(*encoderConv4_)["0.weight"]	= Tensor(parser["encoderConv4_.ConvBnRelu_1.conv.weight"]);
	(*encoderConv4_)["0.bias"]		= Tensor(parser["encoderConv4_.ConvBnRelu_1.conv.bias"]);
	(*encoderConv4_)["0.gamma"]		= Tensor(parser["encoderConv4_.ConvBnRelu_1.bn.weight"]);
	(*encoderConv4_)["0.beta"]		= Tensor(parser["encoderConv4_.ConvBnRelu_1.bn.bias"]);
	(*encoderConv4_)["0.mean"]		= Tensor(parser["encoderConv4_.ConvBnRelu_1.bn.running_mean"]);
	(*encoderConv4_)["0.variance"]	= Tensor(parser["encoderConv4_.ConvBnRelu_1.bn.running_var"]);

	(*encoderConv4_)["1.weight"]	= Tensor(parser["encoderConv4_.ConvBnRelu_2.conv.weight"]);
	(*encoderConv4_)["1.bias"]		= Tensor(parser["encoderConv4_.ConvBnRelu_2.conv.bias"]);
	(*encoderConv4_)["1.gamma"]		= Tensor(parser["encoderConv4_.ConvBnRelu_2.bn.weight"]);
	(*encoderConv4_)["1.beta"]		= Tensor(parser["encoderConv4_.ConvBnRelu_2.bn.bias"]);
	(*encoderConv4_)["1.mean"]		= Tensor(parser["encoderConv4_.ConvBnRelu_2.bn.running_mean"]);
	(*encoderConv4_)["1.variance"]	= Tensor(parser["encoderConv4_.ConvBnRelu_2.bn.running_var"]);

	/////////////////////////////////bottleneck/////////////////////////////////
	(*bottleneck_)["0.weight"]		= Tensor(parser["bottleneck_.ConvBnRelu_1.conv.weight"]);
	(*bottleneck_)["0.bias"]		= Tensor(parser["bottleneck_.ConvBnRelu_1.conv.bias"]);
	(*bottleneck_)["0.gamma"]		= Tensor(parser["bottleneck_.ConvBnRelu_1.bn.weight"]);
	(*bottleneck_)["0.beta"]		= Tensor(parser["bottleneck_.ConvBnRelu_1.bn.bias"]);
	(*bottleneck_)["0.mean"]		= Tensor(parser["bottleneck_.ConvBnRelu_1.bn.running_mean"]);
	(*bottleneck_)["0.variance"]	= Tensor(parser["bottleneck_.ConvBnRelu_1.bn.running_var"]);

	(*bottleneck_)["1.weight"]		= Tensor(parser["bottleneck_.ConvBnRelu_2.conv.weight"]);
	(*bottleneck_)["1.bias"]		= Tensor(parser["bottleneck_.ConvBnRelu_2.conv.bias"]);
	(*bottleneck_)["1.gamma"]		= Tensor(parser["bottleneck_.ConvBnRelu_2.bn.weight"]);
	(*bottleneck_)["1.beta"]		= Tensor(parser["bottleneck_.ConvBnRelu_2.bn.bias"]);
	(*bottleneck_)["1.mean"]		= Tensor(parser["bottleneck_.ConvBnRelu_2.bn.running_mean"]);
	(*bottleneck_)["1.variance"]	= Tensor(parser["bottleneck_.ConvBnRelu_2.bn.running_var"]);

	/////////////////////////////////Decoder 4/////////////////////////////////
	(*upConv4_)["weight"]			= Tensor(parser["upConv4_.weight"]);
	(*upConv4_)["bias"]				= Tensor(parser["upConv4_.bias"]);

	(*decoderConv4_)["0.weight"]	= Tensor(parser["decoderConv4_.ConvBnRelu_1.conv.weight"]);
	(*decoderConv4_)["0.bias"]		= Tensor(parser["decoderConv4_.ConvBnRelu_1.conv.bias"]);
	(*decoderConv4_)["0.gamma"]		= Tensor(parser["decoderConv4_.ConvBnRelu_1.bn.weight"]);
	(*decoderConv4_)["0.beta"]		= Tensor(parser["decoderConv4_.ConvBnRelu_1.bn.bias"]);
	(*decoderConv4_)["0.mean"]		= Tensor(parser["decoderConv4_.ConvBnRelu_1.bn.running_mean"]);
	(*decoderConv4_)["0.variance"]	= Tensor(parser["decoderConv4_.ConvBnRelu_1.bn.running_var"]);

	(*decoderConv4_)["1.weight"]	= Tensor(parser["decoderConv4_.ConvBnRelu_2.conv.weight"]);
	(*decoderConv4_)["1.bias"]		= Tensor(parser["decoderConv4_.ConvBnRelu_2.conv.bias"]);
	(*decoderConv4_)["1.gamma"]		= Tensor(parser["decoderConv4_.ConvBnRelu_2.bn.weight"]);
	(*decoderConv4_)["1.beta"]		= Tensor(parser["decoderConv4_.ConvBnRelu_2.bn.bias"]);
	(*decoderConv4_)["1.mean"]		= Tensor(parser["decoderConv4_.ConvBnRelu_2.bn.running_mean"]);
	(*decoderConv4_)["1.variance"]	= Tensor(parser["decoderConv4_.ConvBnRelu_2.bn.running_var"]);

	/////////////////////////////////Decoder 3/////////////////////////////////
	(*upConv3_)["weight"]			= Tensor(parser["upConv3_.weight"]);
	(*upConv3_)["bias"]				= Tensor(parser["upConv3_.bias"]);

	(*decoderConv3_)["0.weight"]	= Tensor(parser["decoderConv3_.ConvBnRelu_1.conv.weight"]);
	(*decoderConv3_)["0.bias"]		= Tensor(parser["decoderConv3_.ConvBnRelu_1.conv.bias"]);
	(*decoderConv3_)["0.gamma"]		= Tensor(parser["decoderConv3_.ConvBnRelu_1.bn.weight"]);
	(*decoderConv3_)["0.beta"]		= Tensor(parser["decoderConv3_.ConvBnRelu_1.bn.bias"]);
	(*decoderConv3_)["0.mean"]		= Tensor(parser["decoderConv3_.ConvBnRelu_1.bn.running_mean"]);
	(*decoderConv3_)["0.variance"]	= Tensor(parser["decoderConv3_.ConvBnRelu_1.bn.running_var"]);

	(*decoderConv3_)["1.weight"]	= Tensor(parser["decoderConv3_.ConvBnRelu_2.conv.weight"]);
	(*decoderConv3_)["1.bias"]		= Tensor(parser["decoderConv3_.ConvBnRelu_2.conv.bias"]);
	(*decoderConv3_)["1.gamma"]		= Tensor(parser["decoderConv3_.ConvBnRelu_2.bn.weight"]);
	(*decoderConv3_)["1.beta"]		= Tensor(parser["decoderConv3_.ConvBnRelu_2.bn.bias"]);
	(*decoderConv3_)["1.mean"]		= Tensor(parser["decoderConv3_.ConvBnRelu_2.bn.running_mean"]);
	(*decoderConv3_)["1.variance"]	= Tensor(parser["decoderConv3_.ConvBnRelu_2.bn.running_var"]);

	/////////////////////////////////Decoder 2/////////////////////////////////
	(*upConv2_)["weight"]			= Tensor(parser["upConv2_.weight"]);
	(*upConv2_)["bias"]				= Tensor(parser["upConv2_.bias"]);

	(*decoderConv2_)["0.weight"]	= Tensor(parser["decoderConv2_.ConvBnRelu_1.conv.weight"]);
	(*decoderConv2_)["0.bias"]		= Tensor(parser["decoderConv2_.ConvBnRelu_1.conv.bias"]);
	(*decoderConv2_)["0.gamma"]		= Tensor(parser["decoderConv2_.ConvBnRelu_1.bn.weight"]);
	(*decoderConv2_)["0.beta"]		= Tensor(parser["decoderConv2_.ConvBnRelu_1.bn.bias"]);
	(*decoderConv2_)["0.mean"]		= Tensor(parser["decoderConv2_.ConvBnRelu_1.bn.running_mean"]);
	(*decoderConv2_)["0.variance"]	= Tensor(parser["decoderConv2_.ConvBnRelu_1.bn.running_var"]);

	(*decoderConv2_)["1.weight"]	= Tensor(parser["decoderConv2_.ConvBnRelu_2.conv.weight"]);
	(*decoderConv2_)["1.bias"]		= Tensor(parser["decoderConv2_.ConvBnRelu_2.conv.bias"]);
	(*decoderConv2_)["1.gamma"]		= Tensor(parser["decoderConv2_.ConvBnRelu_2.bn.weight"]);
	(*decoderConv2_)["1.beta"]		= Tensor(parser["decoderConv2_.ConvBnRelu_2.bn.bias"]);
	(*decoderConv2_)["1.mean"]		= Tensor(parser["decoderConv2_.ConvBnRelu_2.bn.running_mean"]);
	(*decoderConv2_)["1.variance"]	= Tensor(parser["decoderConv2_.ConvBnRelu_2.bn.running_var"]);

	/////////////////////////////////Decoder 1/////////////////////////////////
	(*upConv1_)["weight"]			= Tensor(parser["upConv1_.weight"]);
	(*upConv1_)["bias"]				= Tensor(parser["upConv1_.bias"]);

	(*decoderConv1_)["0.weight"]	= Tensor(parser["decoderConv1_.ConvBnRelu_1.conv.weight"]);
	(*decoderConv1_)["0.bias"]		= Tensor(parser["decoderConv1_.ConvBnRelu_1.conv.bias"]);
	(*decoderConv1_)["0.gamma"]		= Tensor(parser["decoderConv1_.ConvBnRelu_1.bn.weight"]);
	(*decoderConv1_)["0.beta"]		= Tensor(parser["decoderConv1_.ConvBnRelu_1.bn.bias"]);
	(*decoderConv1_)["0.mean"]		= Tensor(parser["decoderConv1_.ConvBnRelu_1.bn.running_mean"]);
	(*decoderConv1_)["0.variance"]	= Tensor(parser["decoderConv1_.ConvBnRelu_1.bn.running_var"]);

	(*decoderConv1_)["1.weight"]	= Tensor(parser["decoderConv1_.ConvBnRelu_2.conv.weight"]);
	(*decoderConv1_)["1.bias"]		= Tensor(parser["decoderConv1_.ConvBnRelu_2.conv.bias"]);
	(*decoderConv1_)["1.gamma"]		= Tensor(parser["decoderConv1_.ConvBnRelu_2.bn.weight"]);
	(*decoderConv1_)["1.beta"]		= Tensor(parser["decoderConv1_.ConvBnRelu_2.bn.bias"]);
	(*decoderConv1_)["1.mean"]		= Tensor(parser["decoderConv1_.ConvBnRelu_2.bn.running_mean"]);
	(*decoderConv1_)["1.variance"]	= Tensor(parser["decoderConv1_.ConvBnRelu_2.bn.running_var"]);

	/////////////////////////////////Header/////////////////////////////////
	(*header_)["weight"]			= Tensor(parser["header_.weight"]);
	(*header_)["bias"]				= Tensor(parser["header_.bias"]); 

	return;
}

TestNet::TestNet(Device& device, utils::ImageInfo info, uint32_t numInputs = 1, uint32_t numOutputs = 1)
	: NeuralNet(device, numInputs, numOutputs)
	, device_(device), height_(info.height), width_(info.width), channel_(info.channels), numInputs_(numInputs), numOutputs_(numOutputs)
{
	convTrans_	= std::make_unique<ConvTransposeNode>(3, 8, 2);
	batchnorm_	= std::make_unique<BatchNormNode>();
	sigmoid_	= std::make_unique<SigmoidNode>();
	concat_		= std::make_unique<ConcatNode>();

	input(0) - *convTrans_ - output(0);

	cout << "TestNet Init Done" << endl;
}

void TestNet::setWeight() 
{
	auto w = Tensor::fromFile("D:\\VAI\\vai-sample\\100-UNet-seungwoo\\python\\workspace\\bin\\convTrans_weight.bin");
	auto shape = w.shape(); // (in_channels, out_channels, kernel_height, kernel_width)

	w.permute(0, 2, 3, 1);

	(*convTrans_)["weight"]		= w.reshape(shape[0] * shape[2] * shape[3], shape[1]);
	(*convTrans_)["bias"]		= Tensor::fromFile("D:\\VAI\\vai-sample\\100-UNet-seungwoo\\python\\workspace\\bin\\convTrans_bias.bin");

	(*batchnorm_)["gamma"]		= Tensor::fromFile("D:\\VAI\\vai-sample\\100-UNet-seungwoo\\python\\workspace\\bin\\batchNorm_gamma.bin");
	(*batchnorm_)["beta"]		= Tensor::fromFile("D:\\VAI\\vai-sample\\100-UNet-seungwoo\\python\\workspace\\bin\\batchNorm_beta.bin");
	(*batchnorm_)["mean"]		= Tensor::fromFile("D:\\VAI\\vai-sample\\100-UNet-seungwoo\\python\\workspace\\bin\\batchNorm_running_mean.bin");
	(*batchnorm_)["variance"]	= Tensor::fromFile("D:\\VAI\\vai-sample\\100-UNet-seungwoo\\python\\workspace\\bin\\batchNorm_running_var.bin");

	return;
}
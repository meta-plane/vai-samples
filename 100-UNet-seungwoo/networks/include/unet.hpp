#pragma once

#include <iostream>
#include <memory>

#include "../../library/neuralNet.h"
#include "../../library/neuralNodes.h"
#include "../../library/vulkanApp.h"
#include  "../../safeTensor/safeTensorsParser.h"

#include "layers.hpp"
#include "../../utils.hpp"

namespace networks 
{
	class Unet : public NeuralNet
	{
	private:
		Device					device_;
		uint32_t				numInputs_;
		uint32_t				numOutputs_; 

		uint32_t				height_;
		uint32_t				width_;
		uint32_t				channel_;

		std::unique_ptr<doubleConvBnRelu>		encoderConv1_;
		std::unique_ptr<MaxPoolingNode>			pool1_;

		std::unique_ptr<doubleConvBnRelu>		encoderConv2_;
		std::unique_ptr<MaxPoolingNode>			pool2_;

		std::unique_ptr<doubleConvBnRelu>		encoderConv3_;
		std::unique_ptr<MaxPoolingNode>			pool3_;

		std::unique_ptr<doubleConvBnRelu>		encoderConv4_;
		std::unique_ptr<MaxPoolingNode>			pool4_;

		std::unique_ptr<doubleConvBnRelu>		bottleneck_;

		std::unique_ptr<ConvTransposeNode>		upConv4_;
		std::unique_ptr<ConcatNode>				concat4_;
		std::unique_ptr<doubleConvBnRelu>		decoderConv4_;

		std::unique_ptr<ConvTransposeNode>		upConv3_;
		std::unique_ptr<ConcatNode>				concat3_;
		std::unique_ptr<doubleConvBnRelu>		decoderConv3_;

		std::unique_ptr<ConvTransposeNode>		upConv2_;
		std::unique_ptr<ConcatNode>				concat2_;
		std::unique_ptr<doubleConvBnRelu>		decoderConv2_;

		std::unique_ptr<ConvTransposeNode>		upConv1_;
		std::unique_ptr<ConcatNode>				concat1_;
		std::unique_ptr<doubleConvBnRelu>		decoderConv1_;

		std::unique_ptr<ConvolutionNode>		header_;
		std::unique_ptr<SigmoidNode>			sigmoid_;

	public:
		using s_ptr = std::shared_ptr<Unet>;
		using u_ptr = std::unique_ptr<Unet>;
		using w_ptr = std::weak_ptr<Unet>;

		Unet(Device& device, uint32_t  height, uint32_t width, uint32_t channel, uint32_t numInputs = 1, uint32_t numOutputs = 1);
		~Unet() = default;

		void setWeight(SafeTensorsParser& parser);
		void saveLayerOutputs(const std::string& output_dir = ".");
	};

	class TestNet : public NeuralNet
	{

		Device					device_;
		uint32_t				numInputs_;
		uint32_t				numOutputs_;

		uint32_t				height_;
		uint32_t				width_;
		uint32_t				channel_;

		std::unique_ptr<ConvTransposeNode>		convTrans_;
		std::unique_ptr<BatchNormNode>			batchnorm_;
		std::unique_ptr<SigmoidNode>			sigmoid_;
		std::unique_ptr<ConcatNode>				concat_;
	
	public:
		using s_ptr = std::shared_ptr<TestNet>;
		using u_ptr = std::unique_ptr<TestNet>;
		using w_ptr = std::weak_ptr<TestNet>;

		TestNet(Device& device, utils::ImageInfo info, uint32_t numInputs, uint32_t numOutputs);
		~TestNet() = default;

		void setWeight();
	};
}

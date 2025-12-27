#include <chrono>
#include <iostream>
#include "../library/neuralNodes.h"
#include "../library/vulkanApp.h"



// Helper function to run a single benchmark test case
template<typename NodeType>
void runBenchmark(
    const char* nodeName,
    NodeType& node,
    uint32_t inputH,
    uint32_t inputW,
    uint32_t channels,
    uint32_t warmupIter,
    uint32_t benchmarkIter)
{
    auto device = VulkanApp::get().device();

    // Create a neural network with the node
    NeuralNet net(device);
    net.input(0) - node - net.output(0);

    // Create input tensor [H, W, C]
    Tensor input(inputH, inputW, channels);
    std::vector<float> dummyData(inputH * inputW * channels, 0.5f);
    input.set(dummyData);

    // Warmup phase
    Tensor output;
    for (uint32_t i = 0; i < warmupIter; ++i)
    {
        auto outputs = net(input);
        if (outputs.empty())
        {
            printf("    Error: Network returned empty output vector\n");
            return;
        }
        output = outputs[0];
    }

    // Benchmark phase
    auto startTime = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < benchmarkIter; ++i)
    {
        auto outputs = net(input);
        if (!outputs.empty())
            output = outputs[0];
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    // Print results
    float avgTime = static_cast<float>(duration.count()) / benchmarkIter;
    printf("    Total time: %lld ms, Avg: %.3f ms per iter\n", duration.count(), avgTime);

    const auto& shape = output.shape();
    if (shape.size() == 3)
    {
        printf("    Output shape: [%u, %u, %u]\n", shape[0], shape[1], shape[2]);
    }
    else if (shape.size() > 0)
    {
        printf("    Output shape: [");
        for (size_t i = 0; i < shape.size(); ++i)
        {
            printf("%u%s", shape[i], (i < shape.size() - 1) ? ", " : "");
        }
        printf("]\n");
    }
    else
    {
        printf("    Output shape: empty\n");
    }
}

void benchmarkDepthwiseConv()
{
    auto device = VulkanApp::get().device();
    printf("\n[Benchmark: DepthwiseConv]\n");
    printf("================================================\n");

    const uint32_t warmupIter = 3;
    const uint32_t benchmarkIter = 100;

    // Define test cases
    struct TestCase {
        uint32_t inputH, inputW, channels;
        uint32_t kernelSize, stride, padding;
        const char* description;
    };

    std::vector<TestCase> testCases = {
        // Add or remove test cases here
        {224, 224, 32, 3, 1, 1, "Large input (224x224x32), K=3, S=1"},
        {112, 112, 64, 3, 1, 1, "Medium input (112x112x64), K=3, S=1"},
        {56, 56, 128, 3, 1, 1, "Small input (56x56x128), K=3, S=1"},
        {56, 56, 32, 3, 2, 1, "Stride=2 (56x56x32), K=3, S=2"},
        {56, 56, 32, 5, 1, 2, "Kernel=5 (56x56x32), K=5, S=1"},
    };

    for (const auto& tc : testCases)
    {
        printf("\n  Test: %s\n", tc.description);
        printf("  Config: Input=[%u, %u, %u], Kernel=%u, Stride=%u, Padding=%u\n",
               tc.inputH, tc.inputW, tc.channels, tc.kernelSize, tc.stride, tc.padding);

        // Create node (channels, kernelWidth, stride) - no padding parameter
        DepthwiseConvNode node(tc.channels, tc.kernelSize, tc.stride);

        // Set weights [K*K, C]
        Tensor weights(tc.kernelSize * tc.kernelSize, tc.channels);
        std::vector<float> weightData(tc.kernelSize * tc.kernelSize * tc.channels, 0.01f);
        weights.set(weightData);
        weights.setConstant(true);
        node["weight"] = std::move(weights);

        // Run benchmark
        runBenchmark("DepthwiseConv", node, tc.inputH, tc.inputW, tc.channels,
                     warmupIter, benchmarkIter);
    }

    printf("\n================================================\n");
}

void benchmarkPointwiseConv()
{
    auto device = VulkanApp::get().device();
    printf("\n[Benchmark: PointwiseConv]\n");
    printf("================================================\n");

    const uint32_t warmupIter = 3;
    const uint32_t benchmarkIter = 100;

    // Define test cases
    struct TestCase {
        uint32_t inputH, inputW;
        uint32_t inputChannels, outputChannels;
        const char* description;
    };

    std::vector<TestCase> testCases = {
        // Add or remove test cases here
        {224, 224, 32, 64, "Large input (224x224), 32->64 channels"},
        {112, 112, 64, 128, "Medium input (112x112), 64->128 channels"},
        {56, 56, 128, 256, "Small input (56x56), 128->256 channels"},
        {56, 56, 32, 192, "Expansion (56x56), 32->192 (6x)"},
        {28, 28, 192, 32, "Projection (28x28), 192->32 (1/6x)"},
    };

    for (const auto& tc : testCases)
    {
        printf("\n  Test: %s\n", tc.description);
        printf("  Config: Input=[%u, %u, %u], Output channels=%u\n",
               tc.inputH, tc.inputW, tc.inputChannels, tc.outputChannels);

        // Create node (inChannels, outChannels)
        PointwiseConvNode node(tc.inputChannels, tc.outputChannels);

        // Set weights [C_in, C_out]
        Tensor weights(tc.inputChannels, tc.outputChannels);
        std::vector<float> weightData(tc.inputChannels * tc.outputChannels, 0.01f);
        weights.set(weightData);
        weights.setConstant(true);
        node["weight"] = std::move(weights);

        // Run benchmark
        runBenchmark("PointwiseConv", node, tc.inputH, tc.inputW, tc.inputChannels,
                     warmupIter, benchmarkIter);
    }

    printf("\n================================================\n");
}

void benchmarkConvBnReLU6()
{
    auto device = VulkanApp::get().device();
    printf("\n[Benchmark: ConvBnReLU6]\n");
    printf("================================================\n");

    const uint32_t warmupIter = 3;
    const uint32_t benchmarkIter = 100;

    // Define test cases
    struct TestCase {
        uint32_t inputH, inputW;
        uint32_t inputChannels, outputChannels;
        uint32_t kernelSize, stride, padding;
        const char* description;
    };

    std::vector<TestCase> testCases = {
        // Add or remove test cases here
        {224, 224, 3, 32, 3, 2, 1, "Stem layer (224x224x3->32), K=3, S=2"},
        {112, 112, 32, 64, 3, 2, 1, "Downsampling (112x112x32->64), K=3, S=2"},
        {56, 56, 64, 128, 3, 1, 1, "Same size (56x56x64->128), K=3, S=1"},
        {224, 224, 3, 32, 5, 2, 2, "Large kernel (224x224x3->32), K=5, S=2"},
    };

    for (const auto& tc : testCases)
    {
        printf("\n  Test: %s\n", tc.description);
        printf("  Config: Input=[%u, %u, %u], Output ch=%u, Kernel=%u, Stride=%u, Padding=%u\n",
               tc.inputH, tc.inputW, tc.inputChannels, tc.outputChannels,
               tc.kernelSize, tc.stride, tc.padding);

        // Create ConvBnReLU6 node (inChannels, outChannels, kernel, stride, padding)
        ConvBNReLU6 node(tc.inputChannels, tc.outputChannels, tc.kernelSize, tc.stride, tc.padding);

        // Set conv weights [K*K*C_in, C_out]
        Tensor convWeights(tc.kernelSize * tc.kernelSize * tc.inputChannels, tc.outputChannels);
        std::vector<float> convWeightData(tc.kernelSize * tc.kernelSize * tc.inputChannels * tc.outputChannels, 0.01f);
        convWeights.set(convWeightData);
        convWeights.setConstant(true);
        node["conv.weight"] = std::move(convWeights);

        // Set BatchNorm parameters
        Tensor bnMean(tc.outputChannels), bnVariance(tc.outputChannels),
               bnGamma(tc.outputChannels), bnBeta(tc.outputChannels);
        std::vector<float> zeros(tc.outputChannels, 0.0f), ones(tc.outputChannels, 1.0f);
        bnMean.set(zeros);
        bnVariance.set(ones);
        bnGamma.set(ones);
        bnBeta.set(zeros);
        bnMean.setConstant(true);
        bnVariance.setConstant(true);
        bnGamma.setConstant(true);
        bnBeta.setConstant(true);
        node["bn.mean"] = std::move(bnMean);
        node["bn.variance"] = std::move(bnVariance);
        node["bn.gamma"] = std::move(bnGamma);
        node["bn.beta"] = std::move(bnBeta);

        // Run benchmark
        runBenchmark("ConvBnReLU6", node, tc.inputH, tc.inputW, tc.inputChannels,
                     warmupIter, benchmarkIter);
    }

    printf("\n================================================\n");
}

void benchmarkInvertedResidualBlock()
{
    auto device = VulkanApp::get().device();
    printf("\n[Benchmark: InvertedResidualBlock]\n");
    printf("================================================\n");

    const uint32_t warmupIter = 3;
    const uint32_t benchmarkIter = 100;

    // Define test cases
    struct TestCase {
        uint32_t inputH, inputW;
        uint32_t inputChannels, outputChannels;
        uint32_t expansionRatio, stride;
        const char* description;
    };

    std::vector<TestCase> testCases = {
        // Add or remove test cases here
        {112, 112, 16, 24, 6, 2, "First IRB (112x112), 16->24, t=6, stride=2"},
        {56, 56, 24, 32, 6, 2, "Downsampling (56x56), 24->32, t=6, stride=2"},
        {56, 56, 32, 32, 6, 1, "Same size (56x56), 32->32, t=6, stride=1"},
        {28, 28, 64, 96, 6, 1, "Mid layer (28x28), 64->96, t=6, stride=1"},
        {14, 14, 160, 320, 6, 1, "Late layer (14x14), 160->320, t=6, stride=1"},
    };

    for (const auto& tc : testCases)
    {
        const uint32_t expandedChannels = tc.inputChannels * tc.expansionRatio;

        printf("\n  Test: %s\n", tc.description);
        printf("  Config: Input=[%u, %u, %u], Output ch=%u, Expansion=%u (->%u), Stride=%u\n",
               tc.inputH, tc.inputW, tc.inputChannels, tc.outputChannels,
               tc.expansionRatio, expandedChannels, tc.stride);

        // Create IRB (inChannels, outChannels, expansionFactor, stride)
        InvertedResidualBlock node(tc.inputChannels, tc.outputChannels, tc.expansionRatio, tc.stride);

        // Set expand conv weights (1x1 conv: inputChannels -> expandedChannels)
        Tensor expandWeights(tc.inputChannels, expandedChannels);
        std::vector<float> expandWeightData(tc.inputChannels * expandedChannels, 0.01f);
        expandWeights.set(expandWeightData);
        expandWeights.setConstant(true);
        node["pwConvBNReLU6.pointwiseConv.weight"] = std::move(expandWeights);

        // Set expand BN parameters
        Tensor expandBnMean(expandedChannels), expandBnVariance(expandedChannels),
               expandBnGamma(expandedChannels), expandBnBeta(expandedChannels);
        std::vector<float> zerosExpand(expandedChannels, 0.0f), onesExpand(expandedChannels, 1.0f);
        expandBnMean.set(zerosExpand);
        expandBnVariance.set(onesExpand);
        expandBnGamma.set(onesExpand);
        expandBnBeta.set(zerosExpand);
        expandBnMean.setConstant(true);
        expandBnVariance.setConstant(true);
        expandBnGamma.setConstant(true);
        expandBnBeta.setConstant(true);
        node["pwConvBNReLU6.bn.mean"] = std::move(expandBnMean);
        node["pwConvBNReLU6.bn.variance"] = std::move(expandBnVariance);
        node["pwConvBNReLU6.bn.gamma"] = std::move(expandBnGamma);
        node["pwConvBNReLU6.bn.beta"] = std::move(expandBnBeta);

        // Set depthwise conv weights (3x3 depthwise)
        Tensor dwWeights(3 * 3, expandedChannels);
        std::vector<float> dwWeightData(3 * 3 * expandedChannels, 0.01f);
        dwWeights.set(dwWeightData);
        dwWeights.setConstant(true);
        node["dwConvBNReLU6.depthwiseConv.weight"] = std::move(dwWeights);

        // Set depthwise BN parameters
        Tensor dwBnMean(expandedChannels), dwBnVariance(expandedChannels),
               dwBnGamma(expandedChannels), dwBnBeta(expandedChannels);
        dwBnMean.set(zerosExpand);
        dwBnVariance.set(onesExpand);
        dwBnGamma.set(onesExpand);
        dwBnBeta.set(zerosExpand);
        dwBnMean.setConstant(true);
        dwBnVariance.setConstant(true);
        dwBnGamma.setConstant(true);
        dwBnBeta.setConstant(true);
        node["dwConvBNReLU6.bn.mean"] = std::move(dwBnMean);
        node["dwConvBNReLU6.bn.variance"] = std::move(dwBnVariance);
        node["dwConvBNReLU6.bn.gamma"] = std::move(dwBnGamma);
        node["dwConvBNReLU6.bn.beta"] = std::move(dwBnBeta);

        // Set project conv weights (1x1 conv: expandedChannels -> outputChannels)
        Tensor projectWeights(expandedChannels, tc.outputChannels);
        std::vector<float> projectWeightData(expandedChannels * tc.outputChannels, 0.01f);
        projectWeights.set(projectWeightData);
        projectWeights.setConstant(true);
        node["pwConvBN.pointwiseConv.weight"] = std::move(projectWeights);

        // Set project BN parameters
        Tensor projectBnMean(tc.outputChannels), projectBnVariance(tc.outputChannels),
               projectBnGamma(tc.outputChannels), projectBnBeta(tc.outputChannels);
        std::vector<float> zerosOutput(tc.outputChannels, 0.0f), onesOutput(tc.outputChannels, 1.0f);
        projectBnMean.set(zerosOutput);
        projectBnVariance.set(onesOutput);
        projectBnGamma.set(onesOutput);
        projectBnBeta.set(zerosOutput);
        projectBnMean.setConstant(true);
        projectBnVariance.setConstant(true);
        projectBnGamma.setConstant(true);
        projectBnBeta.setConstant(true);
        node["pwConvBN.bn.mean"] = std::move(projectBnMean);
        node["pwConvBN.bn.variance"] = std::move(projectBnVariance);
        node["pwConvBN.bn.gamma"] = std::move(projectBnGamma);
        node["pwConvBN.bn.beta"] = std::move(projectBnBeta);

        // Run benchmark
        runBenchmark("InvertedResidualBlock", node, tc.inputH, tc.inputW, tc.inputChannels,
                     warmupIter, benchmarkIter);
    }

    printf("\n================================================\n");
}

void benchmarkBatchNorm()
{
    auto device = VulkanApp::get().device();
    printf("\n[Benchmark: BatchNorm]\n");
    printf("================================================\n");

    const uint32_t warmupIter = 3;
    const uint32_t benchmarkIter = 100;

    struct TestCase {
        uint32_t inputH, inputW, channels;
        const char* description;
    };

    std::vector<TestCase> testCases = {
        {224, 224, 32, "Large input (224x224x32)"},
        {112, 112, 64, "Medium input (112x112x64)"},
        {56, 56, 128, "Small input (56x56x128)"},
        {28, 28, 256, "Tiny input (28x28x256)"},
    };

    for (const auto& tc : testCases)
    {
        printf("\n  Test: %s\n", tc.description);
        printf("  Config: Input=[%u, %u, %u]\n", tc.inputH, tc.inputW, tc.channels);

        BatchNormNode node;

        // Set BatchNorm parameters
        Tensor bnMean(tc.channels), bnVariance(tc.channels),
               bnGamma(tc.channels), bnBeta(tc.channels);
        std::vector<float> zeros(tc.channels, 0.0f), ones(tc.channels, 1.0f);
        bnMean.set(zeros);
        bnVariance.set(ones);
        bnGamma.set(ones);
        bnBeta.set(zeros);
        bnMean.setConstant(true);
        bnVariance.setConstant(true);
        bnGamma.setConstant(true);
        bnBeta.setConstant(true);
        node["mean"] = std::move(bnMean);
        node["variance"] = std::move(bnVariance);
        node["gamma"] = std::move(bnGamma);
        node["beta"] = std::move(bnBeta);

        runBenchmark("BatchNorm", node, tc.inputH, tc.inputW, tc.channels,
                     warmupIter, benchmarkIter);
    }

    printf("\n================================================\n");
}

void benchmarkRelu6()
{
    auto device = VulkanApp::get().device();
    printf("\n[Benchmark: ReLU6]\n");
    printf("================================================\n");

    const uint32_t warmupIter = 3;
    const uint32_t benchmarkIter = 100;

    struct TestCase {
        uint32_t inputH, inputW, channels;
        const char* description;
    };

    std::vector<TestCase> testCases = {
        {224, 224, 32, "Large input (224x224x32)"},
        {112, 112, 64, "Medium input (112x112x64)"},
        {56, 56, 128, "Small input (56x56x128)"},
    };

    for (const auto& tc : testCases)
    {
        printf("\n  Test: %s\n", tc.description);
        printf("  Config: Input=[%u, %u, %u]\n", tc.inputH, tc.inputW, tc.channels);

        Relu6Node node;

        runBenchmark("ReLU6", node, tc.inputH, tc.inputW, tc.channels,
                     warmupIter, benchmarkIter);
    }

    printf("\n================================================\n");
}

void benchmarkAdd()
{
    auto device = VulkanApp::get().device();
    printf("\n[Benchmark: Add (Residual)]\n");
    printf("================================================\n");

    const uint32_t warmupIter = 3;
    const uint32_t benchmarkIter = 100;

    struct TestCase {
        uint32_t inputH, inputW, channels;
        const char* description;
    };

    std::vector<TestCase> testCases = {
        {224, 224, 32, "Large input (224x224x32)"},
        {112, 112, 64, "Medium input (112x112x64)"},
        {56, 56, 128, "Small input (56x56x128)"},
    };

    for (const auto& tc : testCases)
    {
        printf("\n  Test: %s\n", tc.description);
        printf("  Config: Input=[%u, %u, %u]\n", tc.inputH, tc.inputW, tc.channels);

        AddNode node;

        // Note: AddNode requires two inputs, we'll need to modify runBenchmark for this
        // For now, skip or use a modified approach
        printf("    Note: AddNode requires 2 inputs - skipping automated benchmark\n");
    }

    printf("\n================================================\n");
}

void benchmarkMaxPooling()
{
    auto device = VulkanApp::get().device();
    printf("\n[Benchmark: MaxPooling]\n");
    printf("================================================\n");

    const uint32_t warmupIter = 3;
    const uint32_t benchmarkIter = 100;

    struct TestCase {
        uint32_t inputH, inputW, channels;
        uint32_t poolSize;
        const char* description;
    };

    std::vector<TestCase> testCases = {
        {224, 224, 32, 2, "Large input (224x224x32), Pool=2"},
        {112, 112, 64, 2, "Medium input (112x112x64), Pool=2"},
        {56, 56, 128, 2, "Small input (56x56x128), Pool=2"},
        {224, 224, 32, 3, "Large input (224x224x32), Pool=3"},
    };

    for (const auto& tc : testCases)
    {
        printf("\n  Test: %s\n", tc.description);
        printf("  Config: Input=[%u, %u, %u], PoolSize=%u\n",
               tc.inputH, tc.inputW, tc.channels, tc.poolSize);

        MaxPoolingNode node(tc.poolSize);

        runBenchmark("MaxPooling", node, tc.inputH, tc.inputW, tc.channels,
                     warmupIter, benchmarkIter);
    }

    printf("\n================================================\n");
}

void benchmarkGlobalAvgPool()
{
    auto device = VulkanApp::get().device();
    printf("\n[Benchmark: GlobalAvgPool]\n");
    printf("================================================\n");

    const uint32_t warmupIter = 3;
    const uint32_t benchmarkIter = 100;

    struct TestCase {
        uint32_t inputH, inputW, channels;
        const char* description;
    };

    std::vector<TestCase> testCases = {
        {7, 7, 1280, "Final feature map (7x7x1280)"},
        {14, 14, 320, "Mid feature map (14x14x320)"},
        {28, 28, 160, "Early feature map (28x28x160)"},
    };

    for (const auto& tc : testCases)
    {
        printf("\n  Test: %s\n", tc.description);
        printf("  Config: Input=[%u, %u, %u]\n", tc.inputH, tc.inputW, tc.channels);

        GlobalAvgPoolNode node;

        runBenchmark("GlobalAvgPool", node, tc.inputH, tc.inputW, tc.channels,
                     warmupIter, benchmarkIter);
    }

    printf("\n================================================\n");
}

void benchmarkFullyConnected()
{
    auto device = VulkanApp::get().device();
    printf("\n[Benchmark: FullyConnected]\n");
    printf("================================================\n");

    const uint32_t warmupIter = 3;
    const uint32_t benchmarkIter = 100;

    struct TestCase {
        uint32_t inputDim, outputDim;
        const char* description;
    };

    std::vector<TestCase> testCases = {
        {1280, 1000, "Classifier (1280->1000)"},
        {1280, 100, "Small classifier (1280->100)"},
        {512, 256, "Hidden layer (512->256)"},
        {2048, 1000, "Large classifier (2048->1000)"},
    };

    for (const auto& tc : testCases)
    {
        printf("\n  Test: %s\n", tc.description);
        printf("  Config: Input=%u, Output=%u\n", tc.inputDim, tc.outputDim);

        FullyConnectedNode node(tc.inputDim, tc.outputDim);

        // Set weights [inputDim, outputDim]
        Tensor weights(tc.inputDim, tc.outputDim);
        std::vector<float> weightData(tc.inputDim * tc.outputDim, 0.01f);
        weights.set(weightData);
        weights.setConstant(true);
        node["weight"] = std::move(weights);

        // Set bias [outputDim]
        Tensor bias(tc.outputDim);
        std::vector<float> biasData(tc.outputDim, 0.0f);
        bias.set(biasData);
        bias.setConstant(true);
        node["bias"] = std::move(bias);

        // FC layer expects 1D input tensor
        NeuralNet net(device);
        net.input(0) - node - net.output(0);

        // Create 1D input tensor [inputDim]
        Tensor input(tc.inputDim);
        std::vector<float> dummyData(tc.inputDim, 0.5f);
        input.set(dummyData);

        // Warmup phase
        Tensor output;
        for (uint32_t i = 0; i < warmupIter; ++i)
        {
            auto outputs = net(input);
            if (outputs.empty())
            {
                printf("    Error: Network returned empty output vector\n");
                continue;
            }
            output = outputs[0];
        }

        // Benchmark phase
        auto startTime = std::chrono::high_resolution_clock::now();
        for (uint32_t i = 0; i < benchmarkIter; ++i)
        {
            auto outputs = net(input);
            if (!outputs.empty())
                output = outputs[0];
        }
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        // Print results
        float avgTime = static_cast<float>(duration.count()) / benchmarkIter;
        printf("    Total time: %lld ms, Avg: %.3f ms per iter\n", duration.count(), avgTime);

        const auto& shape = output.shape();
        if (shape.size() > 0)
        {
            printf("    Output shape: [");
            for (size_t i = 0; i < shape.size(); ++i)
            {
                printf("%u%s", shape[i], (i < shape.size() - 1) ? ", " : "");
            }
            printf("]\n");
        }
        else
        {
            printf("    Output shape: empty\n");
        }
    }

    printf("\n================================================\n");
}

void benchmarkConvolution()
{
    auto device = VulkanApp::get().device();
    printf("\n[Benchmark: Convolution]\n");
    printf("================================================\n");

    const uint32_t warmupIter = 3;
    const uint32_t benchmarkIter = 100;

    struct TestCase {
        uint32_t inputH, inputW;
        uint32_t inputChannels, outputChannels;
        uint32_t kernelSize, stride, padding;
        const char* description;
    };

    std::vector<TestCase> testCases = {
        {224, 224, 3, 32, 3, 2, 1, "Stem layer (224x224x3->32), K=3, S=2"},
        {112, 112, 32, 64, 3, 2, 1, "Downsampling (112x112x32->64), K=3, S=2"},
        {56, 56, 64, 128, 3, 1, 1, "Same size (56x56x64->128), K=3, S=1"},
        {224, 224, 3, 64, 7, 2, 3, "Large kernel (224x224x3->64), K=7, S=2"},
    };

    for (const auto& tc : testCases)
    {
        printf("\n  Test: %s\n", tc.description);
        printf("  Config: Input=[%u, %u, %u], Output ch=%u, Kernel=%u, Stride=%u, Padding=%u\n",
               tc.inputH, tc.inputW, tc.inputChannels, tc.outputChannels,
               tc.kernelSize, tc.stride, tc.padding);

        ConvolutionNode node(tc.inputChannels, tc.outputChannels, tc.kernelSize, tc.stride, tc.padding);

        // Set conv weights [K*K*C_in, C_out]
        Tensor weights(tc.kernelSize * tc.kernelSize * tc.inputChannels, tc.outputChannels);
        std::vector<float> weightData(tc.kernelSize * tc.kernelSize * tc.inputChannels * tc.outputChannels, 0.01f);
        weights.set(weightData);
        weights.setConstant(true);
        node["weight"] = std::move(weights);

        runBenchmark("Convolution", node, tc.inputH, tc.inputW, tc.inputChannels,
                     warmupIter, benchmarkIter);
    }

    printf("\n================================================\n");
}

void benchmarkAllLayers()
{
    printf("\n");
    printf("================================================\n");
    printf("  MobileNetV2 Layer-wise Benchmark Suite\n");
    printf("================================================\n");

    // Basic operation nodes
    benchmarkBatchNorm();
    benchmarkRelu6();
    benchmarkAdd();
    benchmarkMaxPooling();
    benchmarkGlobalAvgPool();
    benchmarkFullyConnected();

    // Convolution nodes
    benchmarkConvolution();
    benchmarkDepthwiseConv();
    benchmarkPointwiseConv();

    // Composite nodes
    benchmarkConvBnReLU6();
    benchmarkInvertedResidualBlock();

    printf("\n");
    printf("================================================\n");
    printf("  Benchmark Complete\n");
    printf("================================================\n");
}

#include "./models/mobilenetv2.h"
#include "./core/timeChecker.hpp"
#include "./utils/utils.h"
#include <stb/stb_image.h>
#include <cstring>
#include <cstdio>
#include <chrono>

void test()
{
    printf("============================================================\n");
    printf("    MobileNetV2 Full Classification (17 IRB blocks)\n");
    printf("============================================================\n");

    const char* weightsPath = PROJECT_CURRENT_DIR "/weights/mobilenet_v2_imagenet1k.safetensors";
    const char* imagePath = PROJECT_CURRENT_DIR "/images/shark.png";

    printf("\n[1/4] Loading image...\n");
    int w, h, c;
    uint8_t* imgData = stbi_load(imagePath, &w, &h, &c, 3);
    if (!imgData) {
        printf("Failed to load image: %s\n", imagePath);
        return;
    }
    printf("  Image: %s (%d x %d)\n", imagePath, w, h);

    const int size = 224;
    std::vector<float> input(size * size * 3);
    int cropSize = std::min(w, h);
    int offsetX = (w - cropSize) / 2;
    int offsetY = (h - cropSize) / 2;
    const float mean[3] = { 0.485f, 0.456f, 0.406f };
    const float std_[3] = { 0.229f, 0.224f, 0.225f };

    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            int srcX = offsetX + (x * cropSize) / size;
            int srcY = offsetY + (y * cropSize) / size;
            srcX = std::max(0, std::min(w - 1, srcX));
            srcY = std::max(0, std::min(h - 1, srcY));
            int srcIdx = (srcY * w + srcX) * 3;
            int dstIdx = (y * size + x) * 3;
            for (int ch = 0; ch < 3; ++ch) {
                float pixel = imgData[srcIdx + ch] / 255.0f;
                input[dstIdx + ch] = (pixel - mean[ch]) / std_[ch];
            }
        }
    }

    stbi_image_free(imgData);
    printf("\nPreprocessed to 224x224\n");

    printf("\n[2/4] Building MobileNetV2 model...\n");
    MobileNetV2 model(gDevice);
    model.loadWeights(weightsPath);

    printf("\n[3/4] Running inference...\n");
    Tensor inputTensor(224, 224, 3);
    inputTensor.set(input);

    auto start = std::chrono::high_resolution_clock::now();
    auto results = model(std::move(inputTensor));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("  Inference time: %lld ms\n", (long long)duration.count());

    printf("\n[4/4] Processing results...\n");
    Buffer outBuffer = gDevice.createBuffer({
        1000 * sizeof(float),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    });

    gDevice.newCommandBuffer(queue_compute)
        .begin()
        .copyBuffer(outBuffer, results[0].buffer())
        .end()
        .submit()
        .wait();

    float* logits = (float*)outBuffer.map();
    std::vector<float> probs(logits, logits + 1000);
    softmax(probs.data(), 1000);
    auto top5 = getTopK(probs.data(), 1000, 5);

    printf("\n");
    printf("============================================================\n");
    printf("                    CLASSIFICATION RESULTS\n");
    printf("============================================================\n");
    printf("\n  Image: %s\n\n", imagePath);
    printf("  Top-5 Predictions:\n");
    printf("  ----------------------------------------------------------\n");
    for (int rank = 0; rank < 5; ++rank) {
        int classIdx = top5[rank];
        float confidence = probs[classIdx];
        printf("   #%d: Class %d - Confidence: %.4f%%\n", rank + 1, classIdx, confidence * 100.0f);
    }

    printf("\n============================================================\n");
}

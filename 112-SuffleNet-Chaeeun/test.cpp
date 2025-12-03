#include "neuralNet.h"
#include "neuralNodes.h"
#include "safeTensorsParser.h"
#include "SuffleNet.h"
#include "timeChecker.hpp"
#include "jsonParser.h"
#include <cstring>
#include <vector>
#include <string>
#include <algorithm> // clamp
#include <cmath>     // exp, round
#include <stb/stb_image.h>
#include <cstdlib>
#include <filesystem>

void loadShaders();

// Reuse the same image reader pattern as 11-mnist-refactor
template<uint32_t Channels>
auto readImage(const char* filename)
{
    int w = 0, h = 0, c0 = 0;
    const int c = static_cast<int>(Channels);
    std::vector<uint8_t> srcImage;

    if (uint8_t* input = stbi_load(filename, &w, &h, &c0, c))
    {
        srcImage.assign(input, input + (size_t)w * h * c);
        stbi_image_free(input);
    }
    else
    {
        printf("%s\n", stbi_failure_reason());
        fflush(stdout);
        throw;
    }

    return std::make_tuple(srcImage, (uint32_t)w, (uint32_t)h);
}

static void load_shufflenet_safetensors(SuffleNetV2& net, const char* path)
{
    SafeTensorsParser st(path);
    auto names = st.getTensorNames();
    size_t total = names.size();
    size_t bound = 0, skipped = 0;
    size_t wcnt = 0, bcnt = 0;
    size_t gammacnt = 0, betacnt = 0, rmcnt = 0, rvcnt = 0;
    size_t other = 0;

    for (const std::string& key : names)
    {
        try {
            auto ref = st[key];
            std::vector<uint32_t> shape = ref.getShape();
            std::string dtype = ref.getDataType();

            // Ensure destination exists in network before parsing large tensor payload
            Tensor* dst = nullptr;
            try { dst = &net[key]; }
            catch (const std::exception& e) {
                printf("[weights][skip] unknown param in net: %s (dtype=%s)\n", key.c_str(), dtype.c_str());
                skipped++; continue;
            }
            catch (...) {
                printf("[weights][skip] unknown param in net: %s (dtype=%s, reason=unknown)\n", key.c_str(), dtype.c_str());
                skipped++; continue;
            }

            if (dtype != "F32")
                printf("[weights][warn] unexpected dtype for %s: %s (expect F32)\n", key.c_str(), dtype.c_str());

            std::vector<float> data = ref.parseNDArray();
            Tensor t(shape);
            t.set(std::move(data)).markConstant();
            net[key] = std::move(t);
            bound++;

            auto ends_with = [](const std::string& s, const char* suf) {
                size_t n = std::strlen(suf);
                return s.size() >= n && 0 == s.compare(s.size()-n, n, suf);
            };
            bool isW = ends_with(key, ".weight");
            bool isB = ends_with(key, ".bias");
            bool isG = ends_with(key, ".gamma");
            bool isBe= ends_with(key, ".beta");
            bool isRM= ends_with(key, ".running_mean");
            bool isRV= ends_with(key, ".running_var");
            if (isW) wcnt++; else if (isB) bcnt++; else if (isG) gammacnt++; else if (isBe) betacnt++; else if (isRM) rmcnt++; else if (isRV) rvcnt++; else other++;

            if (isW || isB || isG || isBe || isRM || isRV) {
                printf("[weights][bound] %-40s shape=", key.c_str());
                for (size_t i = 0; i < shape.size(); ++i) {
                    printf("%u%s", shape[i], (i+1<shape.size()?"x":""));
                }
                printf(", dtype=%s\n", dtype.c_str());
            }
        } catch (const std::exception& e) {
            printf("[weights][error] key=%s reason=%s\n", key.c_str(), e.what());
            skipped++;
        } catch (...) {
            printf("[weights][error] key=%s reason=unknown\n", key.c_str());
            skipped++;
        }
    }

    printf("[weights] summary: total=%zu bound=%zu skipped=%zu (weights=%zu, bias=%zu, gamma=%zu, beta=%zu, running_mean=%zu, running_var=%zu, others=%zu)\n",
           total, bound, skipped, wcnt, bcnt, gammacnt, betacnt, rmcnt, rvcnt, other);
}

// Eval function in the style of 11-mnist-refactor: run net for `iter` iterations
static Tensor eval_shufflenet(const Tensor& input, SuffleNetV2& net, uint32_t iter)
{
    Tensor result;
    Tensor in = input; // keep a local copy to avoid moving the original
    for (uint32_t i = 0; i < iter; ++i)
        result = net(in)[0];
    return result;
}

static bool tensorToHost(Tensor& tensor, std::vector<float>& host, std::vector<uint32_t>& shapeOut)
{
    const auto& shape = tensor.shape();
    if (shape.empty()) {
        printf("[debug] tensor has no shape\n");
        return false;
    }
    size_t elemCount = tensor.numElements();
    if (elemCount == 0) {
        printf("[debug] tensor empty\n");
        return false;
    }
    shapeOut.assign(shape.begin(), shape.end());
    host.resize(elemCount);
    if (tensor.hasHostData()) {
        std::memcpy(host.data(), tensor.hostData(), elemCount * sizeof(float));
    } else if (tensor.hasDeviceData()) {
        vk::Buffer dbgBuffer = netGlobalDevice.createBuffer({
            elemCount * sizeof(float),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        });

        netGlobalDevice.newCommandBuffer(queue_compute)
            .begin()
            .copyBuffer(dbgBuffer, tensor.buffer())
            .end()
            .submit()
            .wait();

        std::memcpy(host.data(), dbgBuffer.map(), elemCount * sizeof(float));
    } else {
        printf("[debug] tensor has no data\n");
        return false;
    }
    return true;
}

static void dumpTensorSlice(const char* label, Tensor& tensor)
{
    std::vector<float> host;
    std::vector<uint32_t> shape;
    if (!tensorToHost(tensor, host, shape))
        return;
    size_t show = std::min<size_t>(5, host.size());
    printf("[debug][%s] shape=[", label);
    for (size_t i = 0; i < shape.size(); ++i)
        printf("%u%s", shape[i], (i + 1 < shape.size() ? "," : ""));
    printf("] values=");
    if (show) {
        printf("[");
        for (size_t i = 0; i < show; ++i)
            printf("%s%.6f", (i == 0 ? "" : ", "), host[i]);
        printf("]");
    } else {
        printf("[]");
    }
    printf("\n");
}

static void dumpTensorToJson(const std::string& basePath, Tensor& tensor)
{
    std::vector<float> host;
    std::vector<uint32_t> shape;
    if (!tensorToHost(tensor, host, shape)) {
        printf("[debug] cannot dump tensor %s\n", basePath.c_str());
        return;
    }

    const std::filesystem::path dumpDir = std::filesystem::path(PROJECT_CURRENT_DIR) / "dump";
    std::error_code ec;
    std::filesystem::create_directories(dumpDir, ec);
    const std::filesystem::path jsonPath = dumpDir / (basePath + ".json");
    if (!writeTensorToJson(host, shape, jsonPath.string())) {
        printf("[debug] failed to open %s for writing\n", jsonPath.string().c_str());
        return;
    }
    printf("[debug] dumped tensor to %s\n", jsonPath.string().c_str());
}

void Run()
{
    void loadShaders();
    loadShaders();
    printf("[Run] shaders loaded\n");

    // Resolve weights path once
    const char* candidates[] = {
        PROJECT_CURRENT_DIR"/python/weights_cpp.safetensors",
    };
    const char* weightsPath = nullptr;
    for (const char* path : candidates) {
        printf("[Run] try weights: %s\n", path);
        try {
            SafeTensorsParser st(path);
            (void)st;
            weightsPath = path;
            printf("[Run] loaded weights: %s\n", path);
            break;
        } catch(const std::exception& e) {
            printf("[Run] not found: %s\n", path);
        } catch(...) {
            printf("[Run] not found: %s (unknown)\n", path);
        }
    }
    if (!weightsPath)
        throw std::runtime_error("Failed to load weights.safetensors from known locations.");

    const bool dumpDebug = (std::getenv("SHUFFLENET_DEBUG") != nullptr);

    // Loader using readImage<>; converts to BGR float (0..255), no resize/crop
    auto load_image_hwc = [](const char* path, uint32_t /*outH*/, uint32_t /*outW*/) -> Tensor
    {
        constexpr uint32_t reqC = 3;
        auto [srcImage, width, height] = readImage<reqC>(path);
        printf("[Run] source image size: %ux%u\n", width, height);
        // Convert RGB uint8 -> BGR float (0..255)
        std::vector<float> bgr((size_t)width * height * reqC);
        for (size_t i = 0; i < (size_t)width * height; ++i)
        {
            const uint8_t r = srcImage[i * reqC + 0];
            const uint8_t g = srcImage[i * reqC + 1];
            const uint8_t b = srcImage[i * reqC + 2];
            bgr[i * reqC + 0] = (float)b;
            bgr[i * reqC + 1] = (float)g;
            bgr[i * reqC + 2] = (float)r;
        }
        return Tensor(height, width, reqC).set(std::move(bgr));
    };

    const uint32_t H = 224, W = 224, C = 3;
    const char* imgCandidates[] = {
        PROJECT_CURRENT_DIR"/data/cat_281.jpg",
        PROJECT_CURRENT_DIR"/data/dog_207.jpg",
        PROJECT_CURRENT_DIR"/data/dog2_207.jpg",
        PROJECT_CURRENT_DIR"/data/tiger_292.jpg",
        PROJECT_CURRENT_DIR"/data/trafficlight_920.jpg",
        PROJECT_CURRENT_DIR"/data/zebra_340.jpg",
    };
    // Inference for each image, print argmax class
    // Build net once and load weights once to avoid descriptor pool exhaustion and ensure stability
    std::vector<int> architecture = {0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2};
    SuffleNetV2 net(netGlobalDevice, architecture, 1000);
    {
        TimeChecker timer("(VAI) Load ShuffleNet weights");
        load_shufflenet_safetensors(net, weightsPath);
    }
    printf("[Run] weights bound to network\n");
    if (dumpDebug)
    {
        dumpTensorToJson("stem_conv_weight_cpp", net["stem.conv.weight"]);
        dumpTensorToJson("stem_bn_gamma_cpp", net["stem.bn.gamma"]);
        dumpTensorToJson("stem_bn_beta_cpp", net["stem.bn.beta"]);
    }

    int attempted = 0, processed = 0;
    const uint32_t iter = 1; // align with 11-mnist style (configurable if needed)
    for (const char* p : imgCandidates)
    {
        attempted++;
        const char* base = p; for (const char* q = p; *q; ++q) if (*q=='/' || *q=='\\') base = q+1;
        Tensor input;
        try { input = load_image_hwc(p, H, W); printf("[Run] image loaded: %s\n", base); }
        catch (const std::exception& e) { printf("[Run] skip image %s: %s\n", base, e.what()); continue; }
        catch (...) { printf("[Run] skip image %s: unknown error\n", base); continue; }
        if (dumpDebug)
            dumpTensorSlice("input", input);

        std::vector<std::pair<std::string, Tensor*>> debugSlots;
        if (dumpDebug)
        {
            debugSlots = {
                {"stem.conv", &net.debug_first_conv_out()},
                {"stem.bn", &net.debug_first_bn_out()},
                {"first_hs", &net.debug_first_hs()},
            };
            for (uint32_t i = 0; i < net.debug_feature_count(); ++i) {
                debugSlots.push_back({"feat." + std::to_string(i), &net.debug_feature_out(i)});
            }
            debugSlots.push_back({"conv_last", &net.debug_conv_last_out()});
            debugSlots.push_back({"bn_last", &net.debug_bn_last_out()});
            debugSlots.push_back({"last_hs", &net.debug_last_hs_out()});
            gNeuralNetKeepTensors = true;
        }

        Tensor logits;
        {
            TimeChecker timer("(VAI) ShuffleNet evaluation: {} iterations", iter);
            logits = eval_shufflenet(input, net, iter);
        }
        if (dumpDebug)
            gNeuralNetKeepTensors = false;

        if (dumpDebug)
        {
            std::string baseNameStr(base);
            size_t dot = baseNameStr.find_last_of('.');
            if (dot != std::string::npos)
                baseNameStr = baseNameStr.substr(0, dot);

            for (auto& slot : debugSlots)
            {
                dumpTensorSlice(slot.first.c_str(), *slot.second);
                const std::string& label = slot.first;
                if (label == "stem.conv")
                    dumpTensorToJson(baseNameStr + "_stem_conv_cpp", *slot.second);
                else if (label == "stem.bn")
                    dumpTensorToJson(baseNameStr + "_stem_bn_cpp", *slot.second);
                slot.second->markConstant(false);
                *slot.second = Tensor();
            }
        }

        // Read back all logits and find argmax
        const auto& s = logits.shape();
        uint32_t outN = s.empty() ? 1000u : s[0];
        vk::Buffer outBuffer = netGlobalDevice.createBuffer({
            outN * sizeof(float),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        });

        netGlobalDevice.newCommandBuffer(queue_compute)
            .begin()
            .copyBuffer(outBuffer, logits.buffer())
            .end()
            .submit()
            .wait();

        std::vector<float> host(outN);
        std::memcpy(host.data(), outBuffer.map(), outN * sizeof(float));

        // softmax for probability
        float maxLogit = host[0];
        for (uint32_t i = 1; i < outN; ++i) if (host[i] > maxLogit) maxLogit = host[i];
        double sumExp = 0.0;
        for (uint32_t i = 0; i < outN; ++i) { host[i] = std::exp(host[i] - maxLogit); sumExp += host[i]; }
        uint32_t bestIdx = 0; float bestProb = float(host[0] / sumExp);
        for (uint32_t i = 1; i < outN; ++i) {
            float p = float(host[i] / sumExp);
            if (p > bestProb) { bestProb = p; bestIdx = i; }
        }

        printf("%s -> %u (prob=%.4f)\n", base, bestIdx, bestProb);
        processed++;
    }
    printf("[Run] images tried=%d, processed=%d\n", attempted, processed);
}

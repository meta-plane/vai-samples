import torch
import torchvision.transforms as T
import torchvision.models as models

from PIL import Image


def hook_fn(module, input, output):    
    # convert NCHW to HWC and show first 10 elements
    if output.dim() == 4:  # Conv layer output
        output_hwc = output.squeeze(0).permute(1, 2, 0)
    else:
        output_hwc = output.squeeze(0)

    print(f"[HOOK] Layer: {module.__class__.__name__}, Output shape: {output_hwc.shape}")
    print("First 10 elements of output tensor:")
    print(output_hwc.flatten()[:10].tolist())
    print("-" * 50)


debug = True
img_path = "../img/shark.png"

# 1) 모델 로드
weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
model = models.mobilenet_v2(weights=weights)
model.eval()

# Forward Hook 등록
if debug:
    for name, module in model.named_modules():
        module.register_forward_hook(hook_fn)

# 2) Resize 224×224 + Normalize(ImageNet)
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# 3) 이미지 로드 & 전처리
img = Image.open(img_path).convert("RGB")
input_tensor = preprocess(img).unsqueeze(0)  # (1,3,224,224)

input_tensor_HWC = input_tensor.squeeze(0).permute(1, 2, 0)
print("Input Tensor Shape (HWC):", input_tensor_HWC.shape)
print("First 10 elements of Input Tensor:"
      , input_tensor_HWC.flatten()[:10].tolist())
print("-" * 50)

# 4) 추론
with torch.no_grad():
    logits = model(input_tensor)

# 5) Top-5 결과
topk = 5
top_logits, top_indices = torch.topk(logits[0], k=topk)

# print("Top-5 indices:", top_indices.tolist())
# print("Top-5 logits:", top_logits.tolist())

# # 라벨 출력
# categories = weights.meta["categories"]
# for rank, (idx, logit) in enumerate(zip(top_indices, top_logits), start=1):
#     print(f"{rank}: idx={idx.item()}, logit={logit.item():.6f}, label={categories[idx]}")

# if debug:
#     # Image Normalize 전후 픽셀값 비교
#     loc = [(0,0), (56,56), (112,112)]
#     for y, x in loc:
#         pixel = input_tensor[0, :, y, x]  # 채널별 픽셀값
#         print(f"Pixel at ({y},{x}):")
#         print("[Before Normalize] R={:.4f}, G={:.4f}, B={:.4f}".format(
#             img.getpixel((x,y))[0],
#             img.getpixel((x,y))[1],
#             img.getpixel((x,y))[2]
#         ))
#         print("[After  Normalize] R={:.4f}, G={:.4f}, B={:.4f}".format(
#             pixel[0].item(), pixel[1].item(), pixel[2].item())
#         )
    


'''
[pyTorch]
- Top-5 indices: [3, 2, 4, 394, 5]
- Top-5 logits: [22.970073699951172, 16.97878074645996, 16.22966957092285, 14.80838394165039, 14.718649864196777]
    1: idx=3, logit=22.970074, label=tiger shark
    2: idx=2, logit=16.978781, label=great white shark
    3: idx=4, logit=16.229670, label=hammerhead
    4: idx=394, logit=14.808384, label=sturgeon
    5: idx=5, logit=14.718650, label=electric ray

[vAI]
- Top-5 Results:
    1. Class 845: 0.230635
    2. Class 470: 0.193255
    3. Class 523: 0.180866
    4. Class 178: 0.179018
    5. Class 650: 0.167551    
    
'''
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Load model
model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
model.eval()

# Hook to capture intermediate outputs
intermediate_outputs = {}
def get_activation(name):
    def hook(model, input, output):
        intermediate_outputs[name] = output.detach()
    return hook

# Register hooks matching C++ debug points
model.conv1.conv.register_forward_hook(get_activation('conv1.conv.out')) # Pre-BN
model.conv1.bn.register_forward_hook(get_activation('conv1.bn.out'))     # Post-BN
model.conv1.register_forward_hook(get_activation('conv1.out'))           # Post-ReLU
model.maxpool1.register_forward_hook(get_activation('pool1.out'))
model.conv2.register_forward_hook(get_activation('conv2_reduce.out')) # PyTorch conv2 is 1x1 (C++ conv2_reduce)
model.conv3.register_forward_hook(get_activation('conv2.out'))        # PyTorch conv3 is 3x3 (C++ conv2)
model.maxpool2.register_forward_hook(get_activation('pool2.out'))
model.inception3a.register_forward_hook(get_activation('inception3a.out'))
model.inception3b.register_forward_hook(get_activation('inception3b.out'))
model.maxpool3.register_forward_hook(get_activation('pool3.out'))
model.inception4a.register_forward_hook(get_activation('inception4a.out'))
model.inception4b.register_forward_hook(get_activation('inception4b.out'))
model.inception4c.register_forward_hook(get_activation('inception4c.out'))
model.inception4d.register_forward_hook(get_activation('inception4d.out'))
model.inception4e.register_forward_hook(get_activation('inception4e.out'))
model.maxpool4.register_forward_hook(get_activation('pool4.out'))
model.inception5a.register_forward_hook(get_activation('inception5a.out'))
model.inception5b.register_forward_hook(get_activation('inception5b.out'))
model.avgpool.register_forward_hook(get_activation('avgpool.out'))
model.fc.register_forward_hook(get_activation('fc.out'))

import os

# Load image
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, 'data', 'dog.jpg')
print(f"Loading image from: {img_path}")
img = Image.open(img_path).convert('RGB')

# Preprocessing (match C++ exactly: resize to 224x224 if needed, normalize)
# C++ logic: if 224x224, skip resize/crop.
if img.size == (224, 224):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
else:
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

input_tensor = preprocess(img).unsqueeze(0)

# Run inference
with torch.no_grad():
    output = model(input_tensor)

# Print Weight stats
print(f"[ref] conv1.weight stats: min={model.conv1.conv.weight.min():.4f}, max={model.conv1.conv.weight.max():.4f}, mean={model.conv1.conv.weight.mean():.4f}")
print(f"[ref] inception3a.1x1.weight stats: min={model.inception3a.branch1.conv.weight.min():.4f}, max={model.inception3a.branch1.conv.weight.max():.4f}, mean={model.inception3a.branch1.conv.weight.mean():.4f}")
# branch2 is 1x1 -> 3x3. We check 3x3 (index 1)
print(f"[ref] inception3a.3x3.weight stats: min={model.inception3a.branch2[1].conv.weight.min():.4f}, max={model.inception3a.branch2[1].conv.weight.max():.4f}, mean={model.inception3a.branch2[1].conv.weight.mean():.4f}")
# branch3 is 1x1 -> 5x5. We check 5x5 (index 1)
print(f"[ref] inception3a.5x5.weight stats: min={model.inception3a.branch3[1].conv.weight.min():.4f}, max={model.inception3a.branch3[1].conv.weight.max():.4f}, mean={model.inception3a.branch3[1].conv.weight.mean():.4f}")
# branch4 is Pool -> 1x1. We check 1x1 (index 1)
print(f"[ref] inception3a.pool_proj.weight stats: min={model.inception3a.branch4[1].conv.weight.min():.4f}, max={model.inception3a.branch4[1].conv.weight.max():.4f}, mean={model.inception3a.branch4[1].conv.weight.mean():.4f}")
print(f"[ref] fc.weight stats: min={model.fc.weight.min():.4f}, max={model.fc.weight.max():.4f}, mean={model.fc.weight.mean():.4f}")

# Print BN stats
bn = model.conv1.bn
print(f"[ref] conv1.bn.running_mean stats: min={bn.running_mean.min():.4f}, max={bn.running_mean.max():.4f}, mean={bn.running_mean.mean():.4f}")
print(f"[ref] conv1.bn.running_var stats: min={bn.running_var.min():.4f}, max={bn.running_var.max():.4f}, mean={bn.running_var.mean():.4f}")
print(f"[ref] conv1.bn.weight (gamma) stats: min={bn.weight.min():.4f}, max={bn.weight.max():.4f}, mean={bn.weight.mean():.4f}")
print(f"[ref] conv1.bn.bias (beta) stats: min={bn.bias.min():.4f}, max={bn.bias.max():.4f}, mean={bn.bias.mean():.4f}")
if model.conv1.conv.bias is not None:
    print(f"[ref] conv1.conv.bias stats: min={model.conv1.conv.bias.min():.4f}, max={model.conv1.conv.bias.max():.4f}, mean={model.conv1.conv.bias.mean():.4f}")
else:
    print("[ref] conv1.conv.bias is None")

print(f"Model transform_input: {model.transform_input}")

# Explicit run of conv1 for comparison
with torch.no_grad():
    explicit_out = model.conv1(input_tensor)
    print(f"[Explicit] conv1 output stats: min={explicit_out.min():.4f}, max={explicit_out.max():.4f}, mean={explicit_out.mean():.4f}")
    
    explicit_conv = model.conv1.conv(input_tensor)
    print(f"[Explicit] conv1.conv (Pre-BN) stats: min={explicit_conv.min():.4f}, max={explicit_conv.max():.4f}")
    print(f"[Explicit] conv1.conv (Pre-BN) first 10: {explicit_conv.flatten()[:10].tolist()}")

# Print stats
print(f"Input stats: min={input_tensor.min():.4f}, max={input_tensor.max():.4f}, mean={input_tensor.mean():.4f}")
print(f"Input first 6: {input_tensor.flatten()[:6].tolist()}")

for name, tensor in intermediate_outputs.items():
    t = tensor.flatten()
    print(f"[ref] {name} stats (min/max/mean/std): {t.min():.4f} / {t.max():.4f} / {t.mean():.4f} / {t.std():.4f} shape={list(tensor.shape)}")

# Top 5
probs = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_catid = torch.topk(probs, 5)
print("Top-5 predictions:")
for i in range(5):
    print(f"{top5_catid[i]}: {top5_prob[i]:.6f}")

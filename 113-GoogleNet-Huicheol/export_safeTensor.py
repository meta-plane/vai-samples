import torch
import torch.nn as nn
import torchvision.models as models
from safetensors.torch import save_file
import os

def fuse_conv_bn(conv, bn):
    """
    Fuses Conv2d and BatchNorm2d into a single Conv2d.
    w_fused = w * (gamma / sqrt(var + eps))
    b_fused = (b - mean) * (gamma / sqrt(var + eps)) + beta
    """
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=True
    )
    
    # Disable gradients
    conv.weight.requires_grad = False
    bn.weight.requires_grad = False
    bn.bias.requires_grad = False
    bn.running_mean.requires_grad = False
    bn.running_var.requires_grad = False
    
    # Calculate fusion parameters
    factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    
    print(f"    [debug] conv.weight stats: min={conv.weight.min():.4f}, max={conv.weight.max():.4f}, mean={conv.weight.mean():.4f}")
    print(f"    [debug] factor stats: min={factor.min():.4f}, max={factor.max():.4f}, mean={factor.mean():.4f}")
    
    # Fuse weight
    # conv.weight shape: [out_ch, in_ch, k, k]
    # factor shape: [out_ch]
    # We need to reshape factor to [out_ch, 1, 1, 1] for broadcasting
    with torch.no_grad():
        fused_weight = conv.weight * factor.reshape(-1, 1, 1, 1)
        print(f"    [debug] fused_weight stats: min={fused_weight.min():.4f}, max={fused_weight.max():.4f}, mean={fused_weight.mean():.4f}")
        
        # Fuse bias
        # If conv has bias, include it. (GoogLeNet BasicConv2d usually has bias=False because of BN)
        conv_bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
        fused_bias = (conv_bias - bn.running_mean) * factor + bn.bias
        
        fused_conv.weight.copy_(fused_weight)
        fused_conv.bias.copy_(fused_bias)
    
    return fused_conv

def export_weights():
    print("Loading PyTorch GoogLeNet (IMAGENET1K_V1)...")
    model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
    model.eval()
    
    state_dict = {}
    
    def add_conv(name, module):
        # module is BasicConv2d, which has .conv and .bn
        print(f"Fusing {name}...")
        print(f"  bn.running_var mean: {module.bn.running_var.mean():.4f}")
        print(f"  bn.weight (gamma) mean: {module.bn.weight.mean():.4f}")
        fused = fuse_conv_bn(module.conv, module.bn)
        state_dict[f"{name}.weight"] = fused.weight.data
        state_dict[f"{name}.bias"] = fused.bias.data
        print(f"  Fused {name}: weight range [{fused.weight.min():.4f}, {fused.weight.max():.4f}]")

    # Stem
    add_conv("conv1", model.conv1)
    add_conv("conv2_reduce", model.conv2) # PyTorch conv2 (1x1) -> C++ conv2_reduce
    add_conv("conv2", model.conv3)        # PyTorch conv3 (3x3) -> C++ conv2
    
    # Inception Blocks
    # PyTorch Inception structure:
    # branch1: BasicConv2d (1x1)
    # branch2: Sequential(BasicConv2d (1x1), BasicConv2d (3x3))
    # branch3: Sequential(BasicConv2d (1x1), BasicConv2d (5x5))
    # branch4: Sequential(MaxPool, BasicConv2d (1x1))
    
    def add_inception(name, module):
        # branch1 (1x1)
        add_conv(f"{name}.1x1", module.branch1)
        
        # branch2 (1x1 -> 3x3)
        add_conv(f"{name}.3x3_reduce", module.branch2[0])
        add_conv(f"{name}.3x3", module.branch2[1])
        
        # branch3 (1x1 -> 5x5)
        add_conv(f"{name}.5x5_reduce", module.branch3[0])
        add_conv(f"{name}.5x5", module.branch3[1])
        
        # branch4 (Pool -> 1x1)
        add_conv(f"{name}.pool_proj", module.branch4[1])

    add_inception("inception3a", model.inception3a)
    add_inception("inception3b", model.inception3b)
    
    add_inception("inception4a", model.inception4a)
    add_inception("inception4b", model.inception4b)
    add_inception("inception4c", model.inception4c)
    add_inception("inception4d", model.inception4d)
    add_inception("inception4e", model.inception4e)
    
    add_inception("inception5a", model.inception5a)
    add_inception("inception5b", model.inception5b)
    
    # FC
    # PyTorch FC has bias=True
    state_dict["fc.weight"] = model.fc.weight.data
    state_dict["fc.bias"] = model.fc.bias.data
    print(f"Added fc: weight range [{model.fc.weight.min():.4f}, {model.fc.weight.max():.4f}]")
    
    # Verification
    print("\nVerifying fusion correctness on conv1...")
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Original (Conv -> BN -> ReLU)
    with torch.no_grad():
        out_orig = model.conv1(dummy_input)
        
    # Fused
    fused_conv1 = fuse_conv_bn(model.conv1.conv, model.conv1.bn)
    with torch.no_grad():
        out_fused_pre_relu = fused_conv1(dummy_input)
        out_fused = torch.nn.functional.relu(out_fused_pre_relu)
        
    diff = (out_orig - out_fused).abs().max().item()
    print(f"Max difference between Original (Conv+BN+ReLU) and Fused (Conv+ReLU): {diff:.6f}")
    
    if diff > 1e-3:
        print("WARNING: Fusion mismatch! Check logic.")
    else:
        print("Fusion verified.")

    # Verification on dog.jpg
    print("\nVerifying fusion correctness on dog.jpg...")
    try:
        from PIL import Image
        import torchvision.transforms as transforms
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(script_dir, 'data', 'dog.jpg')
        img = Image.open(img_path).convert('RGB')
        
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
        
        # Original (Conv -> BN -> ReLU)
        with torch.no_grad():
            out_orig = model.conv1(input_tensor)
            
        # Fused
        # fused_conv1 is already created above
        with torch.no_grad():
            out_fused_pre_relu = fused_conv1(input_tensor)
            out_fused = torch.nn.functional.relu(out_fused_pre_relu)
            
        diff = (out_orig - out_fused).abs().max().item()
        print(f"Max difference on dog.jpg: {diff:.6f}")
        print(f"Original max: {out_orig.max():.4f}")
        print(f"Fused max:    {out_fused.max():.4f}")
        
    except Exception as e:
        print(f"Skipping dog.jpg verification: {e}")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights_fixed.safetensors")
    save_file(state_dict, out_path)
    print(f"Saved fixed weights to {out_path}")

if __name__ == "__main__":
    try:
        export_weights()
    except ImportError:
        print("Error: 'safetensors' library not found. Please install it with: pip install safetensors")

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

def run():
    print("Loading model...")
    model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
    model.eval()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, 'data', 'dog.jpg')
    print(f"Loading {img_path}")
    img = Image.open(img_path).convert('RGB')

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(img).unsqueeze(0)
    print(f"Image size: {img.size}")
    print(f"Input stats: min={input_tensor.min():.4f}, max={input_tensor.max():.4f}")
    print(f"Input first 10: {input_tensor.flatten()[:10].tolist()}")

    with torch.no_grad():
        # Check weights
        w = model.conv1.conv.weight
        print(f"Weight stats: min={w.min():.4f}, max={w.max():.4f}")
        print(f"Weight first 10: {w.flatten()[:10].tolist()}")

        # Run full conv1 (Conv+BN+ReLU)
        out = model.conv1(input_tensor)
        print(f"Conv1 Output stats: min={out.min():.4f}, max={out.max():.4f}, mean={out.mean():.4f}")
        print(f"Conv1 Output first 10: {out.flatten()[:10].tolist()}")

        # Run parts manually
        x = model.conv1.conv(input_tensor)
        print(f"Conv1.conv (Pre-BN) stats: min={x.min():.4f}, max={x.max():.4f}")
        print(f"Conv1.conv (Pre-BN) first 10: {x.flatten()[:10].tolist()}")
        
        x = model.conv1.bn(x)
        print(f"Conv1.bn (Pre-ReLU) stats: min={x.min():.4f}, max={x.max():.4f}")
        
        x = torch.nn.functional.relu(x)
        print(f"Conv1.relu (Final) stats: min={x.min():.4f}, max={x.max():.4f}")

if __name__ == "__main__":
    run()

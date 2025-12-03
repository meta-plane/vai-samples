"""
MobileNetV2 Pretrained Weights Download Script
==============================================

이 스크립트는 PyTorch의 MobileNetV2 pretrained 가중치를 다운로드하고
SafeTensors 포맷으로 저장합니다.

설치 필요 패키지:
    pip install torch torchvision safetensors

사용법:
    python download_mobilenetv2_weights.py

출력:
    mobilenetv2_weights.safetensors
"""

import torch
import torchvision.models as models
from safetensors.torch import save_file
import json


def download_and_save_mobilenetv2():
    print("=" * 60)
    print("MobileNetV2 Pretrained Weights Downloader")
    print("=" * 60)
    
    # 1. MobileNetV2 모델 로드 (ImageNet pretrained)
    print("\n[1/4] Loading MobileNetV2 pretrained model...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.eval()
    
    # 2. State dict 가져오기
    print("[2/4] Extracting state dict...")
    state_dict = model.state_dict()
    
    # 3. 가중치 정보 출력
    print(f"\n[3/4] Model has {len(state_dict)} tensors:")
    print("-" * 60)
    
    total_params = 0
    for name, tensor in state_dict.items():
        shape_str = str(list(tensor.shape))
        num_params = tensor.numel()
        total_params += num_params
        print(f"  {name}: {shape_str}, {num_params:,} params")
    
    print("-" * 60)
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 4. SafeTensors로 저장
    print("\n[4/4] Saving to SafeTensors format...")
    output_file = "mobilenetv2_weights.safetensors"
    save_file(state_dict, output_file)
    
    print(f"\n✓ Saved to: {output_file}")
    
    # 파일 크기 출력
    import os
    file_size = os.path.getsize(output_file)
    print(f"  File size: {file_size / 1024 / 1024:.2f} MB")
    
    # 레이어 이름 매핑 정보 저장 (C++에서 참조용)
    print("\n[Bonus] Saving layer name mapping...")
    layer_info = {}
    for name, tensor in state_dict.items():
        layer_info[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": tensor.numel()
        }
    
    with open("mobilenetv2_layer_info.json", "w") as f:
        json.dump(layer_info, f, indent=2)
    
    print("✓ Saved layer info to: mobilenetv2_layer_info.json")
    
    return output_file


def print_layer_structure():
    """MobileNetV2 레이어 구조 출력 (가중치 이름 매핑용)"""
    print("\n" + "=" * 60)
    print("MobileNetV2 Layer Name Structure")
    print("=" * 60)
    print("""
PyTorch MobileNetV2 가중치 이름 규칙:
=====================================

features.0.0.weight          - 첫 번째 Conv (3->32, k=3, s=2)
features.0.1.weight/bias/... - 첫 번째 BatchNorm

features.1.conv.0.0.weight   - 첫 번째 InvertedResidual의 Depthwise Conv
features.1.conv.0.1.*        - 첫 번째 InvertedResidual의 Depthwise BN
features.1.conv.1.weight     - 첫 번째 InvertedResidual의 Projection Conv
features.1.conv.2.*          - 첫 번째 InvertedResidual의 Projection BN

features.N.conv.0.0.weight   - N번째 IRB의 Expansion Conv (if t>1)
features.N.conv.0.1.*        - N번째 IRB의 Expansion BN
features.N.conv.1.0.weight   - N번째 IRB의 Depthwise Conv
features.N.conv.1.1.*        - N번째 IRB의 Depthwise BN
features.N.conv.2.weight     - N번째 IRB의 Projection Conv
features.N.conv.3.*          - N번째 IRB의 Projection BN

features.18.0.weight         - 마지막 Conv (320->1280, k=1)
features.18.1.*              - 마지막 BatchNorm

classifier.1.weight          - FC 가중치 (1280->1000)
classifier.1.bias            - FC 바이어스

BatchNorm 파라미터:
- weight = gamma
- bias = beta  
- running_mean
- running_var
- num_batches_tracked (무시)
""")


def test_model():
    """모델 테스트 (선택사항)"""
    print("\n[Test] Running inference test...")
    
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.eval()
    
    # 더미 입력 (224x224 RGB)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  Input shape: {list(dummy_input.shape)}")
    print(f"  Output shape: {list(output.shape)}")
    print(f"  Top-5 class indices: {output.topk(5).indices.tolist()}")
    print("✓ Model inference test passed!")


if __name__ == "__main__":
    # 가중치 다운로드 및 저장
    download_and_save_mobilenetv2()
    
    # 레이어 구조 출력
    print_layer_structure()
    
    # 모델 테스트 (선택)
    test_model()
    
    print("\n" + "=" * 60)
    print("Done! You can now use the .safetensors file in C++")
    print("=" * 60)

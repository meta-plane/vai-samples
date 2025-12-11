import torch
import numpy as np
import argparse
from models.unet import Unet
from safetensors.torch import save_file

# ==========================================
# 1. 변환 헬퍼 함수 (C++ Vulkan 메모리 레이아웃 최적화)
# ==========================================

def _to_cpu_np(t):
    return t.detach().cpu().numpy()

def _tn(x: np.ndarray) -> torch.Tensor:
    """Numpy 배열을 메모리 연속적인(Contiguous) Float32 Tensor로 변환"""
    return torch.from_numpy(np.ascontiguousarray(x.astype(np.float32, copy=False))).contiguous()

def conv_to_vkB(w_tensor, is_deconv=False):
    """
    C++ Vulkan Shader (GEMM)를 위한 가중치 변환 함수
    
    [목표 형태]
    Output Shape: [In * kH * kW, Out] 
    
    이 형태는 행렬 곱(MatMul)을 위해 채널과 커널을 행(Row)으로 묶고, 
    출력 채널을 열(Col)로 배치한 것입니다.
    """
    w = _to_cpu_np(w_tensor) # Numpy 변환 (float32)

    if is_deconv:
        # ---------------------------------------------------------
        # Case 1: ConvTranspose2d (Deconvolution)
        # PyTorch 저장 형태: [In, Out, kH, kW]
        # ---------------------------------------------------------
        I, O, kH, kW = w.shape
        
        # 목표: [In * kH * kW, Out]
        # 설명: PyTorch는 (I, O, H, W) 순서이므로, O를 맨 뒤로 보내야 합니다.
        # 1. Permute: (In, Out, kH, kW) -> (In, kH, kW, Out)
        w = w.transpose(0, 2, 3, 1) 
        
        # 2. Reshape: (In * kH * kW, Out)
        return w.reshape(I * kH * kW, O).astype(np.float32, copy=True)

    else:
        # ---------------------------------------------------------
        # Case 2: Conv2d (Standard Convolution)
        # PyTorch 저장 형태: [Out, In, kH, kW]
        # ---------------------------------------------------------
        O, I, kH, kW = w.shape
        
        # 목표: [In * kH * kW, Out]
        # 설명: (Out, In, H, W)를 (Out, In*H*W)로 펴준 뒤, 
        #      행/열을 바꿔서 (In*H*W, Out)으로 만듭니다.
        
        # 1. Reshape: [Out, In * kH * kW]
        # 2. Transpose: [In * kH * kW, Out]
        return w.reshape(O, I * kH * kW).transpose(1, 0).astype(np.float32, copy=True)

def bn_to_dict(bn_layer):
    """BatchNorm 파라미터 추출"""
    return {
        "weight": _to_cpu_np(bn_layer.weight),       # gamma
        "bias": _to_cpu_np(bn_layer.bias),           # beta
        "running_mean": _to_cpu_np(bn_layer.running_mean), # mean
        "running_var": _to_cpu_np(bn_layer.running_var),   # variance
    }

# ==========================================
# 2. 블록별 추출 함수 (C++ 키 이름과 정확히 매칭)
# ==========================================

def extract_conv_bn_relu(layer, prefix, tensors_dict):
    """
    ConvBnRelu 블록 처리
    prefix 예시: "encoderConv1_.ConvBnRelu_1"
    """
    # 1. Conv Weight & Bias
    tensors_dict[f"{prefix}.conv.weight"] = _tn(conv_to_vkB(layer.conv.weight))
    
    if layer.conv.bias is not None:
        tensors_dict[f"{prefix}.conv.bias"] = _tn(_to_cpu_np(layer.conv.bias))
    else:
        out_ch = layer.conv.out_channels
        tensors_dict[f"{prefix}.conv.bias"] = torch.zeros((out_ch,), dtype=torch.float32)

    # 2. BatchNorm (gamma, beta, mean, var)
    bn_params = bn_to_dict(layer.bn)
    for k, v in bn_params.items():
        # k는 weight, bias, running_mean, running_var 중 하나
        tensors_dict[f"{prefix}.bn.{k}"] = _tn(v)

def extract_double_conv(layer, block_name, tensors_dict):
    """
    DoubleConvBnRelu 블록 처리
    block_name 예시: "encoderConv1_"
    생성되는 키: "encoderConv1_.ConvBnRelu_1.conv.weight" 등
    """
    # 첫 번째 ConvBnRelu
    extract_conv_bn_relu(layer.ConvBnRelu_1, f"{block_name}.ConvBnRelu_1", tensors_dict)
    # 두 번째 ConvBnRelu
    extract_conv_bn_relu(layer.ConvBnRelu_2, f"{block_name}.ConvBnRelu_2", tensors_dict)

# ==========================================
# 3. 메인 변환 로직
# ==========================================

def export_unet_for_cpp(model, out_path):
    model.eval()
    T = {} # 저장할 텐서 딕셔너리

    print(f"Exporting UNet weights to {out_path} ...")

    # ---------------------------------------------------------
    # Encoders
    # C++ Key: encoderConvX_.ConvBnRelu_Y...
    # ---------------------------------------------------------
    extract_double_conv(model.encoderConv1_, "encoderConv1_", T)
    extract_double_conv(model.encoderConv2_, "encoderConv2_", T)
    extract_double_conv(model.encoderConv3_, "encoderConv3_", T)
    extract_double_conv(model.encoderConv4_, "encoderConv4_", T)

    # ---------------------------------------------------------
    # Bottleneck
    # C++ Key: bottleneck_.ConvBnRelu_Y...
    # ---------------------------------------------------------
    extract_double_conv(model.bottleneck_, "bottleneck_", T)

    # ---------------------------------------------------------
    # Decoders (UpConv + DoubleConv)
    # ---------------------------------------------------------
    
    # Decoder 4
    T["upConv4_.weight"] = _tn(conv_to_vkB(model.upConv4_.weight, is_deconv=True))
    T["upConv4_.bias"]   = _tn(_to_cpu_np(model.upConv4_.bias))
    extract_double_conv(model.decoderConv4_, "decoderConv4_", T)

    # Decoder 3
    T["upConv3_.weight"] = _tn(conv_to_vkB(model.upConv3_.weight, is_deconv=True))
    T["upConv3_.bias"]   = _tn(_to_cpu_np(model.upConv3_.bias))
    extract_double_conv(model.decoderConv3_, "decoderConv3_", T)

    # Decoder 2
    T["upConv2_.weight"] = _tn(conv_to_vkB(model.upConv2_.weight, is_deconv=True))
    T["upConv2_.bias"]   = _tn(_to_cpu_np(model.upConv2_.bias))
    extract_double_conv(model.decoderConv2_, "decoderConv2_", T)

    # Decoder 1
    T["upConv1_.weight"] = _tn(conv_to_vkB(model.upConv1_.weight, is_deconv=True))
    T["upConv1_.bias"]   = _tn(_to_cpu_np(model.upConv1_.bias))
    extract_double_conv(model.decoderConv1_, "decoderConv1_", T)

    # ---------------------------------------------------------
    # Header
    # C++ Key: header_.weight
    # ---------------------------------------------------------
    T["header_.weight"] = _tn(conv_to_vkB(model.header_.weight))
    if model.header_.bias is not None:
        T["header_.bias"] = _tn(_to_cpu_np(model.header_.bias))
    else:
        T["header_.bias"] = torch.zeros((model.out_channels,), dtype=torch.float32)

    # 파일 저장
    save_file(T, out_path)
    print(f"✅ Export Complete! Total tensors: {len(T)}")

# ==========================================
# 실행 부분
# ==========================================
if __name__ == "__main__":
    # 1. 모델 생성
    net = Unet(in_channels=3, out_channels=1)
    
    # 2. 체크포인트 로드 (학습된 파일이 있다면 주석 해제)
    checkpoint_path = "workspace/checkpoint/epoch_0049/unet.pth"
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    net.load_state_dict(state_dict["model_state_dict"])

    # 3. 변환 실행
    export_unet_for_cpp(net, "unet.safetensors")
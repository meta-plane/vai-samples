import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import cv2
import numpy as np

from utils import save_tensor_bin

def imageLoader(path: str) -> np.ndarray:
    
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR, uint8
    
    if img is None:
        raise FileNotFoundError(f"이미지를 읽지 못했습니다: {path}")
    
    return img

def test_ConvTranspose(root:str,tensor : torch.Tensor):
    weight_path     = "convTrans_weight.bin"
    bias_path       = "convTrans_bias.bin"
    result_path     = "convTrans_result.bin"

    # ConvTranspose2d 설정 예시
    in_channels = 3
    out_channels = 8
    kernel_size = 2
    stride = 2
    padding = 0
    bias = True

    convTrans = nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias
    )

    with torch.no_grad():
        y = convTrans(tensor)

    save_tensor_bin(os.path.join(root, weight_path), convTrans.weight)

    if bias:
        save_tensor_bin(os.path.join(root, bias_path), convTrans.bias)

    save_tensor_bin(os.path.join(root,result_path), y)

    print("test_ConvTranspose 완료")
    
def test_batchNorm(root: str, tensor: torch.Tensor):
    """
    입력 tensor: (N, C, H, W)
    - BatchNorm2d 파라미터(감마/베타)와 러닝통계(mean/var), 출력 결과 저장
    """
    # 저장 파일명
    gamma_path = "batchNorm_gamma.bin"          # weight (gamma)
    beta_path = "batchNorm_beta.bin"            # bias (beta)
    running_mean_path = "batchNorm_running_mean.bin"
    running_var_path = "batchNorm_running_var.bin"
    result_path = "out_batchNorm_result.bin"

    # 설정 예시
    num_features = tensor.shape[1]  # C
    eps = 1e-5
    momentum = 0.1
    affine = True
    track_running_stats = True

    bn = nn.BatchNorm2d(
        num_features=num_features,
        eps=eps,
        momentum=momentum,
        affine=affine,
        track_running_stats=track_running_stats
    )

    # 러닝통계가 업데이트되지 않도록 eval()로 고정 (재현성/고정 출력 목적)
    bn.eval()

    with torch.no_grad():
        y = bn(tensor)

    os.makedirs(root, exist_ok=True)

    if affine:
        save_tensor_bin(os.path.join(root, gamma_path), bn.weight)  # gamma
        save_tensor_bin(os.path.join(root, beta_path), bn.bias)     # beta

    if track_running_stats:
        save_tensor_bin(os.path.join(root, running_mean_path), bn.running_mean)
        save_tensor_bin(os.path.join(root, running_var_path), bn.running_var)

    save_tensor_bin(os.path.join(root, result_path), y)

    print("test_batchNorm 완료")

def test_Sigmoid(root: str, tensor: torch.Tensor):
    """
    Sigmoid는 파라미터가 없으므로 출력 결과만 저장
    """
    result_path = "out_sigmoid_result.bin"

    with torch.no_grad():
        y = torch.sigmoid(tensor)  # 또는 nn.Sigmoid()(tensor)

    os.makedirs(root, exist_ok=True)
    save_tensor_bin(os.path.join(root, result_path), y)

    print("test_Sigmoid 완료")

def test_concat(root: str, tensor: torch.Tensor):
    """
    concat 테스트:
    - 보통 채널 차원(dim=1)으로 concat하는 케이스가 많아서 그 예시로 작성
    - 두 번째 입력 텐서도 bin으로 저장 (디버깅/검증용)
    """
    in0_path = "concat_in0.bin"
    in1_path = "concat_in1.bin"
    result_path = "out_concat_result.bin"

    # 예시: 같은 shape의 텐서 2개를 만들어 채널 방향 concat
    # tensor: (N, C, H, W) -> cat 후 (N, 2C, H, W)
    t0 = tensor
    t1 = tensor * 0.5  # 두 번째 입력 예시(검증이 쉬움)

    dim = 1  # channel dim

    with torch.no_grad():
        y = torch.cat([t0, t1], dim=dim)

    os.makedirs(root, exist_ok=True)
    save_tensor_bin(os.path.join(root, in0_path), t0)
    save_tensor_bin(os.path.join(root, in1_path), t1)
    save_tensor_bin(os.path.join(root, result_path), y)

    print("test_concat 완료")

def main():
    # ---- 설정 ----
    image_path          = "D:\\VAI\\images\\image.png"
    root_path           = "workspace//bin"

    # ---- 1) 이미지 로드 ----
    img_bgr = imageLoader(image_path)  # HWC, uint8, BGR
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ---- 2) Tensor 변환: (1, C, H, W), float32 [0,1] ----
    x = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).contiguous()
    x = x.float() / 255.0

    test_ConvTranspose(root=root_path, tensor=x)
    test_batchNorm(root=root_path, tensor=x)
    test_Sigmoid(root=root_path, tensor=x)
    test_concat(root=root_path, tensor=x)

if __name__ == "__main__":
    main()

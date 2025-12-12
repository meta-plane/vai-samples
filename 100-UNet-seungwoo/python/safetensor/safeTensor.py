# import torch
# import os
# from safetensors.torch import save_file
# from collections import OrderedDict

# INPUT_PTH = './weight/U_Net_Model.pth'
# OUTPUT_SAFE = './weight/U_Net_Model.safetensors'

# def convert_to_safetensors(input_path, output_path):
#     if not os.path.exists(input_path):
#         print(f"파일을 찾을 수 없습니다: {input_path}")
#         return

#     print(f"변환 시작: {input_path} -> {output_path}")

#     # 1. pth 로드
#     try:
#         checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
#     except Exception as e:
#         print(f"pth 파일 로드 실패: {e}")
#         return

#     print("=== checkpoint 타입:", type(checkpoint))

#     # 2. state_dict 추출
#     state_dict = None

#     if isinstance(checkpoint, dict):
#         print("=== checkpoint keys & types ===")
#         for k, v in checkpoint.items():
#             print(f"  {k}: {type(v)}")

#         if "model_state_dict" in checkpoint:
#             print(" -> 'model_state_dict' 키를 감지했습니다.")
#             state_dict = checkpoint["model_state_dict"]
#         elif "state_dict" in checkpoint:
#             print(" -> 'state_dict' 키를 감지했습니다.")
#             state_dict = checkpoint["state_dict"]
#         elif "net" in checkpoint:
#             print(" -> 'net' 키를 감지했습니다.")
#             net_obj = checkpoint["net"]
#             if isinstance(net_obj, (dict, OrderedDict)):
#                 state_dict = net_obj
#             elif isinstance(net_obj, torch.nn.Module):
#                 state_dict = net_obj.state_dict()
#             else:
#                 print(" [경고] 'net' 타입이 예상과 다릅니다:", type(net_obj))
#                 return
#         else:
#             print(" -> 단순 딕셔너리 구조로 감지했습니다.")
#             state_dict = checkpoint

#     elif isinstance(checkpoint, torch.nn.Module):
#         print(" -> 모델 객체(nn.Module)를 감지했습니다.")
#         state_dict = checkpoint.state_dict()
#     else:
#         print(" -> 알 수 없는 파일 형식입니다.")
#         return

#     if not isinstance(state_dict, dict):
#         print("state_dict가 dict가 아닙니다. 타입:", type(state_dict))
#         return

#     print("=== 최종 state_dict 샘플 키 & 타입 ===")
#     for k, v in list(state_dict.items())[:20]:
#         print(f"  {k}: {type(v)}")

#     # 3. safetensors용 clean_state_dict 생성
#     clean_state_dict = {}
#     for key, value in state_dict.items():
#         if isinstance(value, torch.Tensor):
#             clean_state_dict[key] = value.contiguous()
#         else:
#             print(f" [경고] 텐서가 아닌 데이터 제외됨: {key} (타입: {type(value)})")

#     print(f"총 텐서 개수: {len(clean_state_dict)}")
#     if len(clean_state_dict) == 0:
#         print("[에러] 저장할 텐서가 하나도 없습니다. checkpoint 구조를 다시 확인하세요.")
#         return

#     # 4. safetensors 저장
#     try:
#         save_file(clean_state_dict, output_path)
#         print(f"변환 완료! 저장됨: {output_path}")
#     except Exception as e:
#         print(f"저장 중 에러 발생: {e}")

# if __name__ == "__main__":
#     convert_to_safetensors(INPUT_PTH, OUTPUT_SAFE)

import torch
import os
import numpy as np
from safetensors.torch import save_file
from collections import OrderedDict

INPUT_PTH = './weight/unet.pth'
OUTPUT_SAFE = './weight/unet.safetensors'

# --- [변환 헬퍼 함수 시작] ---
def _to_cpu_np(t):
    return t.detach().cpu().numpy()

def _tn(x: np.ndarray) -> torch.Tensor:
    """Numpy 배열을 Contiguous한 Float32 Torch Tensor로 변환"""
    return torch.from_numpy(np.ascontiguousarray(x.astype(np.float32, copy=False))).contiguous()

def conv_to_vkB(w_tensor):
    """
    PyTorch Conv2d weights: [out, in, kH, kW]
    C++ ConvolutionNode expects B: [in*k*k, out]
    Returns np.float32 array of shape [I*K*K, O].
    """
    w = _to_cpu_np(w_tensor)
    if w.ndim != 4:
        return w # 4차원이 아니면 원본 반환 (예외 처리)
        
    O, I, kH, kW = w.shape
    # [Out, In, H, W] -> [Out, In*H*W] -> [In*H*W, Out] (Transpose)
    return w.reshape(O, I * kH * kW).transpose(1, 0).astype(np.float32, copy=True)

def linear_to_vkB(w_tensor):
    """
    PyTorch Linear weights: [out, in]
    C++ FC expects [in, out]
    """
    w = _to_cpu_np(w_tensor)
    if w.ndim != 2:
        return w
    # [Out, In] -> [In, Out] (Transpose)
    return w.transpose(1, 0).astype(np.float32, copy=True)
# --- [변환 헬퍼 함수 끝] ---


def convert_to_safetensors(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"파일을 찾을 수 없습니다: {input_path}")
        return

    print(f"변환 시작: {input_path} -> {output_path}")

    # 1. pth 로드
    try:
        checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"pth 파일 로드 실패: {e}")
        return

    print("=== checkpoint 타입:", type(checkpoint))

    # 2. state_dict 추출
    state_dict = None

    if isinstance(checkpoint, dict):
        # 키 확인용 출력 (생략 가능)
        # print("=== checkpoint keys & types ===")
        # for k, v in checkpoint.items():
        #     print(f"  {k}: {type(v)}")

        if "model_state_dict" in checkpoint:
            print(" -> 'model_state_dict' 키를 감지했습니다.")
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            print(" -> 'state_dict' 키를 감지했습니다.")
            state_dict = checkpoint["state_dict"]
        elif "net" in checkpoint:
            print(" -> 'net' 키를 감지했습니다.")
            net_obj = checkpoint["net"]
            if isinstance(net_obj, (dict, OrderedDict)):
                state_dict = net_obj
            elif isinstance(net_obj, torch.nn.Module):
                state_dict = net_obj.state_dict()
            else:
                print(" [경고] 'net' 타입이 예상과 다릅니다:", type(net_obj))
                return
        else:
            print(" -> 단순 딕셔너리 구조로 감지했습니다.")
            state_dict = checkpoint

    elif isinstance(checkpoint, torch.nn.Module):
        print(" -> 모델 객체(nn.Module)를 감지했습니다.")
        state_dict = checkpoint.state_dict()
    else:
        print(" -> 알 수 없는 파일 형식입니다.")
        return

    if not isinstance(state_dict, dict):
        print("state_dict가 dict가 아닙니다. 타입:", type(state_dict))
        return

    print("=== 최종 state_dict 샘플 키 & 타입 (상위 5개) ===")
    for k, v in list(state_dict.items())[:5]:
        print(f"  {k}: {type(v)} | Shape: {v.shape if isinstance(v, torch.Tensor) else 'N/A'}")

    # 3. safetensors용 clean_state_dict 생성 및 형상 변환 적용
    clean_state_dict = {}
    converted_count = 0
    
    print("\n=== 텐서 변환 및 패킹 시작 ===")
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            # 3-1. Convolution Weight 감지 (4차원 + 이름에 weight 포함)
            if value.ndim == 4 and "weight" in key:
                # [Out, In, kH, kW] -> [In*k*k, Out] 변환
                np_converted = conv_to_vkB(value)
                clean_state_dict[key] = _tn(np_converted)
                converted_count += 1
                # 로그가 너무 많으면 주석 처리하세요
                # print(f" [Conv 변환] {key}: {value.shape} -> {clean_state_dict[key].shape}")
            
            # 3-2. Linear Weight 감지 (2차원 + 이름에 weight 포함)
            elif value.ndim == 2 and "weight" in key:
                # [Out, In] -> [In, Out] 변환
                np_converted = linear_to_vkB(value)
                clean_state_dict[key] = _tn(np_converted)
                converted_count += 1
                print(f" [Linear 변환] {key}: {value.shape} -> {clean_state_dict[key].shape}")

            # 3-3. 그 외 (Bias, BatchNorm 등) -> 그대로 저장
            else:
                # 안전하게 Contiguous Float32로 통일
                clean_state_dict[key] = value.contiguous()
        else:
            print(f" [경고] 텐서가 아닌 데이터 제외됨: {key} (타입: {type(value)})")

    print(f"총 텐서 개수: {len(clean_state_dict)}")
    print(f"변환된 레이어(Conv/Linear) 개수: {converted_count}")

    if len(clean_state_dict) == 0:
        print("[에러] 저장할 텐서가 하나도 없습니다. checkpoint 구조를 다시 확인하세요.")
        return

    # 4. safetensors 저장
    try:
        save_file(clean_state_dict, output_path)
        print(f"변환 완료! 저장됨: {output_path}")
    except Exception as e:
        print(f"저장 중 에러 발생: {e}")

from safetensors import safe_open

if __name__ == "__main__":
    # convert_to_safetensors(INPUT_PTH, OUTPUT_SAFE)

    with safe_open(OUTPUT_SAFE, framework="torch") as f:
        print("keys:", f.keys())
        total_bytes = 0
        for k in f.keys():
            t = f.get_tensor(k)
            print(k, t.shape, t.dtype)
            total_bytes += t.numel() * t.element_size()
        print("총 크기:", total_bytes / (1024**2), "MB")
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

ROOT_PATH = 'D://VAI//weight//'
INPUT_PTH = 'unet.pth'
OUTPUT_SAFE = 'unet.safetensors'

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
    w_t = w.transpose(1, 2, 3, 0)

    I = w_t.shape[0] # In Channel
    H = w_t.shape[1] # Kernel Height
    W = w_t.shape[2] # Kernel Width
    O = w_t.shape[3] # Out Channel
        
    # [Out, In, H, W] -> [Out, In*H*W] -> [In*H*W, Out] (Transpose)
    return w_t.reshape(I * H * W, O).astype(np.float32, copy=True)

def conv_transpose_to_vkB(w_tensor):
    """
    [C++ Logic Mapping]
    Origin: [In, Out, H, W]
    Permute: (0, 2, 3, 1) -> [In, H, W, Out]
    Reshape: (In * H * W, Out)
    """
    w = _to_cpu_np(w_tensor) # Shape: (In, Out, H, W)

    # 1. Permute (순서 변경): [In, Out, H, W] -> [In, H, W, Out]
    # C++: w.permute(0, 2, 3, 1)
    w_t = w.transpose(0, 2, 3, 1)

    I = w_t.shape[0] # In Channel
    H = w_t.shape[1] # Kernel Height
    W = w_t.shape[2] # Kernel Width
    O = w_t.shape[3] # Out Channel

    # 2. Reshape (모양 변경): [In * H * W, Out]
    # C++: reshape(shape[0] * shape[2] * shape[3], shape[1])
    # 파이썬에서는 -1을 쓰면 자동으로 계산해주지만, C++ 코드와 똑같이 명시하면 아래와 같습니다.
    return w_t.reshape(I * H * W, O).astype(np.float32, copy=True)

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
                
                if "trans" in key or "up" in key or "deconv" in key:
                    print(f" [ConvTranspose 변환] {key}")
                    np_converted = conv_transpose_to_vkB(value)
                else:
                    np_converted = conv_to_vkB(value)
            
                clean_state_dict[key] = _tn(np_converted)
                converted_count += 1
            
            # 3-2. Linear Weight 감지 (2차원 + 이름에 weight 포함)
            elif value.ndim == 2 and "weight" in key:
                # [Out, In] -> [In, Out] 변환
                np_converted = linear_to_vkB(value)
                clean_state_dict[key] = _tn(np_converted)
                converted_count += 1
                print(f" [Linear 변환] {key}: {value.shape} -> {clean_state_dict[key].shape}")

            # 3-3. 그 외 (Bias, BatchNorm 등) -> 그대로 저장
            else:
                if "num_batches_tracked" in key:
                    continue

                # Bias, BN Running Mean/Var 등은 1D이지만 float32로 통일해야 합니다.
                np_converted = _to_cpu_np(value) # CPU로 이동 & Numpy 변환
                target_tensor = _tn(np_converted) # Float32 & Contiguous 강제

                # 딕셔너리에 저장
                clean_state_dict[key] = target_tensor
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
    convert_to_safetensors(os.path.join(ROOT_PATH,INPUT_PTH), os.path.join(ROOT_PATH,OUTPUT_SAFE))

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from utils import save_tensor_bin

from models.unet import Unet

def run_inference(pth_path, image_path, in_channels=1, img_size=(256, 256)):
    """
    pth_path: 저장된 모델 파일 경로 (.pth)
    image_path: 테스트할 이미지 경로
    in_channels: 모델 입력 채널 수 (1: 흑백, 3: 컬러)
    img_size: 모델에 넣을 이미지 크기 (width, height)
    
    """
    # 1. 장치 설정 (GPU 사용 가능 시 GPU, 아니면 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 모델 초기화 및 가중치 로드
    model = Unet(in_channels=in_channels, out_channels=1).to(device)
    
    if os.path.exists(pth_path):
        checkpoint = torch.load(pth_path, map_location=device)
        
        # 저장할 때 'model_state_dict' 키에 저장했으므로 해당 키를 불러옴
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {pth_path} (Epoch: {checkpoint.get('epoch', 'Unknown')})")
        else:
            # 만약 state_dict만 바로 저장된 경우에 대한 대비
            model.load_state_dict(checkpoint)
            print(f"Model loaded from {pth_path} (Direct State Dict)")
    else:
        print("Error: Model file not found!")
        return

    # 3. 모델을 평가 모드로 전환 (Dropout, BatchNorm 등의 동작 고정)
    model.eval()

    SAVE_DIR = "./debug_layers/"  # 저장할 폴더
    os.makedirs(SAVE_DIR, exist_ok=True)
    hook_handles = []

    def get_save_hook(name):
        """레이어의 출력을 가로채서 bin 파일로 저장하는 Hook 함수"""
        def hook(model, input, output):
            file_name = f"{name}.bin"
            save_path = os.path.join(SAVE_DIR, file_name)
            save_tensor_bin(save_path, output)
        return hook

    print("\n=== Registering Hooks for Layer Outputs ===")
    # 모델의 직계 자식 모듈(encoderConv1_, pool1_ 등)에 훅 등록
    for name, module in model.named_children():
        print(f" -> Hook registered: {name}")
        handle = module.register_forward_hook(get_save_hook(name))
        hook_handles.append(handle)
    print("===========================================\n")
    # ---------------------------------------------------------


    # 4. 이미지 전처리
    if in_channels == 1:
        # 흑백으로 읽기
        original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        # 컬러로 읽기 (BGR -> RGB 변환)
        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    if original_img is None:
        print("Error: Image not found!")
        return
    
    # 정규화 (0~255 -> 0.0~1.0)
    img = original_img.astype(np.float32) / 255.0

    tensor = torch.from_numpy(img)
    
    save_tensor_bin('image.bin', tensor)

    # 텐서 변환 및 차원 추가
    if in_channels == 1:
        # [H, W] -> [H, W, 1] -> [1, H, W] (Transpose)
        img = np.expand_dims(img, axis=2) 
    
    # numpy [H, W, C] -> tensor [C, H, W]
    img = img.transpose((2, 0, 1)) 
    
    # Tensor로 변환 및 배치 차원 추가 [1, C, H, W]
    input_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    # 5. 추론 실행
    with torch.no_grad(): # 그라디언트 계산 비활성화 (메모리 절약)
        output = model(input_tensor)
        
        # 모델의 마지막 레이어에 활성화 함수가 없으므로 Sigmoid 적용
        # (Binary Segmentation 가정)
        prediction = torch.sigmoid(output)
        
        # 임계값 적용 (0.5 이상이면 1, 아니면 0) - 필요시 사용
        # mask = (prediction > 0.5).float()
        mask = prediction


    for handle in hook_handles:
        handle.remove()
    print("\n=== Hooks Removed & Inference Done ===\n")
    # 6. 결과 시각화 (Tensor -> Numpy 변환)
    # [1, 1, H, W] -> [H, W]
    pred_mask = mask.squeeze().cpu().numpy()
    
    plt.figure(figsize=(10, 5))
    
    # 원본 이미지
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    if in_channels == 1:
        plt.imshow(cv2.resize(original_img, img_size), cmap='gray')
    else:
        plt.imshow(cv2.resize(original_img, img_size))
    
    # 예측 결과
    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask, cmap='gray') # 마스크는 보통 흑백으로 표현
    
    plt.show()
    plt.waitforbuttonpress()

# --- 실행 예시 ---
# 실제 파일 경로로 수정해서 실행하세요.
# 예: CHECK_POINT_DIR/epoch_0010/unet.pth
PTH_FILE = "D:/VAI/weight/unet.pth" 
IMG_FILE = "D:/VAI/images/image.png"

if __name__ == "__main__":
    run_inference(PTH_FILE, IMG_FILE, in_channels=3, img_size=(512, 512))
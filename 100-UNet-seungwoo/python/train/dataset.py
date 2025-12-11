import torch
import torch.utils.data as data

import os
import numpy as np

import cv2

DATASET = "data"
LABEL =  "label"

def opencv_loader(path, channel : int = 3, isRGB : bool = True):

    data : np.ndarray = None

    if channel == 3:
        data = cv2.imread(path, cv2.IMREAD_COLOR)
    elif channel == 1:
        data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if isRGB:
        data = cv2.cvtColor(data,cv2.COLOR_BAYER_BG2RGB)

    return data

def ResizeTranform(x : np.ndarray, size:tuple[int , int])-> np.ndarray:
    return cv2.resize(x, size)

def NormTranform(x : np.ndarray)-> np.ndarray:
    return x.astype(np.float32) / 255.0

def np_to_tensor(img: np.ndarray) -> torch.Tensor:
    
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(img)}")

    # 단일채널이면 (H,W,1) 형태로 확장
    if img.ndim == 2:
        img = img[..., None]  # (H,W,1)
    elif img.ndim != 3:
        raise ValueError(f"Invalid image shape {img.shape}, expected 2D or 3D")

    # NumPy → Tensor 변환
    tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()
    return tensor

class CustomDataSet(data.Dataset):
    def __init__(self, splits_file_path, dataset_dir, height, width, is_train : bool):

        self.splits_file_path = splits_file_path
        self.dataset_dir = dataset_dir

        self.height = height
        self.width = width

        self.loader = opencv_loader

    def __len__(self): pass

    def __getitem__(self, idx):
        data = self.loader()
        label = self.loader()

        data, label = self.preprocess(data, label)

        g_data = np_to_tensor(data)
        g_label = np_to_tensor(label)

        return g_data, g_label

    def preprocess(self, data : np.ndarray, label : np.ndarray)-> tuple[np.ndarray, np.ndarray]:
        data = ResizeTranform(data, (self.height,self.width))
        label = ResizeTranform(label, (self.height,self.width))

        data = NormTranform(data)
        
        return data , label
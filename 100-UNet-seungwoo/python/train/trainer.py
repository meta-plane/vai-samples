import torch
from torch.utils.data import dataloader 

from unet import Unet
from dataset import CustomDataSet


class Trainer():
    def __init__(self):
        self.device = "cuda"
        self.in_ch = 3
        self.out_ch = 1
        self.epoch = 50
        self.batch_size = 8
        self.height = 512
        self.width = 512
        self.splits_file_path = "/splits.txt"
        self.dataset_dir = "../../"
        
        self.model = Unet(in_channels = self.in_ch, out_channels = self.out_ch)
        self.model.to(self.device)

        self.dataset = CustomDataSet(
            self.splits_file_pat,
            self.dataset_dir
            self.height,
            self.width,
            True
            )
        
        self.dataloader = dataloader()

def train(self):
    process_epoch()

def process_epoch(self):
    for epoch in range(self.epoch):
        
        loss = process_batch()

        loss.backward()


    





def process_batch()->torch.Tensor:
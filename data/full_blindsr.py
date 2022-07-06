
import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
import os
import torch
from data.dataset_blindsr import DatasetBlindSR
from data.xgaze_blindsr import XGazeBlindSR


class FullBlindSR(data.Dataset):
    def __init__(self, opt) -> None:
        super(FullBlindSR).__init__()
        self.datasets = []
        self.xgaze_dataset = XGazeBlindSR(opt)
        self.datasets.append(self.xgaze_dataset)
        self.gt_folders = opt["dataroot_gts"]
        for id in range(len(self.gt_folders)):
            self.datasets.append(DatasetBlindSR(opt, dataset_id = id))
        self.dataset_num = len(self.datasets)
        self.len = 0
        for dataset in self.datasets:
            self.len = max(len(dataset), self.len)
    def __getitem__(self, index):
        i = random.randint(0,self.dataset_num - 1)
        if index < int(self.len / len(self.datasets[i])) * len(self.datasets[i]):
            return self.datasets[i][index % len(self.datasets[i])]
        else:
            return self.datasets[i][random.randint(0,len(self.datasets[i]) - 1)]
    def __len__(self):
        return self.len
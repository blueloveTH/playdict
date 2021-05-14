import torch
from torch.utils.data import Dataset
import cv2
import torchvision.transforms.functional as tF
import pandas as pd
import numpy as np
import torch.nn as nn
import torch

class Binarizer(nn.Module):
    """See reference: https://zhuanlan.zhihu.com/p/209529856"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        h, w = x.size(-2), x.size(-1)
        g_list = torch.zeros([255], device=x.device)

        for t in range(0, 255, 2):
            n0 = x[x < t]               # 0 pixels
            n1 = x[x >= t]              # 1 pixels
            w0 = len(n0) / (h * w)      # 0 pixel weight
            w1 = len(n1) / (h * w)      # 1 pixel weight

            u0 = n0.float().mean() if len(n0) > 0 else torch.tensor(0, device=x.device)
            u1 = n1.float().mean() if len(n1) > 0 else torch.tensor(0, device=x.device)

            g_list[t] = w0 * w1 * (u0 - u1) ** 2

        best_t = g_list.argmax(0)

        x[x < best_t] = 0
        x[x >= best_t] = 255
        return x

binarizer = torch.jit.script(Binarizer())


def transform_grayscale_image(img, img_size):
    img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_NEAREST)
    img = torch.from_numpy(img)
    return img.unsqueeze_(0)

class TrainDataset(Dataset):
    def __init__(self, raw_dataset, max_dec_len, tokenizer, img_size):
        super().__init__()
        self.raw_dataset = raw_dataset
        self.img_size = img_size
        self.tokenizer = tokenizer
        self.max_dec_len = max_dec_len
    
    def __len__(self):
        return len(self.raw_dataset)
    
    def __getitem__(self, i):
        img, tgt = self.raw_dataset[i]
        tgt = self.tokenizer.string_to_indices(tgt, dtype='int64')
        image = transform_grayscale_image(img, self.img_size)
        return image, torch.tensor(len(tgt)), np.pad(tgt, (0, self.max_dec_len-len(tgt)))

class PartitionedTrainDataset(TrainDataset):
    def __init__(self, file_list, cnt_list, max_dec_len, tokenizer, img_size):
        super().__init__(None, max_dec_len, tokenizer, img_size)
        self.file_list_backup = file_list
        self.cnt_list_backup = cnt_list
        self.reset()
        self.raw_dataset = pd.read_pickle(self.file_list[0])

    def __len__(self):
        return sum(self.cnt_list_backup)

    def reset(self):
        self.file_list = self.file_list_backup.copy()
        self.cnt_list = self.cnt_list_backup.copy()

    def __getitem__(self, i):
        if self.cnt_list[0] == 0:
            self.cnt_list.pop(0)
            self.file_list.pop(0)

            if len(self.cnt_list) == 0:
                self.reset()
            self.raw_dataset = pd.read_pickle(self.file_list[0])
            
        self.cnt_list[0] -= 1
        return super().__getitem__(i)



class TestDataset(Dataset):
    def __init__(self, raw_dataset, max_dec_len, tokenizer, img_size):
        super().__init__()
        self.raw_dataset = raw_dataset
        self.img_size = img_size
        self.tokenizer = tokenizer
        self.max_dec_len = max_dec_len
    
    def __len__(self):
        return len(self.raw_dataset)
    
    def __getitem__(self, i):
        img, tgt = self.raw_dataset[i]
        tgt = self.tokenizer.string_to_indices(tgt, dtype='int64')
        image = transform_grayscale_image(img, self.img_size)
        return image
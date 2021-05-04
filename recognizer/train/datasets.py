import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import torchvision.transforms.functional as tF
import pickle
import pandas as pd

'''
def load_image(path, img_size, fn=None):
    img = cv2.imread(path)
    if fn is not None:
        img = fn(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_NEAREST)
    img = torch.from_numpy(img).float() / 255
    img = img.permute(2, 0, 1)
    img = tF.normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), inplace=True)
    return img
'''

def transform_grayscale_image(img, img_size):
    img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_NEAREST)
    img = torch.from_numpy(img).float() / 255
    img = (img - 0.449) / 0.226
    return img.unsqueeze_(0)

class TrainDataset(Dataset):
    def __init__(self, raw_dataset, tokenizer, img_size):
        super().__init__()
        self.raw_dataset = raw_dataset
        self.img_size = img_size
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.raw_dataset)
    
    def __getitem__(self, i):
        img, tgt = self.raw_dataset[i]
        tgt = self.tokenizer.string_to_indices(tgt, dtype='int64')
        image = transform_grayscale_image(img, self.img_size)
        return image, tgt, len(tgt)

class PartitionedTrainDataset(TrainDataset):
    def __init__(self, file_list, cnt_list, tokenizer, img_size):
        super().__init__(None, tokenizer, img_size)
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
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()
    
    def __getitem__(self, i):
        raise NotImplementedError()
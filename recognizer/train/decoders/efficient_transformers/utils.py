import numpy as np
import torch

def upper_triangular_mask(seq_len, device='cpu'):
    mask = torch.ones([seq_len, seq_len], dtype=bool, device=device)
    return ~mask.tril_()

def get_sinusoid_table(seq_len, d_model):
    def get_angle(pos, i, d_model):
        return pos / np.power(10000, (2 * (i//2)) / d_model)
        
    sinusoid_table = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(d_model):
            if i % 2 == 0:
                sinusoid_table[pos, i] = np.sin(get_angle(pos, i, d_model))
            else:
                sinusoid_table[pos, i] = np.cos(get_angle(pos, i, d_model))

    return torch.FloatTensor(sinusoid_table)

def left_pad(a, pad_count):
    pad_array = np.zeros([pad_count] + list(a.shape[1:]), dtype=a.dtype)
    return np.r_[pad_array, a]
import torch
import torch.nn as nn


def ace_loss(input, label):
    bs, h, w = input.size()
    T_ = h, w

    input = input.view(bs, T_)
    input = input + 1e-10

    label[:,0] = T_ - label[:,0]

    # ACE Implementation (four fundamental formulas)
    input = torch.sum(input,1)
    input = input/T_
    label = label/T_
    loss = (-torch.sum(torch.log(input)*label)).mean()

    return loss
import torch.nn as nn
from ._mha import MultiheadAttention
from keras4torch.activations import _create_activation

class TransformerBlockBase(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def create_ffn(vdim, forward_expansion, dropout=0.0, activation='relu'):
        layers = [
            nn.Linear(vdim, forward_expansion*vdim),
            _create_activation(activation),
            nn.Linear(forward_expansion*vdim, vdim)
            ]
        if dropout > 1e-3:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    @staticmethod
    def create_ln_dropout(vdim, dropout=0.0):
        layers = [nn.LayerNorm(vdim)]
        if dropout > 1e-3:
            layers.append(nn.Dropout(dropout))
        if len(layers) > 1:
            return nn.Sequential(*layers)
        return layers[0]


class TransformerBlock(TransformerBlockBase):
    def __init__(self, kdim, vdim, n_heads, dropout, forward_expansion):
        super().__init__()

        self.attn = MultiheadAttention(kdim=kdim, vdim=vdim, n_heads=n_heads, dropout=dropout)
        self.ln_dropout_attn = self.create_ln_dropout(vdim, dropout)

        self.ffn = self.create_ffn(vdim, forward_expansion, dropout)
        self.ln_dropout_ffn = self.create_ln_dropout(vdim, dropout)

    def forward(self, Q, K, V, attn_mask=None):
        sub_out_0 = self.attn(Q, K, V, attn_mask=attn_mask)
        sub_out_0 = self.ln_dropout_attn(sub_out_0 + V)

        sub_out_1 = self.ffn(sub_out_0)
        sub_out_1 = self.ln_dropout_ffn(sub_out_1 + sub_out_0)
        return sub_out_1

class SequentialBlocks(nn.Module):
    def __init__(self, *list):
        super(SequentialBlocks, self).__init__()

        for i in list:
            assert isinstance(i, TransformerBlockBase)
            
        self.first_block = list[0]
        self.blocks = nn.ModuleList(list[1:])

    def forward(self, Q, K, V, *args, **kwargs):
        x = self.first_block(Q, K, V, *args, **kwargs)
        del Q, K, V
        for block in self.blocks:
            x = block(x, x, x, *args, **kwargs)
        return x

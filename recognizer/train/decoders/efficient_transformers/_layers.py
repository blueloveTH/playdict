import torch.nn as nn
from ._mha import MultiheadAttention
from keras4torch.activations import _create_activation
from copy import deepcopy

def create_ffn(vdim, forward_expansion, dropout=0.0, activation='relu'):
    layers = [
        nn.Linear(vdim, forward_expansion*vdim),
        _create_activation(activation)
        ]

    if dropout > 1e-3:
        layers.append(nn.Dropout(dropout))

    layers.append(nn.Linear(forward_expansion*vdim, vdim))
    return nn.Sequential(*layers)


class SkipConnectionLN(nn.Module):
    def __init__(self, vdim, dropout=0.0):
        super().__init__()

        self.ln = nn.LayerNorm(vdim)
        self.dropout = nn.Dropout(dropout) if dropout > 1e-3 else (lambda x: x)

    def forward(self, x, skip_connect_x):
        x += self.dropout(skip_connect_x)
        return self.ln(x)


def get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


class TransformerEncoderLayer(nn.Module):
    def __init__(self, kdim, vdim, n_heads, dropout=0.1, forward_expansion=2):
        super().__init__()

        self.attn = MultiheadAttention(kdim=kdim, vdim=vdim, n_heads=n_heads, dropout=dropout)
        self.skip_c_0 = SkipConnectionLN(vdim, dropout)

        self.ffn = create_ffn(vdim, forward_expansion, dropout)
        self.skip_c_1 = SkipConnectionLN(vdim, dropout)

    def forward(self, Q, K, V, attn_mask=None):
        sub_output = self.attn(Q, K, V, attn_mask=attn_mask)
        del Q, K
        sub_output = self.skip_c_0(sub_output, V)
        return self.skip_c_1(self.ffn(sub_output), sub_output)

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: TransformerEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)

    def forward(self, Q, K, V, *args, **kwargs):
        x = self.layers[0](Q, K, V, *args, **kwargs)
        del Q, K, V
        for layer in self.layers[1:]:
            x = layer(x, x, x, *args, **kwargs)
        return x



class TransformerDecoderLayer(nn.Module):
    def __init__(self, kdim, vdim, n_heads, dropout=0.1, forward_expansion=2):
        super().__init__()

        self.attn_0 = MultiheadAttention(kdim=kdim, vdim=vdim, n_heads=n_heads, dropout=dropout)
        self.skip_c_0 = SkipConnectionLN(vdim, dropout)

        self.attn_1 = MultiheadAttention(kdim=kdim, vdim=vdim, n_heads=n_heads, dropout=dropout)
        self.skip_c_1 = SkipConnectionLN(vdim, dropout)

        self.ffn = create_ffn(vdim, forward_expansion, dropout)
        self.skip_c_2 = SkipConnectionLN(vdim, dropout)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        tmp = self.attn_0(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = self.skip_c_0(tmp, tgt)

        tmp = self.attn_1(tgt, memory, memory)
        tgt = self.skip_c_1(tmp, tgt)

        tmp = self.ffn(tgt)
        return self.skip_c_2(tmp, tgt)


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: TransformerDecoderLayer, num_layers: int):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask, tgt_key_padding_mask)
        return output
import torch
import torch.nn as nn
import keras4torch as k4t
#from .efficient_transformers import TransformerEncoderLayer, TransformerEncoder
from .efficient_transformers._layers import *
from .efficient_transformers.utils import get_sinusoid_table

class TransformerEncoderLayer2(nn.Module):
    def __init__(self, kdim, vdim, n_heads, dropout=0.1, forward_expansion=2):
        super().__init__()

        self.attn = MultiheadAttention(kdim=kdim, vdim=vdim, n_heads=n_heads, dropout=dropout)
        self.skip_c_0 = SkipConnectionLN(vdim, dropout)

        self.ffn = create_ffn(vdim, forward_expansion, dropout)
        self.skip_c_1 = SkipConnectionLN(vdim, dropout)

    def forward(self, x):
        x = self.skip_c_0(self.attn(x, x, x), x)
        x = self.skip_c_1(self.ffn(x), x)
        return x


class NATDecoder(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, vocab_size, max_dec_len, num_pixels):
        super().__init__()

        self.transformer = nn.Sequential(
            TransformerEncoderLayer2(kdim=decoder_dim, vdim=decoder_dim, n_heads=8, dropout=0.0),
            TransformerEncoderLayer2(kdim=decoder_dim, vdim=decoder_dim, n_heads=8, dropout=0.0),
            TransformerEncoderLayer2(kdim=decoder_dim, vdim=decoder_dim, n_heads=8, dropout=0.0),
            #TransformerEncoderLayer2(kdim=decoder_dim, vdim=decoder_dim, n_heads=8, dropout=0.0),
        )

        self.pre_fc = nn.Linear(encoder_dim, decoder_dim)
        self.length_fc = nn.Linear(num_pixels, max_dec_len)
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def forward(self, x):
        # x: [None, num_pixels, encoder_dim]
        x = self.pre_fc(x)              # [bs, num_pixels, decoder_dim]

        x = x.transpose(-1, -2)         # [bs, decoder_dim, num_pixels]
        x = self.length_fc(x)           # [bs, decoder_dim, max_dec_len]
        x = x.transpose(-1, -2)         # [bs, max_dec_len, decoder_dim]

        x = self.transformer(x)         # [bs, num_pixels, decoder_dim]

        return self.fc(x)               # [bs, max_dec_len, vocab_size]


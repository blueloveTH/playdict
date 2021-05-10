import torch
import torch.nn as nn
import keras4torch as k4t
from .efficient_transformers import TransformerEncoderLayer, TransformerEncoder

class NATDecoder(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, vocab_size, max_dec_len):
        super().__init__()
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(kdim=decoder_dim, vdim=decoder_dim, n_heads=8, dropout=0.0), 4)
        self.pre_fc = nn.Sequential(
            nn.Linear(encoder_dim, decoder_dim)                 # [bs, num_pixels, decoder_dim]
        )

        self.post_fc = nn.Sequential(
            k4t.layers.Lambda(lambda x: x.transpose(-1, -2)),   # [bs, decoder_dim, num_pixels]
            k4t.layers.Linear(max_dec_len),                     # [bs, decoder_dim, max_dec_len]
            k4t.layers.Lambda(lambda x: x.transpose(-1, -2)),   # [bs, max_dec_len, decoder_dim]
            nn.Linear(decoder_dim, vocab_size),                 # [bs, max_dec_len, vocab_size]
        )
        

    def forward(self, x):
        # x: [None, num_pixels, encoder_dim]
        x = self.pre_fc(x)              # [bs, num_pixels, decoder_dim]
        x = self.transformer(x, x, x)   # [bs, num_pixels, decoder_dim]
        x = self.post_fc(x)             # [bs, max_dec_len, vocab_size]
        return x


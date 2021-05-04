import torch.nn.functional as F
import torch
import torch.nn as nn

from easyocr_model.modules import VGG_FeatureExtractor, ResNet_FeatureExtractor

class Encoder(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.cnn = ResNet_FeatureExtractor(input_channel=1, output_channel=CFG.encoder_dim)

    def forward(self, x):
        return self.cnn(x)


class Attention(nn.Module):
    def __init__(self, memory_dim, hidden_dim, attn_dim):
        super().__init__()
        self.encoder_att = nn.Linear(memory_dim, attn_dim)
        self.decoder_att = nn.Linear(hidden_dim, attn_dim)
        self.full_att = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(attn_dim, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, memory, h):
        # memory: [None, num_pixels, memory_dim]
        # h: [None, hidden_dim]
        att1 = self.encoder_att(memory)                     # [bs, num_pixels, attn_dim]
        h = self.decoder_att(h).unsqueeze(1)                # [bs, 1, attn_dim]
        alpha = self.full_att(att1 + h)                     # [bs, num_pixels, 1]
        return (memory * alpha).sum(dim=1)                  # [bs, memory_dim]


class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, max_dec_len, encoder_dim, dropout):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.max_dec_len = max_dec_len

        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        self.fc_gate = nn.Sequential(
            nn.Linear(decoder_dim, encoder_dim),
            nn.Sigmoid()
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(decoder_dim, vocab_size),
        )

    def init_hidden_state(self, memory):
        memory = memory.mean(dim=1)         # [bs, encoder_dim]
        h = self.init_h(memory)             # [bs, decoder_dim]
        c = self.init_c(memory)             # [bs, decoder_dim]
        return h, c

    def forward(self, tgt, memory, tgt_lens):
        # tgt: [None, ?]   int64
        # memory: [None, num_pixels, encoder_dim]
        # tgt_lens: [None]  int64
        batch_size = memory.size(0)
        tgt = self.embedding(tgt)  # [bs, max_tgt_len, embed_dim]

        h, c = self.init_hidden_state(memory)  # (batch_size, decoder_dim)

        output = torch.zeros(batch_size, tgt_lens[0], self.vocab_size, device=memory.device)
        output[:, 0, 1] = 1.0
        
        rng = torch.empty(tgt_lens[0], dtype=torch.long, device=memory.device)
        for i in range(1, len(tgt_lens)):
            rng[tgt_lens[i]: tgt_lens[i-1]] = i
        rng[1: tgt_lens[-1]] = len(tgt_lens)

        for t in range(1, tgt_lens[0]):
            bs_t = rng[t]

            memory_with_attn = self.attention(memory[:bs_t], h[:bs_t])
            memory_with_attn *= self.fc_gate(h[:bs_t])          # [bs, encoder_dim]

            h, c = self.decode_step(
                torch.cat([tgt[:bs_t, t-1], memory_with_attn], dim=1),
                (h[:bs_t], c[:bs_t])
            )

            output[:bs_t, t] = self.fc(h)  # (batch_size_t, vocab_size)
        return output
    
    def predict(self, memory):
        batch_size = memory.size(0)

        curr_token = torch.ones(batch_size, dtype=torch.long, device=memory.device)
        curr_token = self.embedding(curr_token)  # [bs, embed_dim]

        h, c = self.init_hidden_state(memory)  # (batch_size, decoder_dim)

        output = torch.zeros(batch_size, self.max_dec_len, self.vocab_size, device=memory.device)
        output[:, 0, 1] = 1.0
        
        for t in range(1, self.max_dec_len):
            memory_with_attn = self.attention(memory, h)
            memory_with_attn *= self.fc_gate(h)          # [bs, memory_dim]

            h, c = self.decode_step(
                torch.cat([curr_token, memory_with_attn], dim=1),
                (h, c)
            )

            curr_pred = self.fc(h).detach()
            output[:, t] = curr_pred

            curr_token = self.embedding(curr_pred.argmax(-1))
        return output


class EncoderDecoderModel(nn.Module):
    def __init__(self, CFG, tokenizer):
        super().__init__()

        self.encoder = Encoder(CFG)
        self.decoder = DecoderWithAttention(attention_dim=CFG.attention_dim,
                                    embed_dim=CFG.embed_dim,
                                    decoder_dim=CFG.decoder_dim,
                                    vocab_size=tokenizer.vocab_size,
                                    encoder_dim=CFG.encoder_dim,
                                    dropout=CFG.dropout,
                                    max_dec_len=CFG.max_dec_len)

    def forward(self, src, tgt=None, tgt_lens=None):
        memory = self.encoder(src)
        if tgt is None:
            return self.decoder.predict(memory)
        return self.decoder(tgt, memory, tgt_lens=tgt_lens)
        
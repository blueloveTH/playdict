from numpy import fabs
import torch.nn.functional as F
import torch
import torch.nn as nn

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


class DecodeSteps(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.cells = nn.ModuleList([nn.LSTMCell(input_size, hidden_size) for _ in range(num_layers)])

    def forward(self, x, t):
        h, c = t
        for cell in self.cells:
            h, c = cell(x, (h, c))
        return h, c
        


class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, max_dec_len, encoder_dim):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.max_dec_len = max_dec_len

        #self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        self.decode_step = DecodeSteps(embed_dim + encoder_dim, decoder_dim, num_layers=2)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        self.fc_gate = nn.Sequential(
            nn.Linear(decoder_dim, encoder_dim),
            nn.Sigmoid()
        )

        self.fc = nn.Linear(decoder_dim, vocab_size)

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


def count_params(module):
    return sum([p.numel() for p in module.parameters()])

class EncoderDecoderModel(nn.Module):
    def __init__(self, encoder, CFG, tokenizer):
        super().__init__()

        self.encoder = encoder
        self.decoder = DecoderWithAttention(attention_dim=CFG.attention_dim,
                                    embed_dim=CFG.embed_dim,
                                    decoder_dim=CFG.decoder_dim,
                                    vocab_size=tokenizer.vocab_size,
                                    encoder_dim=CFG.encoder_dim,
                                    max_dec_len=CFG.max_dec_len)
        print('Encoder:', count_params(self.encoder))
        print('Decoder:', count_params(self.decoder))

        self.deploy_mode = False;

    def deploy(self):
        self.deploy_mode = True

    def forward(self, src, tgt=None, tgt_lens=None):
        if(self.deploy_mode):
            return self.predict_for_deploy(src)

        memory = self.encoder(src)              # NCHW

        memory = memory.permute(0, 3, 1, 2)     # NWCH
        memory = memory.mean(dim=-1)            # NWC

        if tgt is None:
            return self.decoder.predict(memory)
        return self.decoder(tgt, memory, tgt_lens=tgt_lens)

    def predict_for_deploy(self, src):
        memory = self.encoder(src)              # NCHW

        memory = memory.permute(0, 3, 1, 2)     # NWCH
        memory = memory.mean(dim=-1)            # NWC

        logits = self.decoder.predict(memory)
        return logits.argmax(-1)
        
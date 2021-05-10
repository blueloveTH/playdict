import torch.nn as nn 
import torch

def softmax(x, dim):
    x = torch.exp(x - x.max(dim=dim, keepdim=True).values)
    s = x.sum(dim=dim, keepdim=True)
    return x / (s + 1e-5)

class _MHABase(nn.Module):
    def __init__(self, kdim, vdim, n_heads, share_wqk=False):
        super(_MHABase, self).__init__()

        self.n_heads = n_heads

        assert kdim % n_heads == 0
        assert vdim % n_heads == 0

        self.d_k = kdim // n_heads
        self.d_v = vdim // n_heads

        self.WQ = nn.Linear(kdim, kdim)
        self.WK = self.WQ if share_wqk else nn.Linear(kdim, kdim)
        self.WV = nn.Linear(vdim, vdim)
        
        self.final_fc = nn.Linear(n_heads * self.d_v, vdim)

    def in_projection(self, Q, K, V):
        seq_len = V.size(1)
        q_heads = self.WQ(Q).view(-1, seq_len, self.n_heads, self.d_k).transpose(1, 2) 
        k_heads = self.WK(K).view(-1, seq_len, self.n_heads, self.d_k).transpose(1, 2) 
        v_heads = self.WV(V).view(-1, seq_len, self.n_heads, self.d_v).transpose(1, 2) 
        # |q_heads| : (batch_size, n_heads, seq_len, d_k)
        # |k_heads| : (batch_size, n_heads, seq_len, d_k)
        # |v_heads| : (batch_size, n_heads, seq_len, d_v)
        return q_heads, k_heads, v_heads

    def out_projection(self, attn_output):
        attn_output.transpose_(1, 2)                        # [bs, seq_len, n_heads, d_v]
        attn_output = attn_output.reshape(*attn_output.shape[:2], -1)
        # |attn_output| : (batch_size, seq_len, n_heads * d_v)
        return self.final_fc(attn_output)                   # [batch_size, seq_len, vdim]



class ScaledDotProductAttention(nn.Module):
    neg_inf = -30000.0

    def __init__(self, d_k, dropout):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = 1 / (d_k ** 0.5)
        self.dropout = nn.Dropout(dropout) if dropout > 1e-3 else (lambda x: x)
    
    def forward(self, q, k, v, attn_mask=None):
        # |q| : (batch_size, n_heads, seq_len, d_k)
        # |k| : (batch_size, n_heads, seq_len, d_k)
        # |v| : (batch_size, n_heads, seq_len, d_v)
        # |attn_mask| : (batch_size, seq_len, seq_len) or (seq_len, seq_len)

        attn_score = q @ k.transpose(-1, -2) * self.scale_factor

        if attn_mask is not None:
            attn_score.masked_fill_(attn_mask, self.neg_inf)
        # |attn_score| : (batch_size, n_heads, seq_len, seq_len)

        attn_score = torch.softmax(attn_score, dim=-1)
        attn_score = self.dropout(attn_score)
        
        output = attn_score @ v
        # |output| : (batch_size, n_heads, seq_len, d_v)

        return output, attn_score



class MultiheadAttention(_MHABase):
    def __init__(self, kdim, vdim, n_heads, dropout=0.0, share_wqk=False):
        super().__init__(kdim, vdim, n_heads, share_wqk=share_wqk)

        self.attn = ScaledDotProductAttention(self.d_k, dropout=dropout) 

    def forward(self, Q, K, V, attn_mask=None, need_weights=False):
        # |Q| : (batch_size, seq_len, kdim)
        # |K| : (batch_size, seq_len, kdim)
        # |V| : (batch_size, seq_len, vdim)
        # |attn_mask| : (seq_len, seq_len) or (batch_size, seq_len, seq_len)

        q_heads, k_heads, v_heads = self.in_projection(Q, K, V)
        
        attn, attn_weights = self.attn(q_heads, k_heads, v_heads, attn_mask)

        output = self.out_projection(attn)

        if need_weights:
            return output, attn_weights
        else:
            return output

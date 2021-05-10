import torch.nn.functional as F
from ._mha import _MHABase
from collections import OrderedDict
import torch
import torch.nn as nn

def causal_linear_attn(q, k, v, actv_q, actv_k, key_padding_mask):
    q = actv_q(q)                   # [bs, n_heads, seq_len, d_k]
    k = actv_k(k)                   # [bs, n_heads, seq_len, d_k]

    k = k.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(-1), 0)

    q = q.unsqueeze(-2)             # [bs, n_heads, seq_len, 1, d_k]
    k = k.unsqueeze(-1)             # [bs, n_heads, seq_len, d_k, 1]
    v = v.unsqueeze(-2)             # [bs, n_heads, seq_len, 1, d_v]

    S = k @ v                       # [bs, n_heads, seq_len, d_k, d_v]
    S = S.cumsum(-3)                # [bs, n_heads, seq_len, d_k, d_v]
    z = k.cumsum(-3)                # [bs, n_heads, seq_len, d_k, 1]

    norm = (q @ z)                  # [bs, n_heads, seq_len, 1, 1]

    result = (q @ S) / (norm + 1e-5)
    return result.squeeze(-2)


def adaptive_causal_linear_attn_2(q, k, v, actv_q, actv_k, size, key_padding_mask):
    q = actv_q(q)
    k = actv_k(k)
    attn_score = q @ k.transpose(-1, -2)

    seq_len = v.size(-2)
    n_windows = seq_len // size

    mask = torch.ones([seq_len, seq_len], dtype=torch.bool, device=v.device)
    for i in range(n_windows):
        start = i * size
        end = (i + 1) * size
        mask[start:end, start:end].triu_(1)

    attn_score.masked_fill_(mask, 0)
    attn_score.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), 0)

    attn_score /= (attn_score.sum(dim=-1, keepdim=True) + 1e-5)

    return attn_score @ v



def adaptive_causal_linear_attn(q, k, v, actv_q, actv_k, size, key_padding_mask):
    q = actv_q(q)                   # [bs, n_heads, seq_len, d_k]
    k = actv_k(k)                   # [bs, n_heads, seq_len, d_k]
    
    k = k.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(-1), 0)

    q = q.unsqueeze(-2)             # [bs, n_heads, seq_len, 1, d_k]
    k = k.unsqueeze(-1)             # [bs, n_heads, seq_len, d_k, 1]
    v = v.unsqueeze(-2)             # [bs, n_heads, seq_len, 1, d_v]

    S = k @ v                       # [bs, n_heads, seq_len, d_k, d_v]

    bs, n_heads, seq_len, d_k, d_v = S.shape

    S = S.view(bs, n_heads, seq_len//size, size, d_k, d_v)
    z = k.view(bs, n_heads, seq_len//size, size, d_k, 1)

    S = S.cumsum(-3)
    z = z.cumsum(-3)

    S = S.view(bs, n_heads, seq_len, d_k, d_v)
    z = z.view(bs, n_heads, seq_len, d_k, 1)

    norm = (q @ z)                  # [bs, n_heads, seq_len, 1, 1]

    result = (q @ S) / (norm + 1e-5)
    return result.squeeze(-2)


'''
def causal_linear_attn2(q, k, v, actv_q, actv_k):
    q = actv_q(q)                               # [bs, n_heads, seq_len, d_k]
    k = actv_k(k)                               # [bs, n_heads, seq_len, d_k]
    k = k.transpose(-1, -2)                     # [bs, n_heads, d_k, seq_len]
           
    S = 0
    z = 0
    result = torch.empty_like(v)
    for i in range(v.size(-2)):
        qi = q[..., i:i+1, :]                   # [bs, n_heads, 1, d_k]
        kj = k[..., i:i+1]                      # [bs, n_heads, d_k, 1]
        vj = v[..., i:i+1, :]                   # [bs, n_heads, 1, d_v]

        S = S + kj @ vj                         # [bs, n_heads, d_k, d_v]
        z = z + kj                              # [bs, n_heads, d_k, 1]

        attn_i = (qi @ S) / (qi @ z)            # [bs, n_heads, 1, d_v]
        result[..., i:i+1, :] = attn_i

    return result
'''

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
class CausalLinearMHA(_MHABase):
    _activations_dict = OrderedDict({
        'relu': lambda x: F.relu(x, inplace=True),
        'relu6': lambda x: F.relu6(x, inplace=True),
        'elu+1': lambda x: F.elu(x, inplace=True)+1,
        'sigmoid': lambda x: F.sigmoid(x),
        'softmax': lambda x: F.softmax(x, dim=-1),
        'softplus': lambda x: F.softplus(x),
        'ex': lambda x: torch.exp(x)
    })

    @staticmethod
    def _create_activation(i):
        if isinstance(i, str):
            name = i.lower()
            if name not in CausalLinearMHA._activations_dict:
                raise KeyError(f'Invalid name, we support {list(CausalLinearMHA._activations_dict.keys())}.')
            return CausalLinearMHA._activations_dict[name]
        return i

    def __init__(self, kdim, vdim, n_heads, actv_q='softplus', actv_k='sigmoid', share_wqk=False):
        super().__init__(kdim, vdim, n_heads, share_wqk=share_wqk)
        self.actv_q = self._create_activation(actv_q)
        self.actv_k = self._create_activation(actv_k)

    def forward(self, Q, K, V):
        q_heads, k_heads, v_heads = self.in_projection(Q, K, V)

        attn = causal_linear_attn(q_heads, k_heads, v_heads, self.actv_q, self.actv_k)

        return self.out_projection(attn)

class AdaptiveCausalLinearMHA(CausalLinearMHA):
    def __init__(self, kdim, vdim, n_heads, actv_q='softplus', actv_k='sigmoid', share_wqk=False):
        super().__init__(kdim, vdim, n_heads, actv_q=actv_q, actv_k=actv_k, share_wqk=share_wqk)

    def forward(self, Q, K, V, s, key_padding_mask):
        q_heads, k_heads, v_heads = self.in_projection(Q, K, V)

        attn = adaptive_causal_linear_attn(q_heads, k_heads, v_heads, self.actv_q, self.actv_k, s, key_padding_mask)

        return self.out_projection(attn)
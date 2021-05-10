import torch
import torch.nn as nn

@torch.no_grad()
def t_fixup_init_(model, d_model):
    def step1(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            nn.init.zeros_(module.bias.data)
        if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
            nn.init.normal_(module.weight.data, 0, d_model**-0.5)

    def step2(module):
        pass
    
    def step3(module):
        pass

    model.apply(step1)
    model.apply(step2)
    model.apply(step3)


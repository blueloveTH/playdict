import torch
import torch.nn.functional as F
import torch.nn as nn

def count_params(module):
    return sum([p.numel() for p in module.parameters()])

class EncoderDecoderModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.deploy_mode = False
        self.deploy_return_conf = True

    def print_params(self):
        print('Encoder:', count_params(self.encoder))
        print('Decoder:', count_params(self.decoder))

    def deploy(self, return_conf=True):
        self.deploy_mode = True
        self.deploy_return_conf = return_conf
        if hasattr(self.encoder, "switch_to_deplay_in_place"):
            self.encoder.switch_to_deplay_in_place()

    def forward(self, x):
        x = x.float() / 255
        x = (x - 0.449) / 0.226

        x = self.encoder(x)           # NCHW

        x = x.permute(0, 3, 1, 2)     # NWCH
        x = x.mean(dim=-1)            # NWC

        x = self.decoder(x)

        if(self.deploy_mode):
            x = x.softmax(-1)
            y = x.argmax(-1)
            if not self.deploy_return_conf:
                return y
            return y, x.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        return x

        
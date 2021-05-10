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
        self.print_params()

    def print_params(self):
        print('Encoder:', count_params(self.encoder))
        print('Decoder:', count_params(self.decoder))

    def deploy(self):
        self.deploy_mode = True
        if hasattr(self.encoder, "switch_to_deplay_in_place"):
            self.encoder.switch_to_deplay_in_place()

    def forward(self, x):
        x = x.float() / 255
        x = (x - 0.449) / 0.226

        if(self.deploy_mode):
            return self.predict_for_deploy(x)

        x = self.encoder(x)           # NCHW

        x = x.permute(0, 3, 1, 2)     # NWCH
        x = x.mean(dim=-1)            # NWC

        return self.decoder(x)

    def predict_for_deploy(self, x):
        x = self.encoder(x)              # NCHW

        x = x.permute(0, 3, 1, 2)     # NWCH
        x = x.mean(dim=-1)            # NWC

        logits = self.decoder(x)
        return logits.argmax(-1)
        